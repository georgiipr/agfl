#!/bin/bash -l
#SBATCH --job-name=agfl-spatial-continue
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=agfl-spatial-continue-%j.log
#SBATCH --open-mode=append

# Continue the ORIGINAL, unchanged study after its job exits, including timeout:
# sbatch --dependency=afterany:ORIGINAL_JOB_ID eeg_A03_A04_A09_continue.sbatch
# This single file deliberately lives outside agfl/, whose source hash is pinned.
# Upload only this file while the original study runs. No new packages required.
# Completed matching fits are retained; an unfinished fit starts at epoch 1.
# Settings and dataset paths come from the saved accuracy_plan.json.
set -Eeuo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo 'Submit with sbatch from the original cluster AGFL directory.' >&2
    exit 2
fi
cd -- "$SLURM_SUBMIT_DIR"
if [[ -n "${AGFL_PYTHON_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1; then
        echo 'AGFL_PYTHON_MODULE is set but module is unavailable.' >&2
        exit 2
    fi
    set +u
    module load "$AGFL_PYTHON_MODULE"
    set -u
fi
AGFL_ENV_DIR="${AGFL_VENV:-$HOME/.venv}"
if [[ ! -r "$AGFL_ENV_DIR/bin/activate" || ! -x "$AGFL_ENV_DIR/bin/python" ]]; then
    printf 'Missing existing environment: %s\n' "$AGFL_ENV_DIR" >&2
    exit 2
fi
set +u
source "$AGFL_ENV_DIR/bin/activate"
set -u
AGFL_RESULT_PATH="${AGFL_OUTPUT_DIR:-results/eeg-A03-A04-A09-spatial-accuracy-v1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" - "$AGFL_RESULT_PATH" <<'AGFL_CONTINUATION_PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('AGFL requires Python 3.12 or newer')

import copy
import gc
import json
import os
from pathlib import Path
import traceback

import torch
from agfl.accuracy_study import (
    ATTENTIONS, EXPECTED_FITS, MODELS, PRESET, SEEDS, SUBJECTS,
    accuracy_rows, check_electrode_models, write_accuracy_tables,
)
from agfl.config import experiment_identity
from agfl.engine import completed_run, run_experiment
from agfl.reproducibility import provenance
from agfl.session import session_activity, session_paths
from agfl.storage import run_directory, write_json


def read_json(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_source(actual, expected):
    return all(actual.get(k) == expected[k] for k in ('source_sha256', 'packages'))


def continue_study(output_dir):
    session = session_paths(Path(output_dir).expanduser().resolve())
    require((session.report / 'accuracy_plan.json').is_file(),
            'No original accuracy_plan.json. This launcher continues an existing study only.')
    require(session.artifacts.is_dir(), 'The full cluster artifacts folder is required.')
    with session_activity(session.root), run_directory(session.artifacts / '_accuracy_study'):
        plan = read_json(session.report / 'accuracy_plan.json')
        require(plan['preset'] == PRESET and plan['models'] == list(MODELS)
                and plan['subjects'] == list(SUBJECTS) and plan['seeds'] == list(SEEDS)
                and plan['expected_fits'] == EXPECTED_FITS, 'Unexpected original study matrix.')
        current = provenance()
        require(same_source(current, plan['provenance']),
                'Source or package versions changed. Restore the original code/environment; '
                'continuation will not mix old and new implementations.')
        configs = {(c['model'], c['attention'], c['subject_id']): c for c in plan['configs']}
        require(len(configs) == len(plan['configs']) and set(configs) == {
            (m, a, s) for m in MODELS for a in ATTENTIONS for s in SUBJECTS
        }, 'Saved study configurations are missing or duplicated.')
        for config in configs.values():
            require(Path(config['output_dir']).resolve() == session.root
                    and config['seeds'] == list(SEEDS)
                    and config['dataset'] == 'eeg' and config['device'] == 'cuda'
                    and config['training']['save_checkpoints'] is False,
                    'Saved configuration violates the original study contract.')

        records, anchors, paired_protocol = {}, {}, {}
        saved_cells = set()

        def validate_result(result, key):
            require((result['model'], result['attention'], result['subject_id'], result['seed']) == key,
                    'Result has incorrect model/attention/subject/seed.')
            modules = result['attention_modules']
            require(result['token_axis'] == 'electrode' and result['num_tokens'] == 22
                    and modules and result.get('checkpoint_retained') is False
                    and not result.get('synthetic', False)
                    and all(m['token_axis'] == 'electrode' and m['num_tokens'] == 22 for m in modules),
                    'Result violates the electrode/storage study contract.')
            data_key = key[2], key[3]
            identity = result['dataset_fingerprint'], result['split_id']
            require(identity == anchors.setdefault(data_key, identity),
                    'Trials or splits differ across models/attentions.')
            pair = key[0], key[2], key[3]
            require(result['comparison_id'] == paired_protocol.setdefault(pair, result['comparison_id']),
                    'AGFL and MHA have mismatched comparison protocols.')
            accuracy_rows([result])  # Also reject incomplete, nonfinite or invalid scores.

        # Validate saved configurations before changing any report. Incomplete
        # fits also anchor the original dataset/split, but never supply a score.
        for path in sorted((session.artifacts / 'eeg').glob('subject_*/*/seed_*/config.json')):
            saved = read_json(path)
            require(len(saved['seeds']) == 1, f'Invalid per-fit seed configuration: {path}')
            key = saved['model'], saved['attention'], saved['subject_id'], saved['seeds'][0]
            require(key[:3] in configs and key[3] in SEEDS and key not in saved_cells,
                    f'Unexpected or duplicate saved fit: {path}')
            saved_cells.add(key)
            base = {k: v for k, v in saved.items() if k not in {
                'provenance', 'dataset_fingerprint', 'expected_split_id', 'resolved_metadata'
            }}
            base['seeds'] = list(SEEDS)
            require(base == configs[key[:3]] and same_source(saved['provenance'], plan['provenance']),
                    f'Saved fit differs from the original plan/source: {path}')
            expected_dir = (session.artifacts / 'eeg' / f'subject_{key[2]}'
                            / f'{key[0]}-{key[1]}-{experiment_identity(saved)}' / f'seed_{key[3]}')
            require(path.parent == expected_dir, f'Saved fit has an unexpected directory: {path}')
            identity = saved['dataset_fingerprint'], saved['expected_split_id']
            require(identity == anchors.setdefault((key[2], key[3]), identity),
                    f'Saved data/split mismatch: {path}')
            previous = completed_run(path.parent, saved)
            if previous is not None:
                require(previous['config'] == saved, f'Result/config mismatch: {path}')
                validate_result(previous, key)
                records[key] = previous

        reused = len(records)
        failures = []
        print(f'Continuation: {reused}/{EXPECTED_FITS} completed matching fits retained; '
              f'{EXPECTED_FITS - reused} fits remaining.', flush=True)
        if reused < EXPECTED_FITS:
            require(torch.cuda.is_available(), 'CUDA is unavailable inside the allocated job.')
            checks = check_electrode_models(configs)
            write_json(session.report / 'continuation_spatial_checks.json', checks)
        job_id = os.environ['SLURM_JOB_ID']
        write_json(session.report / f'accuracy_continuation_{job_id}.json', {
            'job_id': job_id, 'reused_completed_fits': reused,
            'remaining_fits_at_start': EXPECTED_FITS - reused,
            'provenance': current, 'settings_source': 'accuracy_plan.json',
        })
        try:
            write_accuracy_tables(session.report, list(records.values()), failures, status='running')
            for model in MODELS:
                for subject in SUBJECTS:
                    for seed in SEEDS:
                        for attention in ATTENTIONS:
                            key = model, attention, subject, seed
                            if key in records:
                                print(f'Retained: {model}/{attention}/{subject}/seed_{seed}', flush=True)
                                continue
                            config = copy.deepcopy(configs[key[:3]])
                            config.update(seeds=[seed], provenance=plan['provenance'])
                            if (subject, seed) in anchors:
                                config['dataset_fingerprint'], config['expected_split_id'] = anchors[subject, seed]
                            print(f'Continue {len(records) + 1}/{EXPECTED_FITS}: '
                                  f'{model}/{attention}/{subject}/seed_{seed}', flush=True)
                            try:
                                result, = run_experiment(config, skip_completed=True)
                                validate_result(result, key)
                                records[key] = result
                                print(f"Test accuracy: {100 * result['test']['accuracy']:.2f}%", flush=True)
                            except Exception as error:
                                failures.append({'model': model, 'attention': attention, 'subject': subject,
                                                 'seed': seed, 'type': type(error).__name__, 'message': str(error),
                                                 'traceback': traceback.format_exc()})
                                print(f'Fit failed: {error}', flush=True)
                            write_accuracy_tables(session.report, list(records.values()), failures, status='running')
                            gc.collect()
                            torch.cuda.empty_cache()
        except BaseException:
            write_accuracy_tables(session.report, list(records.values()), failures, status='interrupted')
            raise
        complete = len(records) == EXPECTED_FITS and not failures
        write_accuracy_tables(session.report, list(records.values()), failures,
                              status='complete' if complete else 'incomplete')
        print((session.report / 'accuracy_table.md').read_text(), flush=True)
        require(complete, 'Incomplete study. See accuracy_status.json; this continuation file can be submitted again.')
        print(f'Accuracy table: {session.report / "accuracy_table.csv"}', flush=True)


continue_study(sys.argv[1])
AGFL_CONTINUATION_PY
