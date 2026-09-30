#!/bin/bash -l
#SBATCH --job-name=agfl-signal-transformer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=agfl-signal-transformer-%j.log
#SBATCH --open-mode=append

# Submit from the existing updated AGFL checkout. This is the only new file
# needed on the cluster; no model/source changes or package installs are made.
# Signal Transformer x (base AGFL, MHA, Performer) x A03/A04/A09 x seeds 0-4 = 45 fits.
# Re-submit this SAME file after an interruption to retain completed fits.
# A changed source/config/environment/launcher requires a new AGFL_OUTPUT_DIR.
# Optional overrides: AGFL_DATA_DIR, AGFL_OUTPUT_DIR, AGFL_VENV,
# AGFL_PYTHON_MODULE. No plots or checkpoint files are generated.
set -Eeuo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo 'Submit with sbatch from the cluster AGFL directory.' >&2
    exit 2
fi
cd -- "$SLURM_SUBMIT_DIR"
if [[ ! -f agfl/engine.py ]]; then
    echo 'Submit from the AGFL checkout used for the spatial accuracy study.' >&2
    exit 2
fi
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
AGFL_DATA_PATH="${AGFL_DATA_DIR:-../ml}"
AGFL_RESULT_PATH="${AGFL_OUTPUT_DIR:-results/signal_transformer-A03-A04-A09-attentions-v1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
printf 'Job %s | Signal Transformer | data %s | output %s\n' \
    "$SLURM_JOB_ID" "$AGFL_DATA_PATH" "$AGFL_RESULT_PATH"

srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" - "$AGFL_DATA_PATH" "$AGFL_RESULT_PATH" "$0" <<'AGFL_SIGNAL_TRANSFORMER_PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('AGFL requires Python 3.12 or newer')

import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from agfl.config import experiment_identity, resolve_config
from agfl.engine import completed_run, run_experiment
from agfl.models import get_model_spec
from agfl.reproducibility import provenance, seed_everything
from agfl.session import session_activity, session_paths
from agfl.storage import run_directory, write_json

STUDY = 'signal_transformer-A03-A04-A09-attentions-v1'
SUBJECTS = ('A03', 'A04', 'A09')
SEEDS = (0, 1, 2, 3, 4)
ATTENTIONS = ('agfl', 'mha', 'performer')
EXPECTED_FITS = 45
BASE = {
    'dataset': 'eeg', 'model': 'signal_transformer', 'device': 'cuda',
    'deterministic': True, 'threads': 1, 'seeds': [0, 1, 2, 3, 4],
    'data': {
        'sessions': ['T'], 'normalization': 'train_channel', 'filter_scope': 'trial',
        'lowcut': 2.0, 'highcut': 30.0, 'offset_seconds': 0.0,
        'window': 1000, 'artifact_policy': 'exclude',
    },
    'split': {'protocol': 'stratified', 'train': 0.6, 'validation': 0.2, 'test': 0.2},
    'model_options': {
        'dim': 64, 'depth': 2, 'dropout': 0.1, 'mlp_ratio': 4.0,
        'eeg_temporal_bins': 8, 'eeg_kernel_size': 15, 'ecg_patch_size': 16,
        'spatial_readout': 'learned',
    },
    'training': {
        'epochs': 250, 'batch_size': 64, 'optimizer': 'adamw',
        'learning_rate': 0.005, 'weight_decay': 0.001, 'momentum': 0.9,
        'loss': 'focal', 'class_weights': 'balanced', 'focal_gamma': 3.0,
        'checkpoint_criterion': 'accuracy', 'checkpoint_tiebreaker': 'none',
        'scheduler': 'warmup_cosine', 'warmup_epochs': 10, 'min_lr_ratio': 0.0,
        'gradient_clip': None, 'early_stopping_patience': 0,
        'early_stopping_min_epochs': 100, 'amp': False, 'num_workers': 0,
        'augmentation': {'shift': 0, 'scale': 0.0, 'noise': 0.0,
                         'recombine_segments': 0, 'recombine_probability': 0.5},
        'ema_decay': 0.0, 'save_checkpoints': False,
    },
}
ATTENTION_OPTIONS = {
    'agfl': {
        'heads': 4, 'agfl_variant': 'polynomial', 'K': 2,
        'top_k': 'scheduled', 'top_k_stage': 'scores', 'top_k_ties': 'threshold',
        'top_k_schedule': {'smax': 0.2, 'smin': 0.8, 'alpha': 3.0},
        'graph_normalization': 'softmax', 'post_topk_renormalize': True,
        'coefficient_activation': 'identity', 'coefficient_init': 'zeros',
        'coefficient_conditioning': 'static', 'coefficient_conditioning_scale': 0.5,
        'temporal_bias': False, 'output_gate': False, 'learnable_coefficients': True,
        'projection': 'split_input', 'qkv_bias': False, 'filter_projection': 'separate',
        'filter_projection_init': 'pytorch', 'score_scaling': 'temperature',
        'temperature_init': 1.0, 'hop_normalization': 'feature',
    },
    'mha': {'heads': 4, 'temporal_bias': False, 'output_gate': False},
    'performer': {'heads': 4, 'random_features': 64},
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def write_text(path, text):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def same_source(actual, expected):
    return all(actual.get(k) == expected[k] for k in ('source_sha256', 'packages'))


def make_configs(data_dir, output_dir):
    configs = {}
    for subject in SUBJECTS:
        for attention in ATTENTIONS:
            c = copy.deepcopy(BASE)
            c['data'].update(data_dir=str(data_dir), subjects=[int(subject[1:])])
            c.update(subject_id=subject, attention=attention,
                     attention_options=copy.deepcopy(ATTENTION_OPTIONS[attention]),
                     output_dir=str(output_dir), study_arm=attention)
            c['study'] = {
                'name': STUDY, 'subjects': list(SUBJECTS), 'seeds': list(SEEDS),
                'arms': list(ATTENTIONS),
                'comparisons': [['agfl', 'mha'], ['agfl', 'performer']],
            }
            configs[attention, subject] = resolve_config(c)
    for subject in SUBJECTS:
        reference = configs['agfl', subject]
        for attention in ATTENTIONS:
            c = configs[attention, subject]
            for key in ('data', 'split', 'training', 'model_options', 'device', 'deterministic', 'threads'):
                require(c[key] == reference[key], f'Unmatched {key} for {subject}/{attention}')
            require(c['attention_options']['heads'] == 4, 'Every attention must use four heads.')
    return configs


def spatial_preflight(configs):
    metadata = {'modality': 'eeg', 'channels': 22, 'samples': 1000, 'num_classes': 4}
    checks = []
    for attention in ATTENTIONS:
        c = configs[attention, SUBJECTS[0]]
        seed_everything(0, c['deterministic'], c['threads'])
        model = get_model_spec('signal_transformer').build(c['model_options'], metadata,
                    attention, c['attention_options']).to(c['device']).train()
        layers = {name: m for name, m in model.named_modules() if getattr(m, 'is_attention', False)}
        require(model.token_axis == 'electrode' and model.num_tokens == 22 and len(layers) == 2,
                'Signal Transformer must contain two attention blocks with 22 electrode nodes.')
        observed, hooks = {}, []

        def capture(name):
            def hook(layer, args, output):
                x = args[0]
                require(layer.token_axis == 'electrode' and layer.num_tokens == 22
                        and layer.attention_key == attention and tuple(x.shape) == (2, 22, 64)
                        and output.shape == x.shape, f'{attention}/{name}: invalid electrode attention shape')
                observed[name] = {'input_shape': list(x.shape), 'token_axis': 'electrode', 'num_tokens': 22}
            return hook

        try:
            for name, layer in layers.items():
                hooks.append(layer.register_forward_hook(capture(name)))
            output = model(torch.zeros(2, 22, 1000, device=c['device']))
            require(output.shape == (2, 4) and bool(torch.isfinite(output).all()) and set(observed) == set(layers),
                    f'{attention}: invalid logits or unexecuted attention block')
            output.square().mean().backward()
            require(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                    f'{attention}: nonfinite gradients in deterministic backward preflight')
        finally:
            for hook in hooks:
                hook.remove()
        checks.append({'attention': attention, 'layers': observed, 'deterministic_backward': 'passed'})
        print(f'Electrode preflight passed: signal_transformer/{attention}; 2 blocks, 22 nodes; forward/backward.', flush=True)
        del model, layers, output
        gc.collect()
        torch.cuda.empty_cache()
    return checks


def write_tables(report, records, failures, status):
    rows, means = [], {}
    for subject in SUBJECTS:
        row = {'model': 'signal_transformer', 'subject': subject}
        for attention in ATTENTIONS:
            scores = [records[attention, subject, seed]['test']['accuracy']
                      for seed in SEEDS if (attention, subject, seed) in records]
            value = 100 * statistics.mean(scores) if len(scores) == len(SEEDS) else None
            row[f'{attention}_accuracy_percent'] = value
            means[attention, subject] = value
        row['completed_fits'] = sum(key[1] == subject for key in records)
        row['expected_fits'] = len(ATTENTIONS) * len(SEEDS)
        rows.append(row)
    average = {'model': 'signal_transformer', 'subject': 'AVERAGE',
               'completed_fits': len(records), 'expected_fits': EXPECTED_FITS}
    for attention in ATTENTIONS:
        values = [means[attention, s] for s in SUBJECTS]
        average[f'{attention}_accuracy_percent'] = (statistics.mean(values)
                                                   if all(v is not None for v in values) else None)
    rows.append(average)
    for row in rows:
        a = row['agfl_accuracy_percent']
        for baseline in ('mha', 'performer'):
            b = row[f'{baseline}_accuracy_percent']
            row[f'agfl_minus_{baseline}_pp'] = a - b if a is not None and b is not None else None

    def write_csv(path, values):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(values[0]))
        writer.writeheader()
        for row in values:
            writer.writerow({k: round(v, 6) if isinstance(v, float) else v for k, v in row.items()})
        write_text(path, stream.getvalue())

    write_csv(report / 'accuracy_table.csv', rows)
    seed_rows = []
    for subject in SUBJECTS:
        for seed in SEEDS:
            row = {'subject': subject, 'seed': seed}
            for attention in ATTENTIONS:
                result = records.get((attention, subject, seed))
                row[f'{attention}_accuracy_percent'] = None if result is None else 100 * result['test']['accuracy']
            seed_rows.append(row)
    write_csv(report / 'accuracy_by_seed.csv', seed_rows)
    fmt = lambda value: 'pending' if value is None else f'{value:.2f}'
    lines = ['# Signal Transformer electrode attention: A03, A04, A09', '',
             f'Status: {status}; {len(records)}/{EXPECTED_FITS} completed; {len(failures)} failed fits.', '',
             'Five seeds per subject, then equal subject weighting. Accuracy in percent; differences in percentage points.', '',
             '| Subject | AGFL | MHA | Performer | AGFL - MHA | AGFL - Performer | Fits |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[f'{a}_accuracy_percent'] for a in ATTENTIONS]
        values += [row['agfl_minus_mha_pp'], row['agfl_minus_performer_pp']]
        lines.append(f"| {row['subject']} | " + ' | '.join(fmt(v) for v in values)
                     + f" | {row['completed_fits']}/{row['expected_fits']} |")
    lines += ['', 'Every attention block operates on 22 electrodes. This is the repository generic electrode Transformer.',
              'Best weights are selected using validation accuracy; held-out test scores do not select candidates or epochs.',
              'Linformer and Nystromformer were omitted using the previous EEGNet/A03 ranking, not Signal Transformer results.',
              'Blank/pending means require remaining seeds/subjects. Individual scores are in accuracy_by_seed.csv.',
              'No plots or checkpoint files are produced. These three development subjects do not establish a nine-subject result.', '']
    write_text(report / 'accuracy_table.md', '\n'.join(lines))
    write_json(report / 'accuracy_status.json', {
        'status': status, 'completed_fits': len(records), 'expected_fits': EXPECTED_FITS,
        'failed_fits': failures,
        'missing_fits': [{'attention': a, 'subject': s, 'seed': seed}
                         for s in SUBJECTS for seed in SEEDS for a in ATTENTIONS if (a, s, seed) not in records],
    })


def run_study(data_dir, output_dir, launcher_path):
    data_dir, output_dir = Path(data_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    launcher = Path(launcher_path).read_bytes()
    session = session_paths(output_dir, create=True)
    with session_activity(session.root), run_directory(session.artifacts / '_accuracy_study'):
        require(torch.cuda.is_available(), 'CUDA is unavailable inside the allocated job.')
        for subject in SUBJECTS:
            require((data_dir / f'{subject}T.gdf').is_file(), f'Missing dataset: {data_dir / (subject + "T.gdf")}')
        configs = make_configs(data_dir, session.root)
        current = provenance()
        plan = {
            'schema_version': 1, 'study': STUDY, 'models': ['signal_transformer'],
            'subjects': list(SUBJECTS), 'seeds': list(SEEDS), 'attentions': list(ATTENTIONS),
            'expected_fits': EXPECTED_FITS, 'configs': list(configs.values()),
            'provenance': current, 'launcher_sha256': hashlib.sha256(launcher).hexdigest(),
            'attention_selection': {
                'basis': 'Previous EEGNet/A03 base electrode attention report, five seeds; selected before this Signal Transformer study.',
                'previous_mean_test_percent': {'agfl': 69.259259, 'mha': 66.296296, 'performer': 65.185185,
                                               'linformer': 63.703704, 'nystromformer': 54.074074},
                'omitted': ['linformer', 'nystromformer'],
            },
        }
        plan_path = session.report / 'accuracy_plan.json'
        paths = sorted((session.artifacts / 'eeg').glob('subject_*/*/seed_*/config.json'))
        if plan_path.exists():
            saved_plan = read_json(plan_path)
            require({k: v for k, v in saved_plan.items() if k != 'provenance'}
                    == {k: v for k, v in plan.items() if k != 'provenance'}
                    and same_source(saved_plan['provenance'], current),
                    'Existing session has a different plan, launcher, source or packages. '
                    'Restore its original files/environment or choose a NEW AGFL_OUTPUT_DIR; old scores will not be reused.')
            plan = saved_plan
        else:
            require(not paths, 'Run artifacts exist without their original accuracy_plan.json; choose a new output folder.')
            write_json(plan_path, plan)
            write_text(session.report / 'signal_transformer_launch.sbatch', launcher.decode())

        records, anchors, subject_fingerprints, paired_protocol, failures = {}, {}, {}, {}, []
        seen = set()

        def validate_result(result, key):
            require(result['status'] == 'completed' and result['model'] == 'signal_transformer' and result['dataset'] == 'eeg'
                    and (result['attention'], result['subject_id'], result['seed']) == key,
                    'Unexpected result model/attention/subject/seed/status.')
            score = result['test']['accuracy']
            require(type(score) in (float, int) and math.isfinite(score) and 0 <= score <= 1, 'Invalid test accuracy.')
            modules = result['attention_modules']
            require(result['token_axis'] == 'electrode' and result['num_tokens'] == 22
                    and len(modules) == 2 and result.get('checkpoint_retained') is False
                    and not result.get('synthetic', False)
                    and all(m['token_axis'] == 'electrode' and m['num_tokens'] == 22 and m['attention'] == key[0] for m in modules),
                    'Result violates the Signal Transformer electrode/storage contract.')
            pair = key[1], key[2]
            identity = result['dataset_fingerprint'], result['split_id']
            require(identity[0] == subject_fingerprints.setdefault(key[1], identity[0]),
                    'Dataset changed between seeds for the same subject.')
            require(identity == anchors.setdefault(pair, identity), 'Attention arms used different data/splits.')
            require(result['comparison_id'] == paired_protocol.setdefault(pair, result['comparison_id']),
                    'Attention arms used different training/source protocols.')

        # Restore only fully written runs from this exact saved plan. Partial
        # fits anchor the original data/split, but never count as completed.
        for path in paths:
            saved = read_json(path)
            require(len(saved['seeds']) == 1, f'Invalid per-fit seed list: {path}')
            key = saved['attention'], saved['subject_id'], saved['seeds'][0]
            require(key[:2] in configs and key[2] in SEEDS and key not in seen,
                    f'Unexpected or duplicate saved fit: {path}')
            seen.add(key)
            base = {k: v for k, v in saved.items() if k not in {
                'provenance', 'dataset_fingerprint', 'expected_split_id', 'resolved_metadata'}}
            base['seeds'] = list(SEEDS)
            require(base == configs[key[:2]] and same_source(saved['provenance'], plan['provenance']),
                    f'Saved configuration/source differs from this study: {path}')
            expected_dir = (session.artifacts / 'eeg' / f'subject_{key[1]}'
                            / f'signal_transformer-{key[0]}-{experiment_identity(saved)}' / f'seed_{key[2]}')
            require(path.parent == expected_dir, f'Saved fit has an unexpected directory: {path}')
            identity = saved['dataset_fingerprint'], saved['expected_split_id']
            require(identity[0] == subject_fingerprints.setdefault(key[1], identity[0]),
                    f'Saved dataset differs between seeds: {path}')
            require(identity == anchors.setdefault((key[1], key[2]), identity), f'Saved data/split mismatch: {path}')
            previous = completed_run(path.parent, saved)
            if previous is not None:
                require(previous['config'] == saved, f'Result/config mismatch: {path}')
                validate_result(previous, key)
                records[key] = previous

        reused = len(records)
        print(f'Signal Transformer: {reused}/{EXPECTED_FITS} completed matching fits retained; {EXPECTED_FITS - reused} remaining.', flush=True)
        job_id = os.environ['SLURM_JOB_ID']
        write_json(session.report / f'allocation_{job_id}.json', {
            'job_id': job_id, 'reused_completed_fits': reused,
            'provenance': current, 'launcher_sha256': plan['launcher_sha256'],
        })
        try:
            write_tables(session.report, records, failures, 'preflight')
            if reused < EXPECTED_FITS:
                write_json(session.report / 'spatial_checks.json', spatial_preflight(configs))
            for subject in SUBJECTS:
                for seed in SEEDS:
                    for attention in ATTENTIONS:
                        key = attention, subject, seed
                        if key in records:
                            continue
                        config = copy.deepcopy(configs[attention, subject])
                        config.update(seeds=[seed], provenance=plan['provenance'])
                        if subject in subject_fingerprints:
                            config['dataset_fingerprint'] = subject_fingerprints[subject]
                        if (subject, seed) in anchors:
                            config['dataset_fingerprint'], config['expected_split_id'] = anchors[subject, seed]
                        print(f'Fit {len(records) + len(failures) + 1}/{EXPECTED_FITS}: '
                              f'signal_transformer/{attention}/{subject}/seed_{seed}', flush=True)
                        started = time.monotonic()
                        try:
                            result, = run_experiment(config, skip_completed=True)
                            validate_result(result, key)
                            records[key] = result
                            print(f"Test accuracy: {100 * result['test']['accuracy']:.2f}% "
                                  f"| elapsed {(time.monotonic() - started) / 60:.1f} min", flush=True)
                        except Exception as error:
                            failures.append({'attention': attention, 'subject': subject, 'seed': seed,
                                             'type': type(error).__name__, 'message': str(error),
                                             'traceback': traceback.format_exc()})
                            print(f'Fit failed: {error}', flush=True)
                        write_tables(session.report, records, failures, 'running')
                        gc.collect()
                        torch.cuda.empty_cache()
        except BaseException:
            write_tables(session.report, records, failures, 'interrupted')
            raise
        complete = len(records) == EXPECTED_FITS and not failures
        write_tables(session.report, records, failures, 'complete' if complete else 'incomplete')
        print((session.report / 'accuracy_table.md').read_text(), flush=True)
        require(complete, 'Some fits failed. See accuracy_status.json; resubmit this file to retry unfinished fits.')
        print(f'Finished. Download: {session.report / "accuracy_table.csv"}', flush=True)


run_study(sys.argv[1], sys.argv[2], sys.argv[3])
AGFL_SIGNAL_TRANSFORMER_PY
