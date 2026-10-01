#!/bin/bash -l
#SBATCH --job-name=agfl-eegnet-pre-spatial
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=agfl-eegnet-pre-spatial-%j.log
#SBATCH --open-mode=append

# EEGNet with attention BEFORE the spatial filter (electrode_architecture=pre_spatial)
# under the corrected EEG recipe: A03/A04/A09 x base AGFL / base MHA x seeds 0-4
# = 30 fits of 500 epochs. Submit from the checkout that contains this revision:
#     sbatch eegnet_A03_A04_A09_pre_spatial.sbatch
# Guide with upload/download commands: docs/eegnet_pre_spatial.md
# Re-submitting this SAME file after an interruption keeps completed matching
# fits (--skip-completed) and restarts unfinished fits from epoch 1. Checkpoints
# stay in results/<session>/artifacts; download only results/<session>/report/.
# Optional environment overrides: AGFL_VENV, AGFL_DATA_DIR, AGFL_OUTPUT_DIR,
# AGFL_PYTHON_MODULE, AGFL_BATCH_SIZE (GPU-memory fallback; changes the
# experiment identity, use a new AGFL_OUTPUT_DIR) and AGFL_DIAGNOSTICS=1
# (also generate checkpoint diagnostics after training).
set -Eeuo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo 'Submit with sbatch from the cluster AGFL checkout that contains this revision.' >&2
    exit 2
fi
cd -- "$SLURM_SUBMIT_DIR"
PRESET=eegnet-pre-spatial-a03-a04-a09
if [[ ! -f "agfl/presets/$PRESET.json" ]] || ! grep -q 'pre_spatial' agfl/models/eegnet/backbone.py; then
    echo 'This checkout lacks the pre_spatial revision; upload it first (see docs/eegnet_pre_spatial.md).' >&2
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
AGFL_RESULT_PATH="${AGFL_OUTPUT_DIR:-results/eegnet-A03-A04-A09-pre-spatial-v1}"
OVERRIDES=(--set "data.data_dir=$AGFL_DATA_PATH")
if [[ -n "${AGFL_BATCH_SIZE:-}" ]]; then
    OVERRIDES+=(--set "training.batch_size=$AGFL_BATCH_SIZE")
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$AGFL_ENV_DIR/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "AGFL requires Python 3.12 or newer")'
printf 'Job %s | project %s | data %s | output %s | preset %s\n' \
    "$SLURM_JOB_ID" "$PWD" "$AGFL_DATA_PATH" "$AGFL_RESULT_PATH" "$PRESET"

echo '== 1/4 Resolved settings (no data loaded) =='
"$AGFL_ENV_DIR/bin/python" main.py plan --preset "$PRESET" --output-dir "$AGFL_RESULT_PATH" "${OVERRIDES[@]}"

echo '== 2/4 Recipe and architecture preflight on a synthetic batch =='
srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" - "$PRESET" <<'AGFL_PREFLIGHT_PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('AGFL requires Python 3.12 or newer')
import torch
from agfl.config import merge, resolve_experiments
from agfl.models import get_model_spec
from agfl.presets import load_preset
from agfl.reproducibility import seed_everything

document = load_preset(sys.argv[1])
configs = [c for arm in document['experiments'] for c in resolve_experiments(merge(document['base'], arm))]
if len(configs) != 6 or sum(len(c['seeds']) for c in configs) != 30:
    raise SystemExit(f'Unexpected matrix: {len(configs)} configurations')
for c in configs:
    d, m, t = c['data'], c['model_options'], c['training']
    checks = {
        'artifact_policy=include': d['artifact_policy'] == 'include',
        'filter_scope=run': d['filter_scope'] == 'run',
        'temp_kernel=125': m['temp_kernel'] == 125,
        'electrode_architecture=pre_spatial': m['electrode_architecture'] == 'pre_spatial',
        'loss=cross_entropy': t['loss'] == 'cross_entropy',
        'learning_rate=0.001': t['learning_rate'] == 0.001,
        'epochs=500': t['epochs'] == 500,
        'checkpoint_criterion=loss': t['checkpoint_criterion'] == 'loss',
        'no temporal_bias/output_gate': not c['attention_options'].get('temporal_bias', False)
                                        and not c['attention_options'].get('output_gate', False),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise SystemExit(f"{c['subject_id']}/{c['attention']}: recipe check failed: {failed}")
print('Recipe checks passed for all six subject/attention configurations', flush=True)
metadata = {'modality': 'eeg', 'channels': 22, 'samples': 1000, 'num_classes': 4}
for c in {c['attention']: c for c in configs}.values():
    seed_everything(0, c['deterministic'], c['threads'])
    model = get_model_spec('eegnet').build(c['model_options'], metadata, c['attention'],
                                           c['attention_options']).to(c['device']).eval()
    layer = model.attn_blocks[0]
    seen = []
    hook = layer.register_forward_hook(lambda module, args, output: seen.append((tuple(args[0].shape), tuple(output.shape))))
    try:
        with torch.no_grad():
            logits = model(torch.zeros(2, 22, 1000, device=c['device']))
    finally:
        hook.remove()
    (inputs, outputs), = seen
    ok = (model.pre_spatial and model.spatial_readout is None and not hasattr(model, 'electrode_projection')
          and tuple(model.block2[0].kernel_size) == (22, 1) and model.fc.in_features == 32 * 7
          and inputs == (2 * 1000, 22, 16) and outputs == inputs
          and layer.num_tokens == 22 and layer.token_axis == 'electrode'
          and tuple(logits.shape) == (2, 4) and bool(torch.isfinite(logits).all()))
    if not ok:
        raise SystemExit(f"eegnet/{c['attention']}: pre_spatial preflight failed; attention saw {inputs}")
    print(f"Preflight passed: eegnet/{c['attention']} attention input {inputs} (22 electrode nodes at "
          f"every time step) -> spatial filter {tuple(model.block2[0].kernel_size)} -> {model.fc.in_features} classifier features; "
          f"parameters {sum(p.numel() for p in model.parameters())}", flush=True)
    del model, layer, logits
    torch.cuda.empty_cache()
AGFL_PREFLIGHT_PY

echo '== 3/4 Training: 30 fits; completed matching fits are kept =='
srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" main.py sweep --preset "$PRESET" \
    --output-dir "$AGFL_RESULT_PATH" "${OVERRIDES[@]}" --skip-completed

echo '== 4/4 Result tables and plots from saved artifacts =='
srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" main.py analyze "$AGFL_RESULT_PATH" --plots

if [[ "${AGFL_DIAGNOSTICS:-0}" == "1" ]]; then
    echo '== Optional checkpoint diagnostics (validation partition, PCA embedding) =='
    srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" main.py diagnose-session "$AGFL_RESULT_PATH" \
        --device cuda --partition validation --max-samples 64 --embedding pca
    srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" main.py analyze "$AGFL_RESULT_PATH" --plots
fi

printf '\nFinished. Download only: %s/report/\n' "$AGFL_RESULT_PATH"
