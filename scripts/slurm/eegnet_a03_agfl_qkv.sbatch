#!/bin/bash -l
#SBATCH --job-name=agfl-A03-qkv
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=10G
#SBATCH --time=04:00:00
#SBATCH --output=agfl-A03-qkv-%j.log
#SBATCH --open-mode=append

# Historical optional launcher; the current workflow uses interactive salloc.
# Upload this single file into the updated AGFL checkout on the cluster.
# From that directory: sbatch eegnet-A03-agfl-qkv-v3.sbatch
# No salloc, tmux session, or separate training/plot commands are needed.
# All Python below runs ON THE CLUSTER, only after Slurm grants this job.
# Requires the existing training and plotting environment, without pytest.
# Optional exported overrides: AGFL_VENV, AGFL_DATA_DIR, AGFL_OUTPUT_DIR,
# AGFL_PYTHON_MODULE (only if your site requires an explicit module load).
# Matching completed candidates are reused; incomplete fits restart at epoch 1.
# This script neither installs packages nor submits continuation jobs.

set -Eeuo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo 'Submit this file with sbatch from the cluster AGFL directory.' >&2
    exit 2
fi
cd -- "$SLURM_SUBMIT_DIR"
if [[ ! -f main.py || ! -f agfl/session.py ]]; then
    echo 'The submission directory must contain the updated AGFL checkout.' >&2
    exit 2
fi

STAGE=environment
BATCH_DIR=''
finish() {
    local rc=$?
    trap - EXIT
    set +e
    printf '\nFinished: %s | job=%s | stage=%s | exit=%s\n' \
        "$(date -Is)" "$SLURM_JOB_ID" "$STAGE" "$rc"
    if [[ -n "$BATCH_DIR" ]]; then
        printf 'job_id=%s\nstage=%s\nexit_code=%s\nfinished=%s\n' \
            "$SLURM_JOB_ID" "$STAGE" "$rc" "$(date -Is)" > "$BATCH_DIR/status.txt"
        printf 'Download: %s/report/\n' "$SESSION_DIR"
    fi
    exit "$rc"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Login-shell initialization retains the cluster's module environment. Do not
# guess a module name; the existing ~/.venv was used by the previous runs.
if [[ -n "${AGFL_PYTHON_MODULE:-}" ]]; then
    if ! type module >/dev/null 2>&1; then
        echo 'AGFL_PYTHON_MODULE was set, but the module command is unavailable.' >&2
        exit 2
    fi
    set +u
    module load "$AGFL_PYTHON_MODULE"
    set -u
fi
VENV_DIR="${AGFL_VENV:-$HOME/.venv}"
if [[ ! -r "$VENV_DIR/bin/activate" || ! -x "$VENV_DIR/bin/python" ]]; then
    printf 'Missing Python environment: %s (override with AGFL_VENV).\n' "$VENV_DIR" >&2
    exit 2
fi
set +u
source "$VENV_DIR/bin/activate"
set -u
PYTHON="$VENV_DIR/bin/python"
DATA_DIR="${AGFL_DATA_DIR:-../ml}"
SESSION_DIR="${AGFL_OUTPUT_DIR:-results/eegnet-A03-agfl-qkv-v3}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"

# Initialize the supported session layout before adding report files. Creating
# report/ alone would be interpreted by AGFL as a portable, incomplete session.
STAGE=session_setup
"$PYTHON" - "$SESSION_DIR" <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit('AGFL requires Python 3.12 or newer; check AGFL_VENV.')
from agfl.session import session_paths
session_paths(sys.argv[1], create=True)
PY
BATCH_DIR="$SESSION_DIR/report/batch/job-$SLURM_JOB_ID"
mkdir -p -- "$BATCH_DIR"
exec > >(tee -a "$BATCH_DIR/job.log") 2>&1
cp -- "${BASH_SOURCE[0]}" "$BATCH_DIR/launch.sbatch"
printf 'Started: %s | job=%s | host=%s\n' "$(date -Is)" "$SLURM_JOB_ID" "$(hostname)"
printf 'Project: %s\nPython: %s\nDataset: %s\nSession: %s\n' \
    "$PWD" "$PYTHON" "$DATA_DIR" "$SESSION_DIR"
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git rev-parse HEAD > "$BATCH_DIR/git-commit.txt"
    git status --short > "$BATCH_DIR/git-status.txt"
fi

STAGE=preflight
if [[ ! -r "$DATA_DIR/A03T.gdf" ]]; then
    printf 'Missing or unreadable dataset: %s/A03T.gdf\n' "$DATA_DIR" >&2
    exit 2
fi
"$PYTHON" - <<'PY'
import importlib
import sys
print('Interpreter:', sys.version, flush=True)
for name in ('torch', 'numpy', 'scipy', 'sklearn', 'mne', 'wfdb', 'tqdm', 'matplotlib'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'unknown'), flush=True)
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable inside the allocated job; check the PyTorch environment.')
print('GPU:', torch.cuda.get_device_name(0), flush=True)
PY

STAGE=training
"$PYTHON" main.py tune-eegnet \
    --data-dir "$DATA_DIR" \
    --subjects 3 \
    --seeds 0 1 2 3 4 \
    --epochs 250 \
    --candidate spatial_qkv_dense \
    --candidate spatial_qkv_sparse \
    --candidate spatial_qkv_renorm \
    --output-dir "$SESSION_DIR"

# Generate result figures even if a checkpoint diagnostic fails, then retain a
# failing job status so a missing diagnostic is not reported as success.
STAGE=checkpoint_diagnostics
DIAGNOSTIC_RC=0
"$PYTHON" main.py diagnose-session "$SESSION_DIR" \
    --data-dir "$DATA_DIR" --device cuda --partition validation --embedding tsne \
    || DIAGNOSTIC_RC=$?

STAGE=result_plots
"$PYTHON" main.py analyze "$SESSION_DIR" --plots
if [[ "$DIAGNOSTIC_RC" -ne 0 ]]; then
    STAGE=checkpoint_diagnostics_failed
    echo 'Result plots finished, but checkpoint diagnostics failed; see the job log.' >&2
    exit "$DIAGNOSTIC_RC"
fi
STAGE=complete
echo 'Training, checkpoint diagnostics, and result plots completed.'
