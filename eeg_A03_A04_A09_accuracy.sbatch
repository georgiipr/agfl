#!/bin/bash -l
#SBATCH --job-name=agfl-spatial-accuracy
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=agfl-spatial-accuracy-%j.log
#SBATCH --open-mode=append

# Submit from the updated AGFL checkout: sbatch eeg_A03_A04_A09_accuracy.sbatch
# Five EEG models x AGFL/MHA x A03/A04/A09 x five seeds = 150 fits.
# No salloc, plotting, checkpoint files, new packages, or pytest step.
# Optional environment overrides: AGFL_VENV, AGFL_DATA_DIR, AGFL_OUTPUT_DIR,
# AGFL_PYTHON_MODULE (only if required by the site's existing environment).
# A new submission retrains the matrix; it does not reuse completed old scores.

set -Eeuo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo 'Submit with sbatch from the cluster AGFL directory.' >&2
    exit 2
fi
cd -- "$SLURM_SUBMIT_DIR"
if [[ ! -f agfl/accuracy_study.py || ! -f agfl/presets/eeg-three-subject-spatial-accuracy.json ]]; then
    echo 'Upload the updated source and accuracy preset, then submit from AGFL.' >&2
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
AGFL_RESULT_PATH="${AGFL_OUTPUT_DIR:-results/eeg-A03-A04-A09-spatial-accuracy-v1}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

"$AGFL_ENV_DIR/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else "AGFL requires Python 3.12 or newer")'
printf 'Job %s | project %s | data %s | output %s\n' \
    "$SLURM_JOB_ID" "$PWD" "$AGFL_DATA_PATH" "$AGFL_RESULT_PATH"

srun --ntasks=1 "$AGFL_ENV_DIR/bin/python" -m agfl.accuracy_study \
    --data-dir "$AGFL_DATA_PATH" \
    --output-dir "$AGFL_RESULT_PATH"

printf '\nFinished. Download only: %s/report/accuracy_table.csv\n' "$AGFL_RESULT_PATH"
