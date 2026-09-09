# AGFL research framework

Select a **model** (EEGNet, EEGEncoder, DSTS EEGEncoder or Conformer) and an
independent **attention mechanism**. The selected attention runs inside that
model. Every model owns separate EEG and ECG variants.

This is a source refactor prepared for the experiment machine. No project code,
training, evaluation, installation or tests have been run locally. Numerical
verification and repeated-seed research results remain pending.

## Models and attention

| Selection | Keys |
|---|---|
| `model` / `--model` | `eegnet`, `eegencoder`, `dstseegencoder`, `conformer`, `signal_transformer` |
| `attention` / `--attention` | `agfl`, `mha`, `performer`, `linformer`, `nystromformer` |
| Dataset | `eeg`, `ecg`; additional adapters `npz_eeg`, `npz_ecg` |
| Model settings | `model_options`: the chosen backbone's convolutions, widths, depth and dropout |
| Attention settings | `attention_options`: heads, AGFL K/Top-k, approximation rank, etc. |

`mha` means vanilla multi-head self-attention. `signal_transformer` is the
explicit generic backbone used by earlier framework runs; those runs were
incorrectly named after their attention mechanisms. They are never presented
as EEGNet results. There are no selectable archived model adapters or
no-attention mechanisms.

## Structure

```text
main.py
agfl/
  models/
    eegnet/              config.py, backbone.py, eeg.py, ecg.py
    eegencoder/          config.py, backbone.py, eeg.py, ecg.py
    dstseegencoder/      config.py, backbone.py, eeg.py, ecg.py
    conformer/           config.py, backbone.py, eeg.py, ecg.py
    signal_transformer/ config.py, backbone.py, eeg.py, ecg.py
    _shared/             reusable data/train/evaluate policy and small layers
  attention/
    agfl/                config.py, layer.py
    mha/                 config.py, layer.py
    performer/           config.py, layer.py
    linformer/           config.py, layer.py
    nystromformer/       config.py, layer.py
  datasets/              loaders, stable sample IDs and persisted splits
  presets/               model + attention experiment configurations
  config.py              validation and experiment identities
  engine.py              shared training/evaluation
  analysis.py            saved-result tables and paired statistics
  visualization/         result figures and checkpoint diagnostics
tests/references/        independent original sources, excluded from installation
docs/                    mathematical, dataset and pipeline audits
```

Each model exposes a `ModelSpec` with its defaults, variants and
data/train/evaluate methods. Its constructor receives an attention factory.
The factory inserts the chosen mechanism at the model's attention locations
and isolates initialization RNG so attention capacity does not shift common
backbone weights. Splits and training policy are shared.

EEG uses electrode tokens and learned spatial readouts by default; ECG uses time tokens. These adaptations
change surrounding feature extraction and readout where needed. See
[model structure and architecture changes](docs/model_attention_structure.md).
Preserving AGFL's equations does not imply full-original-backbone parity.
The targeted `eegnet-bci2a` preset below instead applies EEGNet's spatial
convolution across all electrodes first, followed by AGFL over time.

## Launch on the experiment machine

Commands below are for the cluster, from inside `AGFL`, with datasets beside it:

```text
root/
  AGFL/
  ml/                                  A01T.gdf ... A09T.gdf
  mit-bih-arrhythmia-database-1.0.0/     recordings and annotations
```

Use Python 3.12+ and a suitable PyTorch/CUDA environment. If installation is
needed there: `python -m pip install -e '.[dev,plots]'`. Dependencies are defined
in `pyproject.toml`; preserve a target-compatible lock and the installed versions.

Activate your existing environment in each new cluster shell. For the environment
at `~/.venv` used in earlier runs:

```bash
source ~/.venv/bin/activate
python --version
command -v python
```

The version must be 3.12 or newer and the executable should belong to that
environment. An older system Python can report `SyntaxError` at a valid f-string
before loading any model. The package now checks the interpreter version before
importing the CLI and reports an explicit environment error.

BCI IV 2a is trained **separately for each subject**. This command runs one
model and one attention for A01, then A02, through A09, with five independent
seeds per subject: **9 subjects × 5 seeds = 45 runs**. Subjects are never pooled
into one model. Every subject has separate train, validation and test trials.

```bash
python main.py run --preset eeg --model eegnet --attention agfl --output-dir results/eegnet-eeg
```

In an interactive terminal, each subject/seed run has one updating epoch progress
bar showing the model, attention, subject, seed, elapsed time, estimated remaining
time, training loss, training accuracy and validation accuracy. It leaves one
completed bar per run instead of printing a line every epoch. Bars are disabled
automatically when stderr is redirected (for example, to a batch-job log).
All epoch metrics are still saved to `history.json`. The `tqdm` dependency is
installed by the package installation command above.

A short execution check with one subject and one seed (not an accuracy study):

```bash
python main.py run --preset eeg --model eegnet --attention agfl --seeds 0 \
  --set 'data.subjects=[1]' --set training.epochs=2 \
  --output-dir results/quick-eegnet-eeg
```

EEGEncoder with MHA, or Conformer with AGFL on ECG:

```bash
python main.py run --preset eeg --model eegencoder --attention mha
python main.py run --preset ecg --model conformer --attention agfl
```

The EEG preset defaults to EEGNet; the ECG preset defaults to Conformer.
Both use AGFL unless overridden. To run another backbone on either modality,
change `--model`. `model_variant` is inferred from the dataset.
To run selected EEG subjects, add `--set 'data.subjects=[1,3]'`; this creates
two independent subject experiments. To select one seed, add `--seeds 0`.
Omitting these options runs all nine subjects and five seeds.
Inspect available keys and resolved settings:

```bash
python main.py list
python main.py presets
python main.py plan --preset eeg --model dstseegencoder --attention performer
```

Paths resolve from the launch directory. Override `data.data_dir` for another
location. EEG E-session evaluation also requires official `AxxE.mat` labels;
labels are never inferred. ECG selects MLII by name and records missing leads.

## Improve EEGNet + AGFL on BCI IV 2a

For the downloaded **46.76%** study, use this focused workflow. It keeps
**EEGNet + AGFL, four classes and nine separately trained subjects**. The
compact candidates restore early spatial filtering, preserve 31 time tokens,
reduce feature width, use cross-entropy and a lower learning rate, and stop
on validation stagnation. The search compares training-channel versus per-trial
normalization and optional same-class training-trial segment recombination.
**75% mean test accuracy is a target; these settings have not yet demonstrated it.**
See [rationale, candidate settings and evaluation protocol](docs/eegnet_bci2a_improvement.md).

Run these commands **on the cluster, from `AGFL`**, with `A01T.gdf` through
`A09T.gdf` in `../ml`. A two-epoch execution check uses its own output directory:

```bash
source ~/.venv/bin/activate
python -m pip install -e '.[dev,plots]'
python -m pytest tests/test_eegnet_bci2a.py tests/test_engine.py tests/test_models.py tests/test_presets.py
python main.py tune-eegnet --data-dir ../ml --subjects 1 --seeds 0 \
  --candidate compact_trialnorm --epochs 2 --output-dir results/eegnet-bci2a-smoke
```

The full study runs five candidates, nine subjects and five seeds: **225
validation-only training runs, then 45 selected checkpoint test evaluations**.
It does not retrain the selected checkpoints. Run:

```bash
python main.py tune-eegnet --data-dir ../ml --output-dir results/eegnet-bci2a-search
```

Checkpoint selection and candidate selection use validation only, separately
within each subject/seed split. Every candidate uses the same trial IDs in that
split. Candidate runs never evaluate the test partition. Settings for all
requested runs are written to `selection_report.json` before any final testing.
The 60/20/20 T-session split, cue window, artifact exclusion and four-class task
remain fixed. Different seeds use overlapping random splits; their scores do
not represent independent new test cohorts.

Relaunch the **identical command** after an interruption: completed candidates
are reused and incomplete candidates restart from epoch 1 automatically.
Use `--restart` to retrain everything in that same search. Use a new output
directory when changing seeds, subjects, candidate list or epoch budget; this
prevents old selected runs from being included in the new study.
Add `--dry-run` to inspect settings without loading data. Repeat `--candidate`
to request a subset; the five names are listed in the linked guide.

If you want one fixed compact setting instead of searching, run:

```bash
python main.py run --preset eegnet-bci2a --output-dir results/eegnet-bci2a-fixed
```

This is 45 ordinary runs with the `compact_trialnorm` configuration. Use
`--skip-completed` to preserve finished runs when relaunching that command.

**Reports and plots for the search:** the overall score, each subject's score
and the selected settings are in
`results/eegnet-bci2a-search/search_report.md` and `search_result.json`.
Analyze **`selected/`** to generate learning curves, confusion matrices and ROC
plots for the selected checkpoints:

```bash
python main.py analyze results/eegnet-bci2a-search/selected \
  --output-dir results/eegnet-bci2a-search/analysis --plots
```

Open `results/eegnet-bci2a-search/analysis/figures/index.md`. Candidate folders
contain validation histories and `selection.json`, without test result files.
Selected settings may differ across seeds/subjects, so generic `analyze`
tables group them by configuration. **Use `search_report.md` for the overall
score of this tuning procedure**, rather than averaging those configuration
groups. For the fixed-preset command, analyze `results/eegnet-bci2a-fixed`.

Generate signal, embedding and AGFL plots for every selected checkpoint using
this Bash loop, then add sparsity/entropy-versus-accuracy figures:

```bash
task_results_root=results/eegnet-bci2a-search/selected
task_diagnostics_root=results/eegnet-bci2a-search/analysis/diagnostics
find "$task_results_root" -type f -name result.json -print0 |
while IFS= read -r -d '' task_result_file; do
  task_run_dir=${task_result_file%/result.json}
  task_relative_run=${task_run_dir#"$task_results_root"/}
  python main.py diagnose "$task_run_dir" \
    --output-dir "$task_diagnostics_root/$task_relative_run" \
    --device cuda --partition validation --embedding tsne || break
done
python main.py analyze "$task_results_root" \
  --output-dir results/eegnet-bci2a-search/analysis --plots \
  --diagnostics-root "$task_diagnostics_root"
```

## Compare attention within one model

```bash
python main.py sweep --preset eeg-comparison --model eegnet --output-dir results/eegnet-comparison
python main.py sweep --preset eeg-comparison --model eegencoder --output-dir results/eegencoder-comparison
python main.py sweep --preset ecg-comparison --model conformer --output-dir results/conformer-comparison
```

Each EEG comparison runs **one model × nine subjects × five attentions × five
seeds = 225 runs**, always training subjects separately. Each ECG comparison
runs **one model × five attentions × five seeds = 25 runs** on patient-group splits.
The surrounding model, heads, training policy and splits are held fixed.
Parameter counts may differ and are recorded. Use `run` for one attention.
Identical resolved sweep configurations are executed only once, including when
an explicit ablation repeats the chosen model's default.

## Configuration and ablations

```json
{
  "schema_version": 2,
  "model": "eegnet",
  "attention": "agfl",
  "dataset": "eeg",
  "seeds": [0, 1, 2, 3, 4],
  "device": "cuda",
  "data": {"data_dir": "../ml"},
  "model_options": {"f1": 16, "d": 2, "f2": 32},
  "attention_options": {"heads": 4, "K": 2, "top_k": "scheduled"},
  "training": {"epochs": 250, "learning_rate": 0.005},
  "output_dir": "results/my-study"
}
```

Load JSON/TOML with `--config experiment.json`. Nested overrides use `--set`:

```bash
python main.py run --preset eeg --set attention_options.K=3
python main.py run --preset eeg --set attention_options.top_k=null
python main.py run --preset eeg --set attention_options.agfl_variant=renormalized
python main.py sweep --preset eeg-ablations --model eegnet
python main.py sweep --preset ecg-ablations --model conformer
```

AGFL defaults retain the active original polynomial, learned temperature,
scheduled threshold Top-k, signed zero-initialized coefficients, and separate
per-hop feature projections. `K` is maximum hop order, so there are `K+1` taps.
EEGEncoder defaults to K=3; DSTS uses K=1 and two heads; EEGNet, Conformer and
Signal Transformer use K=2 and four heads. Explicit overrides are recorded.

Ablations cover Top-k counts/dense connectivity, orders 0–3, normalization, stage,
tie policy and coefficients. `paper-renormalized-eeg` selects the
manuscript's Q/K/V, value-only taps and feature-norm recursion as an alternative.
All mathematical settings belong in `attention_options`. Approximation options
are `random_features` (Performer), `projection_rank` (Linformer), and
`landmarks`/`pinv_rtol` (Nyströmformer).

## Protocol and saved results

Default seeds are 0–4. For BCI IV 2a, each subject's T-session trials receive a
persisted, class-stratified 60/20/20 train/validation/test split, rounded per class.
These are **within-subject T-session results**, not official T-to-E session
transfer results. The same subject and seed use identical samples for every
model and attention. A pooled-subject/group-split BCI training request is rejected.
ECG retains patient-group 60/20/20 splitting; records 201 and 202 stay together.
Validation selects the checkpoint; test is evaluated once after restoration.

Training defaults are EEG 250 epochs/LR 0.005/focal gamma 3 and ECG
50 epochs/LR 0.0003/focal gamma 1. AdamW, weight decay .001, batch size 64,
warmup/cosine scheduling and validation accuracy selection are configurable.
Focal loss uses unweighted target probability and training-derived class weights.
ECG presets add training-only shift/scale/noise augmentation. AMP is opt-in;
determinism remains strict. These budgets are starting settings, not tuned claims.

EEG defaults now use the 0–4 s window after cue onset (1,000 samples at 250 Hz),
2–30 Hz filtering **within each trial**, exclusion of expert-marked artifacts,
and channel normalization fitted on the training partition only. This preserves
relative trial amplitudes and avoids filtering across held-out trials. These
settings change the dataset fingerprint. ECG retains 0.5–45 Hz filtering per
record, MLII selection, 256-sample beat windows and per-beat/channel normalization.
NPZ adapters require `[N,C,T]` signals and explicit labels/groups; use
`data.normalization=none` for already-preprocessed arrays when appropriate.

For T-to-E transfer, put the official `A01E.mat` ... `A09E.mat` class labels in
`../ml`, or supply their directory below. It still trains each subject separately;
T is split 80/20 for train/validation and E is used only for test:

```bash
python main.py run --preset eeg-session --model eegnet --attention agfl \
  --set data.labels_dir=../official-labels --output-dir results/eegnet-session
```

Omit the labels override if `.mat` files are beside the GDF files. Missing labels
are an error; they are never inferred. See the [dataset audit](docs/data_audit.md).

```text
<output-dir>/eeg/subject_A01/<model>-<attention>-<experiment_id>/seed_<seed>/
  config.json          complete settings, provenance and metadata
  split.json           indices, sample IDs, subject groups and fingerprints
  history.json         train loss/accuracy, validation metrics, LR, AMP skipped steps
  checkpoint.pt        selected weights, buffers and normalization state
  predictions.npz      validation/test probabilities, targets and sample IDs
  result.json          metrics, parameter counts, model and attention locations
```

Each EEG subject gets its own `subject_Axx` directory. ECG keeps
`<output-dir>/ecg/<model>-<attention>-<experiment_id>/seed_<seed>/`.
EEGNet now has a residual path around attention in both modalities. EEG models
use learned, signed electrode filters for spatial readout rather than uniformly
averaging all electrodes. AGFL equations are unchanged. These are new model
settings and require new training; old checkpoints are reconstructed with their
previous readout behavior for diagnostics, never silently upgraded.

Relaunching the same configuration and seed **overwrites that seed's generated
artifacts by default**, including completed runs. Aborted or failed runs restart
from epoch 1 with fresh weights and optimizer state; no manual deletion is needed.
Old metrics, checkpoints, predictions, histories and failure markers are cleared
before training. Other seeds and unrelated files are preserved. An OS lock prevents
simultaneous writers to the same seed and is released when its process exits.

Use `--skip-completed` to retain matching completed seeds while restarting
interrupted or incomplete seeds. For example:

```bash
python main.py sweep --preset eeg-comparison --model eegnet --skip-completed
```

Regenerate analysis and diagnostics after replacing runs so exported figures
reflect the new artifacts. Replay a v2 run using its saved config with the recorded
source, packages and data; choose a new output root if you want to retain the
previous results. These are inference checkpoints, not optimizer-resume files.

## Tables and all plots

**Training does not generate images automatically.** After training, run both
steps below on the cluster. `analyze --plots` uses saved artifacts only;
`diagnose` additionally loads checkpoints and the original recordings. Neither
trains a model. The CLI prints the result-plot command when training finishes.

Install plotting dependencies in the active Python 3.12+ environment once:

```bash
python -m pip install -e '.[plots,umap]'
```

**1. Result figures and tables for every subject and seed.** From `AGFL`, use
the same result root as the training command:

```bash
python main.py analyze results/eegnet-eeg --output-dir analysis/eegnet-eeg --plots
```

Open **`analysis/eegnet-eeg/figures/index.md`** for PNG previews and PDF links.
The main table is **`analysis/eegnet-eeg/report.md`**. CSV/JSON exports include
`per_subject.csv` (mean/sample SD over seeds for each subject),
`across_subjects.csv` (equal-weight mean of subject means and between-subject SD),
`per_model.csv`, `attention_comparison.csv`, `agfl_ablations.csv`,
`statistical_comparisons.csv`, and `aggregation.json`. Unequal completed seed
sets across subjects are flagged; no overall score is silently fabricated.
Subject summaries identify the subjects actually present, including partial studies.

For the existing downloaded study, whose root was simply `results`, use:

```bash
python main.py analyze results --output-dir analysis/existing-results --plots
```

Old runs remain labeled with their original cohort and protocol; they are not
reinterpreted as individual-subject experiments.

**2. Signal, embedding and attention diagnostics for every completed run.**
This loop discovers the actual subject/experiment/seed paths; there are no IDs
to edit manually. Run it from `AGFL` in Bash on the cluster:

```bash
task_results_root=results/eegnet-eeg
task_diagnostics_root=analysis/eegnet-eeg/diagnostics
find "$task_results_root" -type f -name result.json -print0 |
while IFS= read -r -d '' task_result_file; do
  task_run_dir=${task_result_file%/result.json}
  task_relative_run=${task_run_dir#"$task_results_root"/}
  python main.py diagnose "$task_run_dir" \
    --output-dir "$task_diagnostics_root/$task_relative_run" \
    --device cuda --partition validation --embedding tsne || break
done
```

Each output directory contains **`index.md`**, PNG/PDF files and `manifest.json`.
The loop reloads recordings for each checkpoint, so it takes longer than step 1.
To inspect one run first, copy its directory path from the result tree:

```bash
python main.py diagnose PATH_TO_ONE_SEED_DIRECTORY \
  --output-dir analysis/one-run-diagnostics --device cuda
```

Use `--device cpu` for CPU diagnostics, `--embedding pca|tsne|umap`, and
`--max-samples 512` to change the diagnostic subset (default 256). The default
partition is validation. If recordings moved, add `--data-dir /new/data/path`;
for E-session labels add `--labels-dir /new/labels/path`, or for NPZ data use
`--data-path /new/dataset.npz`. Data content must still match the training run.

**3. Add sparsity/entropy-versus-accuracy figures** after step 2:

```bash
python main.py analyze results/eegnet-eeg --output-dir analysis/eegnet-eeg --plots \
  --diagnostics-root analysis/eegnet-eeg/diagnostics
```

| Figures | Generated by | Required artifacts |
|---|---|---|
| Learning curves, train accuracy, validation Accuracy/AUC/F1, LR, selected epoch | `analyze --plots` | Histories; train accuracy exists only in new runs |
| Validation/test confusion matrices and class ROC curves | `analyze --plots` | Predictions and split manifests |
| Seed summaries, subject scores, capacity plots | `analyze --plots` | Completed results |
| Attention comparisons, paired differences and ablations | `analyze --plots` | Matching baseline/ablation experiments; one AGFL setting cannot provide these comparisons |
| Example signals, spectra, EEG alpha/beta scalp maps, C3/C4 time-frequency maps, training-fitted CSP | `diagnose` | Checkpoint, matching recordings and channel metadata |
| PCA/t-SNE/UMAP, per-head graph maps, AGFL coefficients and projections | `diagnose` | Checkpoint and matching recordings |
| Sparsity/entropy versus accuracy | Step 3 | Matching verified diagnostic manifests |

`manifest.json` lists unavailable figures and their reasons (missing metadata,
undefined entropy, or absent comparison runs). It is not evidence that a missing
baseline was trained. Result statistics use paired t/Wilcoxon tests, effect sizes
and Holm corrections only for matching subject/seed/split/model/protocol pairs.
Repeated overlapping splits make seed-level inference exploratory; see
[statistics](docs/statistics.md).

After overwriting training runs, rerun these plotting steps to refresh exports.
Download the `analysis` directory as well as `results` to obtain the images.
Detailed figure meanings and limits are in [plot coverage](docs/visualization.md).

Earlier v1 results remain readable and are labeled with their actual backbone
and attention without rewriting saved identities. Earlier mechanism-as-model
runs are Signal Transformer runs. Their generic checkpoints can be diagnosed;
v1 original-backbone checkpoints require their training checkout. New training
requires a fresh v2 configuration. Keep a running study's checkout stable.

## Verification and extension

Target-only checks:

```bash
python -m pytest -m 'not real_data'
AGFL_REAL_DATA=1 python -m pytest -m real_data
```

The tests cover every model/attention/modality combination, injection and axes,
common initialization, AGFL formulas and gradients, preserved native layouts
with frozen original weights, strict CUDA pooling, data leakage, replay,
statistics and plotting. They are written but have not been executed locally.
Keep the original Git commit for the source-reference check; details are in
[reference provenance](tests/references/ORIGIN.md).

To extend models, add a model package exposing `SPEC`, defaults and EEG/ECG
constructors using the supplied attention factory. To extend attention, add
an attention package exposing `AttentionSpec` with defaults and a
`[B,N,D] -> [B,N,D]` constructor. Both registries discover packages automatically.
Dataset loaders register separately and return explicit `[N,C,T]` arrays,
contiguous class labels, subject groups and stable sample IDs.

See [current model and preprocessing corrections](docs/subject_model_preprocessing_audit.md),
[pipeline findings](docs/pipeline_audit.md),
[mathematical definitions](docs/mathematics.md), and
[request status](docs/request_status.md) for scientific limits and remaining
target-machine acceptance work.
