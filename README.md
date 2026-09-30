# AGFL research framework

Select a **model** (EEGNet, EEGEncoder, DSTS EEGEncoder or Conformer) and an
independent **attention mechanism**. The selected attention runs inside that
model. Every model owns separate EEG and ECG variants.

The current EEG architecture preserves electrode identity until attention;
ECG attention remains temporal. This revision was checked statically only.
No project code, training, inference or tests were run locally. New accuracy
and numerical behavior remain unverified until the cluster run.

## Results to download

Every output session now separates **`report/`** (plots, metrics, predictions,
settings, split/selection records and troubleshooting evidence) from
**`artifacts/`** (canonical training runs and checkpoints kept on the cluster).
**Download only `results/<session>/report/`.** Training commands still receive
`results/<session>` as their `--output-dir`.

For a completed existing folder, run on the cluster:

```bash
python main.py organize-results results/eegnet-A03-interchannel-v1
```

If checkpoint plots are still needed, and the checkout matches the saved model:

```bash
python main.py diagnose-session results/eegnet-A03-interchannel-v1 \
  --device cuda --partition validation --embedding tsne
```

Then refresh the result plots and tables:

```bash
python main.py analyze results/eegnet-A03-interchannel-v1 --plots
```

Keep a running study's checkout and folders stable until it finishes. For an
older checkout, generate missing checkpoint diagnostics with that checkout
before reorganizing the finished folder. Organization does not retrain models,
change metrics or rewrite saved configurations. See the
[session layout and migration guide](docs/result_sessions.md) for details.

## Models and attention

| Selection | Keys |
|---|---|
| `model` / `--model` | `eegnet`, `eegencoder`, `dstseegencoder`, `conformer`, `signal_transformer` |
| `attention` / `--attention` | `agfl`, `mha`, `performer`, `linformer`, `nystromformer` |
| Dataset | `eeg`, `ecg`; additional adapters `npz_eeg`, `npz_ecg` |
| Model settings | `model_options`: the chosen backbone's convolutions, widths, depth and dropout |
| Attention settings | `attention_options`: heads, AGFL K/Top-k, approximation rank, etc. |

`mha` defaults to vanilla multi-head self-attention. Explicit
`attention_options.output_gate=true` selects its gated control for EEG or ECG.
`temporal_bias=true` is allowed only on temporal tokens; it is rejected for EEG.
Reports label these variants separately. `signal_transformer` is the
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

EEG requires electrode tokens and learned spatial readouts; ECG uses time tokens. These adaptations
change surrounding feature extraction and readout where needed. See
[model structure and architecture changes](docs/model_attention_structure.md).
Preserving AGFL's equations does not imply full-original-backbone parity.
Temporal EEG presets are retired. See the [inter-channel contract](docs/eeg_interchannel.md).

## Launch on the experiment machine

Commands below are for the cluster, from inside `AGFL`, with datasets beside it:

```text
root/
  AGFL/
  ml/                                  A01T.gdf ... A09T.gdf
  mit-bih-arrhythmia-database-1.0.0/     recordings and annotations
```

Use the existing Python 3.12+ PyTorch/CUDA environment. `pytest` is an optional
development dependency, not a requirement for training or plotting. Dependencies
are defined in `pyproject.toml`; keep the installed environment stable.

Request an interactive allocation and open a shell on the allocated GPU node:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

Wait until the allocation is granted before continuing. The current
[inter-channel EEGNet guide](docs/eeg_interchannel.md) provides separate training,
checkpoint-diagnostic and result-plot commands. `--report` optionally generates
both plot families automatically after training. No `.sbatch` file or development
test run is required.

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
All epoch metrics are still saved to `history.json`. The existing environment supplies `tqdm`.

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

## Current EEG experiment: inter-channel EEGNet

The current queued experiment compares **base AGFL and MHA across all five EEG
models on A03, A04 and A09**, five seeds each: **150 fits**. Submit the single
[`eeg_A03_A04_A09_accuracy.sbatch`](eeg_A03_A04_A09_accuracy.sbatch) file from the
updated cluster checkout. The [queued accuracy guide](docs/eeg_three_subject_accuracy.md)
contains the changed-file upload, submission and single-CSV download commands.
It verifies 22-electrode attention inputs on the cluster, averages seeds then
subjects equally, and writes `report/accuracy_table.csv`. This requested run
generates **no plots or checkpoint files**; selected weights stay in memory
until test evaluation. No plotting commands or test-suite prerequisites apply.
If the 24-hour allocation is insufficient, use the separate
[`eeg_A03_A04_A09_continue.sbatch`](eeg_A03_A04_A09_continue.sbatch) file to
retain completed matching fits and finish the original study. The guide includes
upload and dependent-submission commands. Keep the original `agfl/` source and
environment unchanged throughout; the fresh-launch file retrains all 150 fits.

For the focused **Conformer-only** comparison, use
[`conformer_A03_A04_A09_attentions.sbatch`](conformer_A03_A04_A09_attentions.sbatch):
base AGFL, MHA and Performer on A03/A04/A09, five seeds each (**45 fits**).
This single file includes automatic continuation and writes an accuracy table
without plots or checkpoints. The [Conformer guide](docs/conformer_three_subject_attentions.md)
contains the one-file upload, submission, continuation and table-download commands.

The next focused comparison uses **Signal Transformer** with the same three
attentions, subjects and seeds (**45 fits**). Upload only
[`signal_transformer_A03_A04_A09_attentions.sbatch`](signal_transformer_A03_A04_A09_attentions.sbatch).
Its [experiment guide](docs/signal_transformer_three_subject_attentions.md)
includes submission, log monitoring, continuation and CSV-download commands.
Both Transformer blocks operate on 22 electrode tokens.

**All new EEG attention graphs operate across electrodes.** ECG attention
continues to operate over time. The earlier EEG temporal studies are archived;
their accuracy and seven-node graphs are not electrode-graph results.

EEGNet now preserves its full-channel depthwise spatial convolution before ELU
and pooling, retaining 224 classifier features for a 1000-sample trial. A parallel
branch encodes each electrode separately, applies the chosen attention to
`[batch, 22, 32]`, and contributes another 32 features to the classifier.
Both branches share the first temporal convolution; AGFL and MHA use the same
backbone. Every attention head has a 22×22 electrode graph. Changing to
`attention_axis=time` or enabling `temporal_bias` for EEG is rejected before
training. Temporal convolutions inside a single electrode remain part of its
feature extractor.

The [complete inter-channel guide](docs/eeg_interchannel.md) contains normal-file
upload commands, allocation/environment setup, a two-epoch check, single-subject
training, the full nine-subject comparison, and every plotting command.
No new libraries or local project execution are required.

The preceding focused EEGNet repair experiment uses **A03 only**, with **base AGFL and base MHA**, five
seeds each, 250 epochs: **10 fits**. This tests the spatial-path repair against
the strongest comparator from the preceding report. The classifier now receives
256 features instead of forcing both mechanisms through a 32-feature bottleneck.
The [repair guide](docs/eegnet_spatial_fusion.md) explains the diagnosed issues,
exact architecture, changed-file upload, cluster setup and checkpoint compatibility.
Base AGFL uses the original layer defaults: globally learned hop coefficients,
split-input graph construction, learned temperature and separate hop projections.
Its settings are identical to the base AGFL arm in the A03/A04 experiment.
Base MHA has no added gate or temporal bias. Both arms use 22-electrode graphs
and the same EEGNet backbone, preprocessing, training settings and per-seed splits.
The improvement is a hypothesis to evaluate, not an already measured gain.
Working directory is
`/beegfs/home/georgii.promyslov/AGFL`, dataset is `../ml`, and the existing
Python environment/GPU allocation must be active.

```bash
python main.py sweep --preset eegnet-a03-spatial-fusion \
  --output-dir results/eegnet-A03-spatial-fusion-v1
```

Generate checkpoint plots, including electrode graphs and routing coefficients:

```bash
python main.py diagnose-session results/eegnet-A03-spatial-fusion-v1 \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

Generate result plots and accuracy tables:

```bash
python main.py analyze results/eegnet-A03-spatial-fusion-v1 --plots
```

Download `results/eegnet-A03-spatial-fusion-v1/report/`. Keep `artifacts/` on the
cluster. PCA is the fast diagnostic default in these commands; use
`--embedding tsne` if desired. Classification accuracy is unaffected.

New EEGNet configurations default to `model_options.electrode_architecture=spatial_fusion`.
The five preceding named `eegnet-*interchannel*` / `eegnet-a03-base-agfl-attentions`
presets explicitly retain `compact`, so rerunning an old study does not silently
change its architecture. This also preserves the current `tune-eegnet` search.
The earlier nine-subject `eegnet-interchannel-attentions` experiment (315 fits)
and its commands remain documented in the inter-channel guide; it uses the
previous compact layout, not the new spatial-fusion repair.
The earlier A03-only preset `eegnet-interchannel-comparison` remains available;
it compares gated AGFL, gated MHA and plain MHA and is a different experiment.
`eegnet-a03-a04-interchannel` retains the completed base/enhanced AGFL versus
base MHA design for A03 and A04. Its original settings and results are preserved.

For accuracy numbers alone, run `analyze SESSION` without `--plots`; send
`SESSION/report/analysis/subject_study.json` for a declared comparison study.
Checkpoint diagnostics are not required to calculate or report accuracy.

EEG maps use electrode labels, class-specific 22×22 matrices, scalp source
weights, directed connection figures and complete edge CSV/NPZ exports.
Gate plots are under `electrode_gate/`. These show learned sensor-level routing,
not anatomical or causal brain connectivity. New models need fresh training;
no existing checkpoint is silently converted.

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
<output-dir>/artifacts/eeg/subject_A01/<model>-<attention>-<experiment_id>/seed_<seed>/
  config.json          complete settings, provenance and metadata
  split.json           indices, sample IDs, subject groups and fingerprints
  history.json         train loss/accuracy, validation metrics, LR, AMP skipped steps
  checkpoint.pt        selected weights, buffers and normalization state
  predictions.npz      validation/test probabilities, targets and sample IDs
  result.json          metrics, parameter counts, model and attention locations
```

Each EEG subject gets its own `subject_Axx` directory. ECG keeps
`<output-dir>/artifacts/ecg/<model>-<attention>-<experiment_id>/seed_<seed>/`.
Small run records are mirrored under `<output-dir>/report/runs/`, preserving
this hierarchy without checkpoint files. Search runs use `artifacts/candidates`
and `artifacts/selected`, with downloadable records under `report/runs`.
Search-level summaries are also available in `report/`.

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

**Training generates images automatically only with `--report`.** Otherwise run both
steps below on the cluster. `analyze --plots` uses saved artifacts only;
`diagnose` additionally loads checkpoints and the original recordings. Neither
trains a model. The CLI prints the result-plot command when training finishes.

Use the existing plotting environment. PCA and t-SNE require no new optional library.

**1. Signal, embedding and attention diagnostics for every completed run.**
From `AGFL` on the cluster, use the session root passed to the training command:

```bash
python main.py diagnose-session results/eegnet-eeg \
  --device cuda --partition validation --embedding tsne
```

This discovers subject/experiment/seed paths automatically. For a tuning search,
it processes selected checkpoints and skips validation-only candidates. Each
output directory below **`results/eegnet-eeg/report/diagnostics`** contains an
**`index.md`**, PNG/PDF files and `manifest.json`. Checkpoint diagnostics reload
recordings for each run. To inspect one checkpoint first, pass its directory
under `artifacts/`; its default output goes to the matching `report/diagnostics`
path:

```bash
python main.py diagnose PATH_TO_ONE_ARTIFACT_SEED_DIRECTORY \
  --device cuda --partition validation --embedding tsne
```

Use `--device cpu` for CPU diagnostics, `--embedding pca|tsne|umap`, and
`--max-samples 512` to change the diagnostic subset (default 256). The default
partition is validation. If recordings moved, add `--data-dir /new/data/path`;
for E-session labels add `--labels-dir /new/labels/path`, or for NPZ data use
`--data-path /new/dataset.npz`. Data content must still match the training run.

**2. Result figures and tables for every subject and seed.**

```bash
python main.py analyze results/eegnet-eeg --plots
```

Open **`results/eegnet-eeg/report/analysis/figures/index.md`** for PNG previews
and PDF links. The main table is **`report/analysis/report.md`**. CSV/JSON exports
include `per_subject.csv` (mean/sample SD over seeds for each subject),
`across_subjects.csv` (equal-weight mean of subject means and between-subject SD),
`per_model.csv`, `attention_comparison.csv`, `agfl_ablations.csv`,
`statistical_comparisons.csv`, and `aggregation.json`. Unequal completed seed
sets across subjects are flagged; no overall score is silently fabricated.
Subject summaries identify the subjects actually present, including partial studies.

The declared nine-subject study additionally writes `subject_study.md`,
`subject_study.json`, `subject_study_per_subject.csv`,
`subject_study_averages.csv` and `subject_study_comparisons.csv`. These verify
the expected subjects, seeds and configurations even when a whole group is
missing. Their paired tests compare subject means; the older
`statistical_comparisons.csv` remains an exploratory seed-level breakdown.

Checkpoint diagnostics in `report/diagnostics` are discovered automatically
for sparsity/entropy-versus-accuracy figures. If they were written elsewhere,
add `--diagnostics-root PATH`. An explicit `--output-dir` still overrides the
analysis destination. Result-only plots can be generated before diagnostics;
rerun `analyze --plots` afterwards to include the diagnostic comparison figures.

For an old completed study whose session root was simply `results`, organize
that directory first, then analyze it:

```bash
python main.py organize-results results
```

```bash
python main.py analyze results --plots
```

Use the actual session folder rather than a parent that contains several
unrelated sessions. Old runs remain labeled with their original cohort and
protocol; they are not reinterpreted as individual-subject experiments.

| Figures | Generated by | Required artifacts |
|---|---|---|
| Learning curves, train accuracy, validation Accuracy/AUC/F1, LR, selected epoch | `analyze --plots` | Histories; train accuracy exists only in new runs |
| Validation/test confusion matrices and class ROC curves | `analyze --plots` | Predictions and split manifests |
| Seed summaries, subject scores, capacity plots | `analyze --plots` | Completed results |
| Attention comparisons, paired differences and ablations | `analyze --plots` | Matching baseline/ablation experiments; one AGFL setting cannot provide these comparisons |
| Example signals, spectra, EEG alpha/beta scalp maps, C3/C4 time-frequency maps, training-fitted CSP | `diagnose` | Checkpoint, matching recordings and channel metadata |
| PCA/t-SNE/UMAP, electrode-labeled/class-specific graph maps and edge tables, scalp connections, AGFL coefficients and gates | `diagnose` | Checkpoint and matching recordings |
| Sparsity/entropy versus accuracy | Step 2, after diagnostics | Matching verified diagnostic manifests |

`manifest.json` lists unavailable figures and their reasons (missing metadata,
undefined entropy, or absent comparison runs). It is not evidence that a missing
baseline was trained. Result statistics use paired t/Wilcoxon tests, effect sizes
and Holm corrections only for matching subject/seed/split/model/protocol pairs.
Repeated overlapping splits make seed-level inference exploratory; see
[statistics](docs/statistics.md).

After overwriting training runs, rerun these plotting steps to refresh exports.
Download only **`results/eegnet-eeg/report/`**, including its subdirectories.
That folder contains both plot families and the small records needed to explain
scores and issues. Keep checkpoints in `artifacts/` on the cluster. A downloaded
report alone can regenerate result plots using `analyze /path/to/report --plots`
on a designated analysis machine; checkpoint diagnostics require the recordings
and artifacts. Detailed figure meanings and limits are in
[plot coverage](docs/visualization.md) and the
[session guide](docs/result_sessions.md).

Earlier v1 results remain readable and are labeled with their actual backbone
and attention without rewriting saved identities. Earlier mechanism-as-model
runs are Signal Transformer runs. Their generic checkpoints can be diagnosed;
v1 original-backbone checkpoints require their training checkout. New training
requires a fresh v2 configuration. Keep a running study's checkout stable.

## Verification and extension

Development tests remain separate from training and plotting. They have not
been run locally; the cluster runtime is unverified. Do not install a test
runner merely to launch an experiment. Historical source references are in
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
