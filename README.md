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

EEG uses electrode tokens by default; ECG uses time tokens. These adaptations
change surrounding feature extraction and readout where needed. See
[model structure and architecture changes](docs/model_attention_structure.md).
Preserving AGFL's equations does not imply full-original-backbone parity.

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

One model with one attention, using all nine EEG subjects and five seeds:

```bash
python main.py run --preset eeg --model eegnet --attention agfl --output-dir results/eegnet-eeg
```

A short execution check with three subjects and one seed:

```bash
python main.py run --preset eeg --model eegnet --attention agfl --seeds 0 \
  --set 'data.subjects=[1,2,3]' --set training.epochs=2 \
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
Inspect available keys and resolved settings:

```bash
python main.py list
python main.py presets
python main.py plan --preset eeg --model dstseegencoder --attention performer
```

Paths resolve from the launch directory. Override `data.data_dir` for another
location. EEG E-session evaluation also requires official `AxxE.mat` labels;
labels are never inferred. ECG selects MLII by name and records missing leads.

## Compare attention within one model

```bash
python main.py sweep --preset eeg-comparison --model eegnet --output-dir results/eegnet-comparison
python main.py sweep --preset eeg-comparison --model eegencoder --output-dir results/eegencoder-comparison
python main.py sweep --preset ecg-comparison --model conformer --output-dir results/conformer-comparison
```

Each command runs **one model × five attentions × five seeds = 25 runs**.
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

Default seeds are 0–4. Persisted 60/20/20 subject-group splits are reused for
matching dataset settings and seeds. Nine EEG subjects yield five/two/two
subjects. ECG records 201 and 202 stay together as one patient.
Validation selects the checkpoint; test is evaluated once after restoration.

Training defaults are EEG 250 epochs/LR 0.005/focal gamma 3 and ECG
50 epochs/LR 0.0003/focal gamma 1. AdamW, weight decay .001, batch size 64,
warmup/cosine scheduling and validation accuracy selection are configurable.
Focal loss uses unweighted target probability and training-derived class weights.
ECG presets add training-only shift/scale/noise augmentation. AMP is opt-in;
determinism remains strict. These budgets are starting settings, not tuned claims.

For subject-dependent EEG use `data.subjects=[1]` and
`split.protocol=stratified`. For T-to-E transfer use `data.sessions=["T","E"]`,
the official labels and `split.protocol=session`. See the
[dataset audit](docs/data_audit.md) for filtering, normalization and exclusions.

```text
results/<dataset>/<model>-<attention>-<experiment_id>/seed_<seed>/
  config.json          complete settings, provenance and metadata
  split.json           indices, sample IDs, subject groups and fingerprints
  history.json         losses, validation metrics, LR and AMP skipped steps
  checkpoint.pt        selected weights, buffers and normalization state
  predictions.npz      validation/test probabilities, targets and sample IDs
  result.json          metrics, parameter counts, model and attention locations
```

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

```bash
python main.py analyze results/eegnet-comparison --output-dir analysis/eegnet --plots
python main.py diagnose results/eegnet-comparison/eeg/eegnet-agfl-EXPERIMENT_ID/seed_0 \
  --output-dir analysis/eegnet-diagnostics
```

Replace `EXPERIMENT_ID` with the actual directory suffix.
Analysis exports CSV/JSON/Markdown, mean/sample SD, paired t/Wilcoxon tests,
effect sizes, Holm-adjusted p-values and pairing diagnostics. Comparisons hold
the actual backbone fixed. Repeated overlapping splits make seed-level inference
exploratory; see [statistics](docs/statistics.md).

`analyze --plots` produces learning curves, confusion/ROC, seed comparisons,
ablations, capacity plots, paired differences and available per-subject metrics.
It only reads saved artifacts. `diagnose` loads one checkpoint and recordings
for signal spectra, scalp/CSP maps, embeddings, attention maps and AGFL
coefficients/projections. It defaults to a validation subset and CPU.
Both export PNG/PDF and an index. See [plot coverage](docs/visualization.md).

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

See [pipeline findings](docs/pipeline_audit.md),
[mathematical definitions](docs/mathematics.md), and
[request status](docs/request_status.md) for scientific limits and remaining
target-machine acceptance work.
