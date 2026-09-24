# AGFL research framework

Select a **model** (EEGNet, EEGEncoder, DSTS EEGEncoder or Conformer) and an
independent **attention mechanism**. The selected attention runs inside that
model. Every model owns separate EEG and ECG variants.

This is a source refactor prepared for the experiment machine. No project code,
training, evaluation, installation or tests have been run locally. Numerical
verification and repeated-seed research results remain pending.

## Results to download

Every output session now separates **`report/`** (plots, metrics, predictions,
settings, split/selection records and troubleshooting evidence) from
**`artifacts/`** (canonical training runs and checkpoints kept on the cluster).
**Download only `results/<session>/report/`.** Training commands still receive
`results/<session>` as their `--output-dir`.

For a completed existing folder, run on the cluster:

```bash
python main.py organize-results results/eegnet-A03-agfl-qkv-v2
```

If checkpoint plots are still needed, and the checkout matches the saved model:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-qkv-v2 \
  --device cuda --partition validation --embedding tsne
```

Then refresh the result plots and tables:

```bash
python main.py analyze results/eegnet-A03-agfl-qkv-v2 --plots
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

Use the existing Python 3.12+ PyTorch/CUDA environment. `pytest` is an optional
development dependency, not a requirement for training or plotting. Dependencies
are defined in `pyproject.toml`; keep the installed environment stable.

Request an interactive allocation and open a shell on the allocated GPU node:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

Wait until the allocation is granted before continuing. The current
[A03 EEGNet token-routing comparison](docs/eegnet_agfl_token_routing.md) runs training,
checkpoint diagnostics and result plots with one command. Separate plotting
commands are also provided for recovery. No `.sbatch` file or development
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

**Current priority: restore the previous EEGNet and improve AGFL only.**
The failed power/EMA recipe is withdrawn from the active comparison. EEGNet
again uses its original convolution path, 32-sample temporal kernel and seven
time tokens; training again uses learning rate 0.005, focal gamma 3, no
augmentation, no EMA and the original checkpoint tie rule. Preprocessing is
unchanged. The new AGFL option learns a token-specific mixture of existing
graph hops from each token and its neighborhood contrast. The
[token-routing workflow](docs/eegnet_agfl_token_routing.md) documents the exact
formula, rollback, comparison and commands. No dependencies are added.

From the cluster `AGFL` directory, with the existing environment and GPU
allocation active and `../ml/A03T.gdf` available, launch everything once:

```bash
python main.py sweep --preset eegnet-a03-token-agfl \
  --output-dir results/eegnet-A03-token-agfl-v1 \
  --report
```

This is **one command, 20 fits**: static sparse AGFL, token-routed sparse AGFL,
MHA and Linformer rank 4, each on A03 with seeds 0–4 and 250 epochs. The two
AGFL arms differ only in coefficient routing; its 192 new parameters start
with zero correction. Every method shares the restored EEGNet, training,
preprocessing and per-seed splits. Validation accuracy selects one checkpoint
per method; every selected checkpoint receives test results. The report
compares token-routed AGFL with static AGFL and both attention baselines and
records each class's recall, so feet gains cannot hide tongue/hand losses.

`--report` is reusable with **`run`, `sweep` and `tune-eegnet`**. It generates
validation checkpoint diagnostics with t-SNE, then runs result analysis with
plots. It uses each session's first configured training device (and announces
that choice for a mixed-device session). It does nothing with `--dry-run`.
Without `--report`, the existing separate plotting workflow remains available.
For `run` and `sweep`, add `--skip-completed` when resuming; provenance and
settings must match for completed runs to be reused. `tune-eegnet`
already reuses matching completed candidates by default. Incomplete fits
restart from epoch 1. Keep the source, configuration and environment stable
throughout the study, and use the fresh output directory above.

If automatic reporting is interrupted, completed training is retained. Recover
checkpoint plots without retraining:

```bash
python main.py diagnose-session results/eegnet-A03-token-agfl-v1 \
  --device cuda --partition validation --embedding tsne
```

Then refresh result plots and tables:

```bash
python main.py analyze results/eegnet-A03-token-agfl-v1 --plots
```

Download **`results/eegnet-A03-token-agfl-v1/report/`** into a new local folder. Open
`analysis/report.md` for accuracy summaries, `analysis/figures/index.md` for
result plots, and each run's `index.md` below `diagnostics/` for checkpoint
plots. `analysis/per_class.csv` and `analysis/per_class_summary.csv` give
validation/test class metrics; figures include class recall histories and
comparisons. Token-routed runs save `coefficient_conditioning.json` and
per-trial/head/token/hop coefficient arrays and heatmaps under `filters/`.
These expose routing variation, bounded corrections and the zero-sum
constraint. Keep `artifacts/` on the cluster for diagnostic regeneration.
There is no candidate-selection report in this fixed comparison. New accuracy
gains remain unmeasured; **90% is a target**. A03 has been repeatedly inspected,
so this remains development evidence.

**Retired A03 power/EMA comparison:** AGFL averaged **77.04%**, MHA **76.30%**
and Linformer rank 4 **76.67%**, below their prior original-EEGNet results.
All 15 fits and 346 figure pairs completed; the power branch and EMA were
active. The [power/EMA record](docs/eegnet_power_ema.md) preserves unfavorable
results and reproduction commands. Its isolated reproduction classes keep
old checkpoints inspectable; the active EEGNet has no power branch.

**Completed A03 conditioning comparison:** original and conditioned AGFL
both averaged **84.44%** test accuracy, versus **85.93%** for Linformer rank 4.
Conditioning improved feet but reduced tongue recall; its controller was
active and numerically finite. All 15 fits and 313 figure pairs completed.
The [conditioning workflow](docs/eegnet_agfl_conditioning.md) preserves the
implementation and reproduction commands. Static AGFL remains the reference.

**Completed A03 capacity study:** the validation-selected procedure reached
**84.44%** mean test accuracy (seed SD **3.10 percentage points**). The
15-token/projected variants did not beat the seven-token sparse control's
mean validation accuracy. The [single-file results review](docs/eegnet_a03_capacity_review.html)
records the evidence; the [capacity workflow](docs/eegnet_agfl_capacity.md)
preserves reproduction commands. It is no longer the next launch.

**Completed A03 graph/filter refinement:** validation-selected test accuracy
is **85.56%** (seed SD **2.03 percentage points**). The new temperature and
coefficient initializations did not beat the sparse control's mean validation
accuracy as fixed candidates. The [refinement workflow](docs/eegnet_agfl_refinement.md)
preserves that experiment's settings and reproduction commands.

**Completed A03 all-attention comparison:** fixed sparse AGFL averages
**84.44%** test accuracy, MHA **85.56%**, Performer **84.44%**, Linformer
rank 4/7 **85.93%/79.63%**, and Nyström landmarks 4/7 **51.48%/85.19%**.
AGFL has the highest mean ROC-AUC (**0.9721**), but no accuracy advantage.
All 35 runs and 691 PNG/PDF figure pairs completed. These fixed-setting results
differ from the earlier validation-selected searches. The
[all-attention workflow](docs/eegnet_a03_all_attentions.md) preserves that
experiment's settings and reproduction commands.

**Completed A03 sparsity check:** selecting between three-neighbor and
five-neighbor Q/K/V AGFL again gave **85.56% mean test accuracy**, with
**5.77 percentage points** seed SD. Top-three did not improve validation
accuracy in any seed: four ties and one loss. The earlier three-candidate
Q/K/V search also averaged 85.56%, equal to the historical MHA mean. These
search results do not establish an AGFL advantage. The
[Q/K/V and sparsity workflow](docs/eegnet_agfl_qkv_improvement.md) preserves
the earlier experiment commands; they are not the next launch.

The same improved AGFL module is available inside **all five backbones**.
See [cross-model improvements and cluster checks](docs/agfl_cross_model_improvements.md)
for the shared attention settings, backbone-specific heads/hop orders,
whole-model initialization checks, and additional numerical safeguards.
Coefficient-routing options live inside AGFL, independently of model choice.
Current work remains on the restored EEGNet at the user's request.

**Manuscript reference and comparison scope:** use the latest located
`Downloads/NEU_art_submission.pdf` (4 May 2026), rather than the local TeX
draft. The completed EEGNet comparison included **MHA, AGFL, Performer,
Linformer and Nystromformer**, with four classes and separate subject
training. The A03 five-attention comparison is complete; the next experiment
compares static/token-routed AGFL, MHA and Linformer inside the restored
EEGNet. See
[the PDF reference and pilot rationale](docs/neurips_submission_reference.md).

**Earlier fixed A03 attention comparison:** MHA averages **85.56% test accuracy**,
AGFL **84.81%**, over five matched seeds. AGFL wins three seeds, ties one and
loses one, for a mean difference of **−0.74 percentage points**. AGFL uses
29.67% fewer parameters, but its saved run times are about 1.85 times longer
in this seven-token configuration; these are not dedicated latency
benchmarks. All **221 figures (442 PNG/PDF files)** are present. The
[single-file review](docs/eegnet_a03_attention_review.html) includes comparison
figures, integrity checks, a package-inventory caveat and the full-study
commands. This pilot does not demonstrate AGFL superiority.

**Latest A06 batch comparison:** batch 48 wins seed-0 validation by one trial
(21/43 versus 20/43), but its selected checkpoint scores **25.58% test
accuracy**, versus the previously evaluated control's **32.56%** on the same
test IDs. Final training accuracy reaches 93.23%, while final validation
accuracy is 41.86%; the reported test checkpoint was selected at epoch 25.
This does not establish a fix for A06. All **27 figures (54 PNG/PDF files)**
are present. The [single-file review with all embedded plots](docs/eegnet_a06_batch_review.html)
documents the comparison, artifact checks and limits. No additional plot
generation is needed for the downloaded batch-comparison folder.

**Direct A06 inspection:** local GDF byte inspection confirms matching files,
cue timing, validation labels and artifact exclusions. A06 has weak
class-associated alpha-power differences in its small training partition;
the code also gives all three tested A06 recipes a five-trial final batch
with BatchNorm. Neither observation alone proves the cause of low accuracy.
See [the direct findings and limits](docs/eegnet_a06_direct_findings.md).
Another cluster audit is not needed to establish those recorded findings.

**Reproduce the completed A06 batch-size experiment.** After updating the cluster checkout, run
from `AGFL` with `../ml/A06T.gdf` available. This runs **two fits, one subject,
seed 0, 250 epochs each**, comparing the existing `spatial_control` against
`spatial_batch48`. Only batch size changes: A06's 133 training trials form
48/48/37 batches instead of 64/64/5. All trials are retained and both recipes
still take three optimizer steps per epoch. The model, attention, data split,
preprocessing and other training settings stay the same. The new candidate
is opt-in and does not expand the default five-candidate search.

Train and select by validation; only the selected checkpoint is tested:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 6 \
  --seeds 0 \
  --candidate spatial_control \
  --candidate spatial_batch48 \
  --output-dir results/eegnet-A06-batch-comparison
```

Generate the selected checkpoint's signal, embedding and attention plots:

```bash
python main.py diagnose-session results/eegnet-A06-batch-comparison \
  --device cuda --partition validation --embedding tsne
```

Then generate the result plots and analysis tables:

```bash
python main.py analyze results/eegnet-A06-batch-comparison --plots
```

Download only **`results/eegnet-A06-batch-comparison/report/`**. Candidate
histories and the selection report compare validation performance; result plots
describe the selected checkpoint. The plot indexes are
`report/analysis/figures/index.md` and
`report/diagnostics/selected/eeg/subject_A06/seed_0/index.md`.
An identical training relaunch reuses completed matching candidates and
restarts incomplete ones. Use this new output directory, not a previous
search directory with different candidates.

This single-seed pilot did not improve selected test accuracy. Batch size
affects both gradient updates and BatchNorm statistics, so it does not isolate
BatchNorm as the cause. Keep these commands as the reproduction record;
the result does not support promoting batch 48 as the new default.

**Optional repeatable raw audit:** `audit-eeg` checks a saved T-session run against its
raw recording without training or checkpoint inference. `plot-eeg-audit`
then creates PNG/PDF plots and a self-contained `report.html`. See
[the raw EEG audit commands and outputs](docs/eeg_raw_audit.md).

**Latest A06 compact check:** neither compact candidate beats the spatial
control on seed-0 validation (39.53% and 37.21%, versus 46.51%). The selected
control reproduces **32.56% test accuracy** exactly. The
[A06 compact review](docs/eegnet_a06_compact_review.md) explains the histories,
checkpoint selection and next diagnostic work. It also includes the `analyze`
command for result plots: the downloaded folder contains the 18 checkpoint
diagnostic figures, but no result-plot index or aggregate analysis tables.

To reproduce a raw-audit report inside the organized session, launch it on the cluster:

```bash
python main.py audit-eeg \
  results/eegnet-A06-compact-check/artifacts/selected/eeg/subject_A06/seed_0 \
  --data-dir ../ml \
  --output-dir results/eegnet-A06-compact-check/report/data-audit
```

Then generate the audit plots and single-file report:

```bash
python main.py plot-eeg-audit results/eegnet-A06-compact-check/report/data-audit
```

The session's `report/` download now includes `data-audit/report.html` and its
supporting numerical records. These audit figures are additional to the
classification plots generated by `analyze`.

**Latest A03/A06 check:** the same validation-selected spatial procedure gives
**86.67%** mean test accuracy on A03 and **31.63%** on A06 over five seeds,
compared with 69.63% and 28.37% previously. All 252 result/diagnostic figures
are present in the downloaded study. A06 remains unresolved, and the
nine-subject 75% target is unproven. The
[A03/A06 review](docs/eegnet_a03_a06_review.md) explains the data and error
checks and gives a three-fit A06-only follow-up with separate plot commands.

**Latest A01 result:** validation selection between `spatial_control` and
`spatial_recombine` gives **71.64% mean test accuracy (SD 3.30 percentage
points)** over five seeds, compared with 68.00% for the fixed spatial control.
Augmentation is selected for four seeds. The 75% target remains unmet.
The [augmentation review](docs/eegnet_a01_augmentation_review.md) documents
this result and the commands used for the A03/A06 check above.

**A01 follow-up:** the spatial model averages **68.00% test accuracy (SD 3.30
percentage points)** over seeds 0–4, compared with 49.82% for the earlier A01
model. The following experiment isolated segment augmentation on this spatial
configuration. The [five-seed review and complete plot commands](docs/eegnet_a01_spatial_seeds_review.md)
describe the evidence and outputs. After updating the cluster checkout, run
from `AGFL`:

```bash
python main.py tune-eegnet --data-dir ../ml --subjects 1 --seeds 0 1 2 3 4 \
  --candidate spatial_control --candidate spatial_recombine \
  --output-dir results/eegnet-A01-spatial-augmentation
```

This is **10 validation-only fits and five selected checkpoint test evaluations**,
all for A01. `spatial_recombine` changes only training-trial segment
recombination. Each seed's winner is chosen on validation. All other settings
and trial IDs are held fixed. Its measured A01 result is 71.64%, as summarized
above; the 75% target is still unmet. The candidate is opt-in; the original five-candidate search
below retains its previous scope.

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
and selected settings are in
`results/eegnet-bci2a-search/report/search_report.md` and `search_result.json`.
Generate signal, embedding and AGFL plots for every selected checkpoint:

```bash
python main.py diagnose-session results/eegnet-bci2a-search \
  --device cuda --partition validation --embedding tsne
```

Then generate learning curves, confusion matrices, ROC curves and tables:

```bash
python main.py analyze results/eegnet-bci2a-search --plots
```

Download **`results/eegnet-bci2a-search/report/`** and open
`analysis/figures/index.md` inside it. The report's candidate records contain
validation histories and `selection.json`, without test result files.
Selected settings may differ across seeds/subjects, so generic `analyze`
tables group them by configuration. **Use `search_report.md` for the overall
score of this tuning procedure**, rather than averaging those configuration
groups. For the fixed-preset command, use `results/eegnet-bci2a-fixed` as the
session argument to both plotting commands.

## Compare attention within one model

### Earlier A03 EEGNet pilot: MHA versus AGFL

These commands preserve the completed two-attention pilot. The completed
[A03 all-attention workflow](docs/eegnet_a03_all_attentions.md) records the
five-method comparison; the current launch is the
[EEGNet token-routing comparison](docs/eegnet_agfl_token_routing.md).

After updating the cluster checkout, run from `AGFL` with the Python
environment active and `../ml/A03T.gdf` available. The
`eegnet-a03-attention` preset runs **one subject, two attentions, five seeds
(0–4): ten fits, each for 250 epochs**. It uses the fixed `spatial_control`
backbone and training recipe for both attentions: batch 64, learning rate
0.005, weighted focal loss, training-channel normalization and no data
augmentation. Each seed uses matching trial splits. Checkpoints are selected
by validation accuracy and both attention methods are evaluated on test.

Train:

```bash
python main.py sweep --preset eegnet-a03-attention \
  --output-dir results/eegnet-A03-attention \
  --skip-completed
```

An identical relaunch keeps completed matching runs and restarts incomplete
runs from epoch 1. Omit `--skip-completed` to overwrite matching completed
runs too. Use a new output directory when changing the study settings.

After training finishes, generate signal, embedding and attention plots for
both methods and every seed. Run this block in **Bash**:

```bash
python main.py diagnose-session results/eegnet-A03-attention \
  --device cuda --partition validation --embedding tsne
```

Then generate the training, confusion-matrix, ROC and paired attention
comparison figures and analysis tables:

```bash
python main.py analyze results/eegnet-A03-attention --plots
```

Download only **`results/eegnet-A03-attention/report/`**. Within that folder,
the main result-plot index is `analysis/figures/index.md`; each directory below
`diagnostics` has its own `index.md`. `analysis/attention_comparison.csv`
contains attention summaries across seeds, `analysis/statistical_comparisons.csv`
contains paired AGFL-minus-MHA differences, and `analysis/report.md`
summarizes the results. Both methods' results remain in the report; this
is an attention comparison, not validation selection of one winning method.
A03 is a pilot chosen using previous results, not confirmation of an
all-nine-subject claim. The AGFL recipe matches the current spatial control;
it does not claim to reproduce every setting described in the historical PDF.

### Deferred: extend the fixed attention comparison to all nine subjects

**Do not launch this as the next experiment.** First complete the
[A03 EEGNet token-routing comparison](docs/eegnet_agfl_token_routing.md).
The commands below preserve the earlier fixed-comparison expansion for
reference; a later study must use the chosen AGFL method or selection procedure.

The completed A03 pilot does not establish an across-subject advantage.
Keep the same model, preprocessing, training recipe and seeds, and extend
the sweep. The existing output directory intentionally retains its A03 name
so matching completed A03 runs can be reused: **80 new fits, 90 total** with
unchanged source, dependencies, data and saved settings. This extension is
for the fixed `sweep` command; scope changes in `tune-eegnet` still require
a new search directory.

```bash
python main.py sweep --preset eegnet-a03-attention \
  --set 'data.subjects=[1,2,3,4,5,6,7,8,9]' \
  --output-dir results/eegnet-A03-attention \
  --skip-completed
```

After training completes, generate all checkpoint diagnostics in Bash:

```bash
python main.py diagnose-session results/eegnet-A03-attention \
  --device cuda --partition validation --embedding tsne
```

Then generate the result plots and paired comparison tables:

```bash
python main.py analyze results/eegnet-A03-attention --plots
```

Download only the updated `report/` folder. Report all nine subject means and the
equal-weight average across them, including A06 and every matched seed.
The expanded study is a measurement of the fixed comparison, not a promise
of either a 75% overall score or AGFL superiority.

### Broader model comparisons

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

**Training does not generate images automatically.** After training, run both
steps below on the cluster. `analyze --plots` uses saved artifacts only;
`diagnose` additionally loads checkpoints and the original recordings. Neither
trains a model. The CLI prints the result-plot command when training finishes.

Install plotting dependencies in the active Python 3.12+ environment once:

```bash
python -m pip install -e '.[plots,umap]'
```

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
| PCA/t-SNE/UMAP, per-head graph maps, AGFL coefficients and projections | `diagnose` | Checkpoint and matching recordings |
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

Optional developer checks on the target machine, for environments that already
have `pytest`. These are separate from the training and plotting workflow:

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
