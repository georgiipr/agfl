# One-command EEGNet + AGFL capacity study on A03

**Completed study:** the capacity-selected procedure reached **84.44%** mean
test accuracy (seed SD **3.10 percentage points**). The larger variants did
not beat the seven-token sparse control's mean validation accuracy. See the
[single-file results review](eegnet_a03_capacity_review.html). The next launch
is the [EEGNet power/EMA comparison](eegnet_power_ema.md); commands
below preserve the completed capacity experiment.

The preceding validation-selected AGFL procedure reached **85.56%** mean test
accuracy across seeds 0–4. Learned graph temperature and the initial local
value mixture did not produce a fixed candidate with better mean validation
accuracy than the sparse control. This capacity study increases temporal detail
and per-hop feature capacity, and compares the loss and learning rate together.
**90% remains a target, not an expected or guaranteed result.**

## What is implemented

- EEGNet's second pooling factor changes from 16 to 8 in the four new
  candidates. A 1,000-sample input gives `1000 // 8 // 8 = 15` time tokens,
  compared with seven in the control. The classifier receives 480 features
  instead of 224. Convolution widths and preprocessing are unchanged.
- `attention_options.filter_projection=separate` gives each head a trainable
  feature transform for each polynomial hop: `sum_k alpha_k W_k(A^k V)`.
  The new `filter_projection_init=identity` option initializes each matrix
  to identity without consuming random draws. With the same tokens and
  parameters, this starts from the value-only computation; it does not imply
  equivalence between the seven-token control and the 15-token candidates.
- Focal loss with gamma 3 and cross-entropy are each tried at learning rates
  0.005 and 0.001. Class weighting remains balanced. The recorded focal gamma
  is inactive for cross-entropy. Validation accuracy selects checkpoints;
  losses from different objectives are not compared for selection.
- `--candidate-set capacity` selects the bounded study below. `--report`
  automatically generates validation checkpoint diagnostics with t-SNE and
  then result plots after training and selected-checkpoint evaluation finish.

Identity initialization is an opt-in AGFL feature usable in every backbone,
with shared or separate hop projections. The default remains `pytorch`,
preserving older initialization behavior. Checkpoint keys and shapes for
existing shared/separate projections are unchanged. Identity initialization
does not overwrite weights restored from a checkpoint.

The study retains Q/K/V projections, fixed square-root score scaling, four
heads, polynomial order `K=2`, learnable coefficients initialized to `[0,1,0]`,
and scheduled Top-k. Zero- and two-hop transforms become active as their
initially zero coefficients learn. More tokens also change the nominal
retained-neighbor count from five of seven to twelve of fifteen with this
schedule; threshold ties can retain additional neighbors. These are combined
architecture changes, not an isolated attribution to one mechanism.

## Bounded workload and selection

| Candidate | Time tokens | Hop projections | Loss | Learning rate |
|---|---:|---|---|---:|
| `spatial_qkv_sparse` | 7 | Value-only | Focal, gamma 3 | 0.005 |
| `spatial_qkv_capacity_focal_005` | 15 | Separate, identity initialization | Focal, gamma 3 | 0.005 |
| `spatial_qkv_capacity_focal_001` | 15 | Separate, identity initialization | Focal, gamma 3 | 0.001 |
| `spatial_qkv_capacity_ce_005` | 15 | Separate, identity initialization | Cross-entropy | 0.005 |
| `spatial_qkv_capacity_ce_001` | 15 | Separate, identity initialization | Cross-entropy | 0.001 |

One launch performs **25 fits: five candidates × five seeds**, on A03 only,
250 epochs each. Fits run sequentially on the allocated GPU. This removes
manual launches; it does not turn four training recipes into one optimization
trajectory or guarantee they fit within a four-hour allocation.

Each subject/seed uses identical trial IDs across candidates. Candidate and
checkpoint selection use validation only. Candidate ties use macro F1, then
declared order; the unchanged control comes first. After every choice is
recorded, **only the five selected checkpoints evaluate test**, without a new
training fit. The original default candidate list is unchanged.

This evaluates the selection procedure. Generic analysis may split the five
selected runs into smaller configuration groups; use `search_report.md` for
the overall mean. Comparing that mean with an older fixed MHA result does not
establish superiority under equal tuning. A03's repeatedly examined splits
are development evidence; a general claim needs separate confirmation.

## Upload only the five changed runtime files

On the Mac:

```bash
cd /Users/egor/Downloads/AGFL
scp agfl/attention/agfl/config.py agfl/attention/agfl/layer.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/attention/agfl/
scp agfl/models/eegnet/search.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/models/eegnet/
scp agfl/cli.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp agfl/visualization/diagnostics.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
```

No new preset, library, installation, test-suite run or batch script is needed.

## Allocate resources and activate the existing environment

Skip allocation if you already have a compute-node shell with the requested
GPU resources. Otherwise, from the cluster login shell:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

After allocation succeeds, enter the project directory and activate the
existing Python 3.12+ CUDA environment. Dataset: **`../ml/A03T.gdf`**.

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

## Launch training, selection and plots once

```bash
python main.py tune-eegnet --candidate-set capacity \
  --data-dir ../ml --subjects 3 --seeds 0 1 2 3 4 --epochs 250 \
  --output-dir results/eegnet-A03-agfl-capacity-v1 --report
```

The process continues from training through checkpoint diagnostics and result
plots without waiting for another command. The normal progress bars remain.
Use this fresh output directory. An identical relaunch reuses matching
completed fits; unfinished fits restart from epoch 1. Keep source files,
environment and configurations unchanged while resuming the study. Add
`--dry-run` only for a configuration preview; it does not train or report.

`--report` also works with ordinary `run` and `sweep` commands. Add
`--skip-completed` when resuming those commands; unlike `tune-eegnet`, their
default is to overwrite matching runs. For each session, reporting uses its
first configured training device and announces any mixed-device choice.

## Plot-only recovery commands

These are optional after a successful `--report` launch. If reporting is
interrupted, completed training remains saved and the command exits with an
error. It prints recovery commands instead of silently claiming completion.

Generate or regenerate checkpoint diagnostics without retraining:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-capacity-v1 \
  --device cuda --partition validation --embedding tsne
```

Then generate or refresh result plots and tables:

```bash
python main.py analyze results/eegnet-A03-agfl-capacity-v1 --plots
```

These commands run from the same cluster directory and environment, with the
full session and dataset present. Existing optional diagnostic skips are
recorded in each plot manifest; completion is not a claim that every possible
figure applies to every run. Do not delete checkpoints to recover plotting.

## Reusable outputs

Download **`results/eegnet-A03-agfl-capacity-v1/report/`** only. It contains:

- `search_report.md` and `search_result.json`: overall selected-procedure
  accuracy and per-seed results.
- `selection_report.json`: every candidate's validation score and the choices.
- `analysis/figures/index.md`: learning curves, confusion, ROC and other result
  figures, with PNG/PDF files alongside.
- `diagnostics/selected/eeg/subject_A03/seed_*/index.md`: signals, embeddings,
  graph/filter plots and parameter diagnostics for selected checkpoints.
- Each diagnostic run's `filter_statistics.json`: learned coefficients,
  projection initialization, each projection's norm and distance from identity.
  Projection heatmaps show how the new trainable matrices changed.
- Mirrored configurations, split records and candidate histories for review.

Keep `artifacts/` on the cluster. In particular,
`artifacts/selected/eeg/subject_A03/seed_*/` contains the chosen checkpoint,
complete `config.json`, `split.json` and training history. These are durable
outputs for reconstruction and further evaluation; the report records the
winning candidate and checkpoint hash. The study does not silently promote a
winner into global defaults. Reusing a selected checkpoint for diagnostics
requires no retraining. The named recipes and identity projection option stay
available for later fixed-configuration experiments.

## Verification status

Only static source, configuration, documentation and shell syntax checks were
performed locally. Target-machine tests were added for initial output/RNG
parity, gradient paths, older checkpoint loading, candidate boundaries and
automatic reporting/recovery; they are not experiment launch prerequisites.
No project code, training, inference or project tests ran locally. Numerical
behavior, runtime and accuracy gains remain to be checked on the cluster.
