# Refine AGFL graph weights and local feature retention on A03

**Completed study:** the validation-selected procedure reached 85.56% mean
test accuracy (seed SD 2.03 percentage points). No new fixed candidate beat
the sparse control's mean validation accuracy. These commands preserve the
completed experiment; the next launch is the
[EEGNet power/EMA comparison](eegnet_power_ema.md).

The completed EEGNet comparison gave fixed sparse AGFL **84.44%** mean test
accuracy, versus **85.56%** for MHA and **85.93%** for Linformer rank 4.
AGFL had the highest mean ROC-AUC, but this did not translate into the best
class predictions. Its feet recall was also weaker on validation: **83.08%**
versus **89.23%** for MHA, counting repeated trial occurrences across seeds.
These are development observations, not evidence of a particular fix.

This follow-up changes AGFL's initialization and learnable graph scaling.
It keeps EEGNet, preprocessing, trial splits, loss, learning rate, epoch
budget, four heads, polynomial order `K=2`, and the five-neighbor graph
recipe unchanged. No class-specific correction is fitted to test labels.

## What changes

`attention_options.temperature_init` sets the existing per-head temperature
parameter's initial value. With `score_scaling=temperature`, scores are
`QKᵀ / (sqrt(d) × clamp(temperature, 0.1, 5.0))`. An initial value of 0.5
sharpens retained graph weights; positive scaling preserves the ranking
used by score-stage Top-k. Each head subsequently learns its own temperature.
The default remains 1.0. A nondefault value requires temperature scaling,
so an inert setting cannot silently enter a comparison.

`coefficient_init=identity_one_hop` initializes polynomial coefficients to
`[0.5, 0.5, 0]`, starting the head at `0.5 V + 0.5 A V` before its output
projection. This gives local values an immediate contribution while keeping
the coefficient sum at one. All taps, including the initially zero two-hop
term, remain learnable. This is the projected value V, distinct from EEGNet's
existing outer residual. The option requires `K >= 1` and identity coefficient
activation.

Both are explicit experimental choices within the existing AGFL module.
They do not guarantee an accuracy improvement or establish that averaging
caused the observed classification errors. The temperature option is an
ablation of the paper's square-root-only score scaling.

## Five candidates

| Candidate | Score scaling | Initial temperature | Initial coefficients |
|---|---|---:|---|
| `spatial_qkv_sparse` | Fixed square-root scaling | Inactive | `[0, 1, 0]` |
| `spatial_qkv_temp1` | Learned temperature | 1.0 | `[0, 1, 0]` |
| `spatial_qkv_temp05` | Learned temperature | 0.5 | `[0, 1, 0]` |
| `spatial_qkv_identity_temp1` | Learned temperature | 1.0 | `[0.5, 0.5, 0]` |
| `spatial_qkv_identity_temp05` | Learned temperature | 0.5 | `[0.5, 0.5, 0]` |

The four experimental candidates form a two-by-two comparison; the original
fixed-temperature control is separate because a learnable temperature of
one can change during training. Put the control first so an exact validation
accuracy/F1 tie retains it. The original default candidate list is unchanged.

There are **25 candidate fits**: A03, seeds 0–4, five candidates, 250 epochs.
Candidate training and checkpoint selection use validation. Candidate choice
uses validation accuracy, then macro F1, then declared order. The command
then evaluates **only the five selected checkpoints on test**. It does not
stop before test evaluation, and its overall test mean measures a selection
procedure rather than one fixed recipe.

## Upload only the five changed runtime files

Run on the Mac, from the local project directory. No installation, new
library, test-suite launch or preset upload is required for this follow-up.

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

## Cluster commands

If an allocation is already active, use its compute-node shell. Otherwise,
request one GPU, four CPUs and 10 GB of memory from the login shell:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

After allocation succeeds, activate the existing environment. The dataset is
`../ml/A03T.gdf`, relative to the cluster project directory:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

Train and evaluate the validation-selected checkpoints:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 3 \
  --seeds 0 1 2 3 4 \
  --epochs 250 \
  --candidate spatial_qkv_sparse \
  --candidate spatial_qkv_temp1 \
  --candidate spatial_qkv_temp05 \
  --candidate spatial_qkv_identity_temp1 \
  --candidate spatial_qkv_identity_temp05 \
  --output-dir results/eegnet-A03-agfl-refinement-v1
```

Generate checkpoint diagnostics on validation for all five selected runs:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-refinement-v1 \
  --device cuda --partition validation --embedding tsne
```

Generate result plots and tables:

```bash
python main.py analyze results/eegnet-A03-agfl-refinement-v1 --plots
```

Download **`results/eegnet-A03-agfl-refinement-v1/report/`**. The new session
preserves earlier results. An identical launch reuses completed candidates
and restarts unfinished ones; keep source and environment unchanged during
the study. Adding a resolved configuration field changes new experiment
identities, so do not relaunch this update into an older search directory.

## What to review

- `selection_report.json`: all candidate validation scores and choices.
- `search_report.md` and `search_result.json`: the overall selected procedure,
  including its five test evaluations.
- `analysis/figures/index.md`: training, confusion, ROC and result figures.
- `diagnostics/selected/eeg/subject_A03/seed_*/index.md`: checkpoint figures.
- Each diagnostic run's `filter_statistics.json`: numeric raw/effective
  coefficients and temperatures, initialization settings, learnability and
  clamp status. Active temperatures also get a figure. Inactive temperatures
  are explicitly marked and do not get a misleading temperature plot.

Generic analysis groups distinct selected configurations separately; if seeds
select different candidates, those groups are not five-seed fixed-method
comparisons. Use the search report for the overall selection result.

The next fixed-method comparison must freeze its recipe using validation
evidence and rerun all baseline settings under the same frozen source and
environment. Earlier results retain their recorded provenance and must not
be relabeled to force pairing. A tuned AGFL procedure versus fixed baseline
settings is not a comparison of equally tuned methods. A03's repeatedly
examined splits remain development data; a broader claim requires separate
confirmation across subjects and models.

## Verification status

Local verification is limited to static source/configuration and shell syntax
checks. Target-machine tests cover graph scaling, filter initialization,
gradient paths, checkpoint compatibility and diagnostic parameter reporting;
they are development checks, not launch prerequisites. No training, inference,
checkpoint reconstruction or project tests were run locally. Accuracy and
runtime changes remain unmeasured until the cluster run finishes.
