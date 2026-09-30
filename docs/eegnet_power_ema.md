> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# Retired experiment: EEGNet power features and EMA

**Completed and withdrawn from the active recipe.** This experiment reduced
AGFL's mean test accuracy from 84.44% to **77.04%**; MHA reached **76.30%**
and Linformer rank 4 **76.67%**, also below their original-EEGNet results.
All 15 fits and 346 figure pairs completed. The feature branch and EMA were
active, and checkpoint diagnostics reproduced the saved predictions.

The active launch now restores the original EEGNet and changes AGFL only:
[token-wise hop routing](eegnet_agfl_token_routing.md). This page preserves
the failed **`eegnet-a03-power-ema`** recipe and reproduction commands, not a
recommendation to run it again. Its combined changes do not identify which
component caused the regression.

The normal EEGNet backbone has been restored. Only explicit archived
`temporal_statistics=mean_logvar` configurations use the isolated classes in
`agfl/models/eegnet/reproduction.py`, preserving old checkpoint keys and
calculations for diagnostics. EMA remains an explicit historical option;
the active comparison sets it to zero.

## What the preceding comparison established

All 15 fits in `eegnet-A03-agfl-conditioned-v1` completed. Saved predictions
reproduce the metrics below; all three methods used the same per-seed splits.

| Method | Mean test accuracy ± seed SD | Macro ROC-AUC | Macro F1 |
|---|---:|---:|---:|
| Original sparse AGFL | 84.44% ± 5.94 pp | 0.9721 | 0.8395 |
| Trial-conditioned AGFL | 84.44% ± 5.80 pp | 0.9690 | 0.8387 |
| Linformer rank 4 | 85.93% ± 9.13 pp | 0.9652 | 0.8535 |

Conditioning changed predictions and learned varying coefficients, but its
12 corrected test errors were offset by 12 newly introduced errors across
the repeated seed evaluations. It is retained for reproduction, not used in
the new primary recipe. The static sparse AGFL attention settings are retained.

The class weakness is visible in **validation**, before using test errors to
describe performance: static AGFL's selected-checkpoint recalls are left hand
91.43%, right hand 97.14%, feet 83.08%, tongue 96.92%. Feet account for **11
of 21 validation errors**, although each training class has 40–41 examples.
The label means **both-feet motor imagery**, as defined in the
[official dataset description](https://www.bbci.de/competition/iv/desc_2a.pdf).

Descriptively, test feet recall rose from 66.15% to 75.38% with conditioning,
but tongue recall fell from 90.77% to 76.92%. A class-specific threshold or
weight adjustment would risk repeating this exchange. No such adjustment
is fitted here. The five splits overlap: 270 test occurrences represent 177
unique trials. Reaching a 90% mean on equal-sized splits would require at
least 15 net additional correct occurrences over the current 228/270.

Late training accuracy averages 95.97%, compared with 87.10% validation
accuracy over the final 25 epochs. Selected validation peaks are transient;
all five conditioned runs selected a unique one-epoch maximum. These findings
motivate better features, training regularization and smoother weight selection,
rather than simply more epochs. Training-mode dropout and the reused small
validation set limit how literally the training/validation gap can be interpreted.

## Retired changes retained for reproduction

### 1. Longer EEGNet temporal filters

The first temporal kernel is **125 samples (0.5 seconds at 250 Hz)** instead
of 32 samples. This gives each learned temporal filter a longer signal context.
The EEGNet authors recommend considering sampling rate when choosing kernel
length and describe a half-second default in their
[original implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py).
This is an explicit recipe change, not a claim that a 32-sample kernel is
invalid or that the longer one must improve accuracy.

### 2. Log-variance features before attention

`model_options.temporal_statistics=mean_logvar` retains the existing EEGNet
waveform path and adds a compact statistics branch. For the spatial
Conv+BatchNorm output **before ELU, pooling and dropout**, it computes
population variance in seven contiguous bins covering the complete trial:

```text
bin edges for 1,000 samples: [0, 142, 285, 428, 571, 714, 857, 1000]
power_features = LayerNorm_features(log(mean_time((x - mean_time(x))²) + 1e-6))
fused_tokens   = original_tokens + Linear_32_to_32(power_features)
output         = EEGNet readout(selected_attention(fused_tokens) + fused_tokens)
```

The log/normalization/projection calculations use float32 under reduced
precision, retaining float64 for reference checks. LayerNorm has no learned
affine parameters. The projection starts at zero and consumes no initialization
random draws, so enabling this branch at a fixed kernel length initially
preserves the original function. The full new recipe also changes kernel length.

The added features pass through **the chosen attention**, with the same
residual arrangement as the original model. There is no separate classifier
that bypasses attention. Feature width stays 32, token count stays seven,
and the classifier size stays unchanged. The branch adds 1,024 parameters;
the longer kernel adds 1,488, for **2,512 additional parameters** compared with
the corresponding old attention configuration.

Explicit variance aggregation is motivated by motor-imagery work such as
[FBCNet](https://arxiv.org/abs/2104.01233). This remains a modified EEGNet,
not an implementation of FBCNet. Existing ELU/average features can already
encode amplitude; the new path supplies an explicit statistic rather than
fixing a proven complete loss of power information. Its full-trial bin edges
differ from the original floor-pooling intervals. Convolutional receptive
fields extend beyond those intervals, so it would be incorrect to claim that
the old path completely ignored its last 104 raw samples.

### 3. Training-only augmentation and a simpler loss

Use random common-channel shifts of up to **25 samples (0.1 seconds)** with
zero padding, and a common amplitude multiplier in **[0.9, 1.1]**. These
operations use only the current training trial and preserve its label and
relative channel amplitudes. Validation, calibration and test are unaugmented.
Segment recombination and additive noise remain off.

Use balanced **cross-entropy**, AdamW at **0.001**, the existing weight decay
0.001 and warmup/cosine schedule, batch 64, 250 epochs. This removes the
gamma-3 focal reweighting of hard examples. Class counts are already nearly
balanced; no special feet penalty is introduced. Cross-entropy and the lower
step size are a declared joint recipe choice, not an established improvement
in isolation. The earlier capacity study changed several different factors
and did not isolate this recipe.

### 4. EMA weights with training-only BatchNorm recalibration

`training.ema_decay=0.99` maintains a second model during training:

```text
ema_parameters = 0.99 * ema_parameters + 0.01 * live_parameters
```

It starts as a copy of the initialized model and updates after successful
optimizer steps and max-norm clipping. Buffers are copied, not averaged.
Before each validation evaluation, BatchNorm statistics for the EMA model
are reset and recalculated using **all unaugmented training inputs only**.
Dropout is disabled; the calibration dataset never reads labels. The final
partial training batch is included, with batch statistics weighted by sample
count. This is a weighted average of batch variances, not exact pooled
population variance. PyTorch's
[weight-averaging documentation](https://docs.pytorch.org/docs/2.6/optim.html#weight-averaging-swa-and-ema)
explains why averaged models need appropriate BatchNorm statistics.

The live model's state, training loader and random streams remain isolated
from calibration. Validation selects EMA checkpoints by accuracy, then macro
F1, then lower loss on ties (`checkpoint_tiebreaker=f1_loss`). The checkpoint
contains exactly the calibrated EMA parameters and buffers that were evaluated.
Final validation, test and diagnostics load that checkpoint unchanged.
`checkpoint_policy` records these details in histories, checkpoints and results.
Plots label live training metrics separately from EMA validation metrics.

EMA adds a second training copy and a training-input forward pass each epoch;
inference still uses one model. It does not ensemble several separately trained
models or consult test data. The default remains `ema_decay=0`, with the old
first-best tie behavior unless explicitly changed.

### 5. Class-specific evidence in every report

Metrics now include confusion counts and per-class recall, precision, F1 and
support. The analysis exports `per_class.csv` and `per_class_summary.csv`, adds
class recall to `report.md`, and generates validation/test class-comparison
plots plus validation class-recall histories. Summaries average per-seed rates;
they do not treat repeated trials as independent observations. Undefined recall
or precision remains null. Older results without these fields remain readable.

Checkpoint diagnostics add `temporal_statistics.json`, numeric arrays with
trial IDs and bin edges, per-class latent-feature heatmaps and a learned
projection heatmap. These show whether the new branch is active and how its
features vary. They do not identify causal physiological features.

## Data, comparison and interpretation

The data loader retains cue-relative 0–4 seconds, 22 EEG channels, artifact
exclusion, trial-local 2–30 Hz filtering and training-only channel normalization.
Cue labels 769–772 map to the four classes; EOG channels are excluded. The
static review found no reason to change labels, cohorts or filtering here.
The 60/20/20 per-subject splits are retained: A03 has 162/54/54 trials per seed.

One launch runs **15 fits**, all on EEGNet/A03: static sparse AGFL, MHA and
Linformer rank 4, each with seeds 0–4 and 250 epochs. Every method gets the
same new backbone, augmentation, loss, EMA and checkpoint-selection policy.
There is no candidate search, no no-attention arm, and no seed cherry-picking.

This measures the combined revised recipe and its within-backbone attention
comparison; it cannot attribute any gain to one component without a later
ablation. Previously inspected A03 results remain development evidence.
Ninety percent and an AGFL advantage are targets, not guaranteed outcomes.
Look for increased average accuracy and feet recall without a compensating
tongue/hand regression, and compare AGFL with both controls. All outcomes remain
in the report. Other backbone work is deferred at the user's request.

## Compatible source for archived checkpoints

Use the coherent source update described in the
[current workflow](eegnet_agfl_token_routing.md#upload-the-prepared-update).
Its restored EEGNet factory and `reproduction.py` must travel together.
The previous ten-file partial upload omitted this new compatibility module
and is no longer a sufficient update procedure.

No dependency installation, development test run or batch script is required.
For exact historical training reproduction, retain the original training
checkout, environment and provenance. A later checkout may replay compatible
weights while recording a source mismatch; it does not become the old source.

## Archived cluster launch — reproduction only

The commands below describe the completed experiment. Use a different output
directory if deliberately reproducing it from a new source checkout, and
replace the session path in both plot commands accordingly. Do not overwrite
the original result record to test another recipe.

Obtain a GPU shell unless already allocated:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

After allocation succeeds, use the existing environment and project directory.
Dataset: **`../ml/A03T.gdf`**. Output: **`results/eegnet-A03-power-ema-v1`**.

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg

python main.py sweep --preset eegnet-a03-power-ema \
  --output-dir results/eegnet-A03-power-ema-v1 \
  --skip-completed --report
```

This trains all methods sequentially, using the existing progress bars, then
generates checkpoint diagnostics and result plots. EMA costs extra computation;
the four-hour request is not a completion-time guarantee. An identical relaunch
keeps completed matching fits and restarts incomplete ones from epoch 1.

Separate checkpoint-diagnostic command, for plotting recovery:

```bash
python main.py diagnose-session results/eegnet-A03-power-ema-v1 \
  --device cuda --partition validation --embedding tsne
```

Separate result-plot command:

```bash
python main.py analyze results/eegnet-A03-power-ema-v1 --plots
```

## Outputs to download

Download only **`results/eegnet-A03-power-ema-v1/report/`**, into a new local
folder such as `Downloads/eegnet-A03-power-ema-v1-report/`. Merging successive
downloads into `Downloads/report/` leaves old files behind, as happened with
the capacity and conditioning reports. Existing research artifacts need not
be deleted. Keep `artifacts/` and checkpoints on the cluster.

Within the report:

- `analysis/report.md`: overall results and per-class recall.
- `analysis/per_class.csv`: individual-seed validation/test class metrics.
- `analysis/per_class_summary.csv`: per-configuration mean/SD and coverage.
- `analysis/statistical_comparisons.csv`: matched AGFL-versus-MHA/Linformer results.
- `analysis/figures/index.md`: all result figures, including class recall.
- `diagnostics/eeg/subject_A03/<model-attention-id>/seed_*/index.md`: checkpoint plots.
- Each diagnostic run's `temporal_statistics.json` and
  `features/temporal_statistics.npz`: full selected-trial arrays and branch measurements.
- `runs/.../result.json` and histories: selected EMA policy, calibration provenance,
  exact settings, metrics and split identifiers.

## Verification

Two independent agents reviewed the feature branch, saved error evidence,
EMA/calibration isolation and reporting. Static syntax/configuration checks
and documentation/shell checks were performed. New development tests cover
old checkpoints and initialization, power sensitivity and bin coverage,
EMA state and RNG, partial-batch calibration, saved checkpoint replay with
the new branch, class metrics and diagnostic sample alignment.

**No project code, tests, training, inference or checkpoint diagnostics ran
locally.** The later cluster report completed all 15 fits with no skipped
diagnostics; saved/replayed validation probabilities agreed within
`5.37e-7`. The branch was active and finite, but the combined recipe's accuracy
regressed. Existing source changes and research artifacts were preserved.
