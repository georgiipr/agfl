# EEGNet + AGFL improvement study: BCI Competition IV-2a

The target is at least **75% mean held-out accuracy across all nine subjects**
on the four-class task. This implementation is an experiment to measure, not
a claim that the target has been reached. Train each subject independently.
The model remains EEGNet with AGFL; neither another backbone nor another
dataset is introduced.

## What the downloaded results show

The 45 downloaded runs (nine subjects × five seeds) average **46.76% test
accuracy**. Final training accuracy is about **97.89%**, while final validation
accuracy is about **46.19%**. Validation accuracy at the selected checkpoints
averages **56.79%**. Those gaps are consistent with overfitting and unstable
checkpoint selection; they do not identify one proven cause.

The worst subject means are A06 **28.37%**, A05 **33.46%** and A02 **36.30%**.
A03 reaches **69.63%** and A09 **66.25%**. Improving only the strongest subjects
cannot establish a strong nine-subject result. Chance accuracy is 25% for
this four-class task. Artifact exclusion and the current split leave roughly
133–165 real training trials per subject/seed.

The downloaded architecture treats electrodes as attention tokens and flattens
the remaining temporal bins into a 224-dimensional electrode feature. It has
91,892 parameters. This differs from EEGNet's usual early spatial convolution
followed by a compact temporal feature map. Its pooling factors of 8 and 16
also leave only seven temporal bins from a 1,000-sample window.

The manuscript's 76.63% number and archived approximately 70–78% validation
curves are historical reference points. The archived trainer selected and
reported on the same validation partition, and preprocessing/model settings
also differed. They do not establish a 75% independent test baseline for the
current protocol.

## Changes to test

The `eegnet-bci2a` preset is explicit so saved electrode-attention models can
still be reconstructed with their original behavior.

| Component | Downloaded setting | Compact candidate |
|---|---|---|
| Spatial filtering | Per-electrode extraction followed by spatial readout | Depthwise convolution across all 22 electrodes before temporal pooling |
| Temporal kernel | 32 samples | 125 samples, approximately half a second at 250 Hz |
| Feature widths | F1=16, D=2, F2=32 | F1=8, D=2, F2=16 |
| Pooling | 8 then 16 | 4 then 8 |
| AGFL input | 22 electrode tokens, 224 features each | 31 time tokens, 16 features each |
| Attention integration | Residual | Residual with 0.1 dropout on the AGFL branch |
| Loss | Weighted focal, gamma 3 | Unweighted cross-entropy |
| Optimizer | AdamW, LR 0.005, weight decay 0.001 | AdamW, LR 0.001, weight decay 0.0001 |
| Batch size | 64 | 32 |
| Checkpoint | Maximum validation accuracy | Minimum validation cross-entropy |
| Stopping | 250 epochs | At most 250; patience 60; minimum 100 epochs |
| Normalization layers | PyTorch defaults | BatchNorm epsilon 0.001, new-observation weight 0.01 |
| Initialization | PyTorch defaults | Xavier for EEGNet convolutions and classifier |

The compact feature layout follows the authors' EEGNet design: temporal
convolution, depthwise spatial convolution, separable temporal convolution,
and pooling by 4 and 8. The authors recommend ordinary dropout for oscillatory
motor-imagery signals. The half-sampling-rate temporal kernel is a starting
choice, not an empirically established optimum for this study.
Sources: [official EEGNet implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py),
[EEGNet paper](https://arxiv.org/abs/1611.08024).

The BatchNorm setting uses PyTorch's convention, where `momentum=0.01` gives
the new batch a weight of 0.01; this corresponds to a 0.99 old-statistic weight.
See [PyTorch BatchNorm2d documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html).
These changes preserve AGFL's graph construction, polynomial, hop order,
coefficient initialization, head count and Top-k policy. The residual path
allows EEGNet's feature extractor to receive gradients when AGFL taps start
at zero. New weights must be trained for the changed architecture.

## Five bounded candidates

| Candidate | Purpose |
|---|---|
| `spatial_control` | Restore early spatial filtering/time attention with the prior width, pooling, focal loss, learning rate and epoch budget; retain training-channel normalization |
| `compact_train_channel` | Compact architecture and cross-entropy training; retain training-channel normalization |
| `compact_trialnorm` | Same compact model; standardize each channel within each individual trial |
| `compact_recombine` | Add training-only segment recombination to `compact_trialnorm` |
| `compact_recombine_slow` | Same augmentation with learning rate 0.0005 |

The control isolates the spatial/time layout change from the compact training
recipe. The compact recipe changes several settings together and is not a
factorial ablation attributing gains to each of them.

Recombination replaces a training example with probability 0.5. It divides
the four-second trial into eight segments. Each segment is drawn from another
training trial of the **same subject and class**, at the same relative time,
with every channel coming from the same donor. Original trials remain in the
training pool. Validation and test examples are never donors or augmented.
This creates new training inputs; it does not create more independent
recordings. The idea is adapted from the primary authors' segmentation and
reconstruction implementation in
[EEG-Conformer](https://github.com/eeyhsong/EEG-Conformer/blob/main/conformer.py).
Only that augmentation idea is used; the backbone here remains EEGNet.

## Preprocessing and evaluation

All candidates use the existing loader with 22 EEG channels, four classes,
250 Hz sampling, a 1,000-sample window starting at the imagery cue, 2–30 Hz
trial-local filtering and exclusion of flagged artifacts. These choices keep
trial membership fixed against the downloaded study. The acquisition/task
description is in the [official BCI IV-2a documentation](https://www.bbci.de/competition/iv/desc_2a.pdf).
Trial-local filtering avoids crossing boundaries between randomly partitioned
trials. It differs from archived continuous-recording filtering and is not
claimed to be the only possible valid preprocessing protocol.

Training-channel normalization fits its statistics on the training partition
only. Per-trial normalization uses each channel's own time samples within that
trial. It needs no statistics from another held-out trial. It can also remove
informative amplitude differences, so both alternatives are included.

Each subject/seed gets the same stratified 60/20/20 T-session partition for
every candidate. The search explicitly verifies matching trial IDs and class
counts. Normalization changes the data fingerprint and split ID, but must not
change trial assignments. Augmentation operates after partitioning.

For **each subject and seed separately**, select checkpoints by the candidate's
declared validation criterion, then choose the candidate with highest
validation accuracy. Macro F1 breaks ties, followed by declared candidate
order. Focal and cross-entropy loss values are not compared across candidates.
Validation metrics from other seeds never influence that split's choice:
a validation trial in one random split can be a test trial in another.

The candidate stage never constructs a test loader or evaluates test metrics.
After every choice is saved, evaluate only the selected checkpoints. The
result measures a validation-selected tuning procedure, whose hyperparameters
may vary by subject and seed. It is not a result for one fixed configuration.
`run --preset eegnet-bci2a` is available to measure one fixed compact setting.

The target statistic is the equal-weight mean of the nine subject means,
including all requested seeds. `search_result.json` marks the nine-subject
target only when all nine are present. A subset is a pilot; a two-epoch run
only checks execution. Seed splits overlap, so seed repetition is not five
independent new cohorts. The earlier test scores have already been inspected;
these improvements are exploratory relative to that study. A later locked
T-to-E evaluation with official E-session labels can provide a separate
confirmation, but is a different evaluation protocol.

## Running, resuming and reading the output

From `AGFL` on the experiment machine:

```bash
python main.py tune-eegnet --data-dir ../ml --output-dir results/eegnet-bci2a-search
```

Defaults: five candidates × nine subjects × five seeds, at most 250 epochs
per candidate. Run directories retain one updating epoch bar in an interactive
terminal; redirected logs avoid per-epoch console output.

```text
results/eegnet-bci2a-search/
  search_plan.json                     immutable settings/scope for this directory
  candidates/<candidate>/A01/seed_0/    checkpoint, history, config, split, selection.json
  selection_report.json                validation scores and choices, saved before testing
  selected/eeg/subject_A01/seed_0/      selected checkpoint, predictions and result.json
  search_result.json                   overall/per-subject selected test metrics
  search_report.md                     readable overall and all-run table
```

Candidates have no `result.json`. A completed candidate is reused only if its
configuration/provenance and checkpoint hash match. An interrupted candidate
restarts from epoch 1 after clearing its generated files. Selected test runs
are reused only with matching configuration identity and checkpoint hashes.
A search lock prevents concurrent writers. `--restart` replaces completed
work too. Changing the search scope requires a new output directory, so
old selected subjects cannot silently enter a smaller rerun's report.

Plot commands are in the [README](../README.md#improve-eegnet--agfl-on-bci-iv-2a).
Use `analyze` on `selected/`, and `diagnose` on each selected seed directory.
Generic analysis groups by configuration; use `search_report.md` for the
overall score when selected configurations vary. No attention-comparison
plot can be inferred from this single-attention study.

## Verification status and acceptance

Only Python syntax, preset JSON and shell-command syntax are checked locally.
No local training, inference, project CLI or project tests are run. Target
tests cover feature gradients despite zero AGFL taps, checkpoint replay,
training-only donors, early stopping without test evaluation, same-split
selection, resumed candidate work and isolation of changed search scopes.

Run the target tests and two-epoch smoke command in the README first. For the
full run, inspect the nine subject means, selected candidate distribution,
learning curves and confusion matrices alongside the overall score. If the
mean remains below 75%, report it as below target. More epochs alone are not
supported by the present near-perfect training scores. A remaining gap
requires another evidence-based iteration on this EEGNet/BCI IV-2a setup;
none of these unmeasured changes justifies promising 75% now.
