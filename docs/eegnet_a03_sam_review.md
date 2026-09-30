# EEGNet/A03 SAM results

Reviewed 24 September 2026. Download: `/Users/egor/Downloads/report`.
Cluster session: `results/eegnet-A03-token-sam-v1`.

Subsequent action: SAM was reverted at the user's request. Its optimizer,
preset and reporting additions are archived as an ordinary
[reproduction patch](archive/eegnet_sam_reproduction.patch), outside the
active package. Historical `sam_rho` settings below describe the saved
experiment; the restored configuration no longer accepts that key.

**SAM did not improve AGFL. Token AGFL fell from 85.19% to 81.11%, a
loss of 4.07 percentage points. Retain token AGFL with ordinary AdamW
(`training.sam_rho=0.0`).** Validation accuracy also fell, from 91.48%
to 89.63%. The ordinary control reproduced the previous per-seed scores
and selected epochs, so the earlier 85.19% result remains available.

All 20 fits completed 250 epochs. All 469 PNG/PDF figure pairs are present;
no plotting recovery or retraining is needed to review this session.

## Five-seed results

| Configuration | Test accuracy mean ± seed SD | Validation mean | Test macro F1 | Test macro ROC-AUC |
|---|---:|---:|---:|---:|
| Token AGFL, ordinary AdamW | **85.19% ± 5.71 pp** | **91.48%** | **0.8496** | **0.9762** |
| Token AGFL, SAM | 81.11% ± 5.14 pp | 89.63% | 0.8029 | 0.9546 |
| Linformer rank 4, SAM | 81.11% ± 3.80 pp | 89.26% | 0.8068 | 0.9514 |
| MHA, SAM | 80.00% ± 4.42 pp | 90.74% | 0.7869 | 0.9558 |

Under the matched SAM recipe, AGFL is **1.11 pp above MHA and tied with
Linformer**. This does not establish improvement over the previous recipe
or superiority to all attentions. The earlier ordinary-AdamW comparison
gave MHA 85.56% and Linformer 85.93%; those baselines were not rerun without
SAM in this session. Comparing ordinary AGFL directly with SAM baselines
would mix training treatments and cannot isolate the attention effect.

All nine reported contrasts (three metrics × three controls) have five
matched seeds and no missing pairs. None survives the report's Holm
correction. Five overlapping splits of one development subject cannot
establish a general superiority claim.

## Per-seed outcomes

Entries are correct predictions out of 54 test trials. One occurrence
corresponds to 1.85 pp within a seed.

| Seed | Ordinary token AGFL | SAM token AGFL | SAM MHA | SAM Linformer | SAM − ordinary AGFL |
|---|---:|---:|---:|---:|---:|
| 0 | 49 | 45 | 45 | 47 | −4 |
| 1 | 46 | 47 | 46 | 42 | +1 |
| 2 | 41 | 40 | 40 | 44 | −1 |
| 3 | 46 | 42 | 43 | 42 | −4 |
| 4 | 48 | 45 | 42 | 44 | −3 |
| Repeated occurrences | 230 | 219 | 216 | 219 | −11 |

SAM wins one seed and loses four against ordinary token AGFL. Across
matched test occurrences it fixes 11 errors but loses 22 previously
correct predictions; one further changed prediction remains wrong.
There are 177 unique test trials across the five overlapping partitions,
not 270 independent trials.

Ordinary AGFL's selected epochs are `[153,104,160,180,130]`, with validation
correct counts `[51,50,48,49,49]`. SAM AGFL selects `[87,120,144,94,104]`,
with validation counts `[50,51,47,45,49]`. SAM improves validation in one
seed, ties one and loses three. It fixes five validation errors but loses
ten previously correct validation predictions.

## Main regression: feet classification

| Class | Ordinary AGFL test recall | SAM AGFL test recall | Change | Net correct occurrences |
|---|---:|---:|---:|---:|
| Left hand | 82.86% | 80.00% | −2.86 pp | −2 |
| Right hand | 94.29% | 95.71% | +1.43 pp | +1 |
| Feet | **83.08%** | **66.15%** | **−16.92 pp** | **−11** |
| Tongue | 80.00% | 81.54% | +1.54 pp | +1 |

The other classes' net changes cancel, leaving the eleven lost feet
predictions as the aggregate loss. Feet→tongue confusions rise from 6 to
11, feet→right-hand from 2 to 6, and feet→left-hand from 3 to 5.
Validation feet recall also falls, from 90.77% to 83.08%.

SAM did not remove the training/validation gap. For seed 3, last-50-epoch
mean training accuracy is 96.26%, but mean validation accuracy is 80.26%;
the ordinary control records 96.19% and 88.89%. Training for all 250 epochs
did not recover that seed's validation performance. This shows that the
tested recipe failed, not that every SAM configuration would fail. The
saved report cannot establish that a smaller radius, a different loss, or
another architecture would achieve the target.

## Execution and artifact checks

This review read saved files only; no local project code, training,
checkpoint inference or development tests were executed.

- All 20 results are complete and have 250 history entries. Every selected
  epoch matches the first-best validation-accuracy rule and its saved
  validation metrics match the corresponding history entry.
- All 40 validation/test prediction sets independently reproduce accuracy,
  macro F1, one-vs-rest macro ROC-AUC and confusion matrices exactly.
  Probabilities are finite and normalized; prediction IDs match split IDs.
- Within each seed, train/validation/test IDs are disjoint, with 162/54/54
  trials. All four arms use matching per-seed splits, data, backbone and
  numerical package versions. The two AGFL configurations differ in
  training only by `sam_rho`; their attention options match.
- All runs share source hash
  `d5b86ce8bbfb1e3ec0534c03276927a9aa9c4ae7e62a33fc9d06735dbe70129d`.
  Full package inventories are not identical: SAM arms report extra
  packaging utilities and a different `platformdirs` entry. Recorded
  Torch/NumPy/SciPy/scikit-learn/MNE/WFDB versions, Python, CUDA, cuDNN and
  GPU model match. The files do not explain the inventory difference;
  this should not be described as an identical full environment.
- Every SAM history records rho 0.05 and three optimizer updates per epoch,
  totaling 750. Policies record replayed randomness and first-pass-only
  buffers. No skipped AMP steps are recorded; AMP is disabled.
- All 20 cluster diagnostic manifests report successful prediction replay,
  with maximum probability discrepancy `2.09e-7`, and matching training
  source. These are saved cluster checks, not new local inference.
- Token routing remains active: SAM gate norms are nonzero (0.55–1.07 per
  head). Mean correction saturation is about 0.049%; maximum zero-sum
  residual is `9.69e-8`. An inactive or saturated router is not supported
  as an explanation by these diagnostics.

These checks support internal consistency of the saved results; they do
not constitute a numerical proof of the two-pass optimizer implementation.
AGFL's recorded training/evaluation time totals 320.1 seconds with SAM
versus 198.7 seconds for the ordinary control, about 1.61×, excluding later
diagnostics. This is a session observation, not a throughput benchmark.

## Plots and files

There are **159 result figure pairs and 310 checkpoint diagnostic figure
pairs**, totaling 469 PNGs and 469 PDFs. Every manifest reference exists and
is nonempty; no figures are marked skipped. The paired-accuracy and seed-3
SAM training figures were also visually inspected.

- [Generated tables and report](/Users/egor/Downloads/report/analysis/report.md).
- [All result plots](/Users/egor/Downloads/report/analysis/figures/index.md).
- [SAM versus ordinary AGFL by seed](/Users/egor/Downloads/report/analysis/figures/paired/dc59f4763bb25f519fce_bfb47eb1907c65099022_accuracy.png).
- [SAM AGFL seed-3 training curves](/Users/egor/Downloads/report/analysis/figures/runs/dc59f4763bb25f519fce/seed_3/training.png).
- [SAM AGFL seed-0 checkpoint diagnostics](/Users/egor/Downloads/report/diagnostics/eeg/subject_A03/eegnet-agfl-dc59f4763bb25f519fce/seed_0/index.md).

## Decision

Keep the original EEGNet and token-routing AGFL with ordinary AdamW as the
retained configuration. SAM at rho 0.05 failed the requested improvement
target and remains an archived ablation. No training or attention code was
changed during the results review itself; the subsequent requested rollback
is recorded above. Saved experiment files remain unchanged.
