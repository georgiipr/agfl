# EEGNet/A03 feature-routing results

Reviewed 24 September 2026. Download: `/Users/egor/Downloads/report`.
Cluster session: `results/eegnet-A03-feature-agfl-v1`.

**Feature-wise AGFL did not improve accuracy. Its mean is 84.07%, down
1.11 percentage points from token AGFL's 85.19%. Keep token AGFL as the
best observed AGFL test result; do not promote the heavier feature variant.**

The plotting fix succeeded: all 25 runs have completed checkpoint diagnostic
manifests and indices, and all 588 PNG/PDF figure pairs are present. No
retraining or plotting recovery is needed for this downloaded session.

## Complete five-seed comparison

| Attention | Test accuracy mean ± seed SD | Validation mean | Test macro F1 | Test macro ROC-AUC | Parameters |
|---|---:|---:|---:|---:|---:|
| Linformer rank 4 | 85.93% ± 9.13 pp | 90.74% | 0.8535 | 0.9652 | 8,260 |
| MHA | 85.56% ± 6.60 pp | 92.96% | 0.8525 | 0.9625 | 8,036 |
| AGFL, token routing | **85.19% ± 5.71 pp** | 91.48% | 0.8496 | **0.9762** | 8,244 |
| AGFL, static | 84.44% ± 5.94 pp | 92.22% | 0.8395 | 0.9721 | 8,052 |
| AGFL, feature routing | **84.07% ± 6.36 pp** | 90.74% | 0.8376 | 0.9681 | 9,588 |

Feature AGFL is 0.37 pp below static AGFL, 1.48 pp below MHA, and 1.85 pp
below Linformer. Token AGFL retains the highest observed macro ROC-AUC,
but that does not establish accuracy superiority or justify changing the
primary outcome after looking at results.

Validation also declined: feature AGFL ties token AGFL in three seeds and
loses in two, with no validation-accuracy win. This experiment therefore
does not support selecting feature routing even without inspecting test scores.

## Every seed

All entries below are correct predictions out of 54 test trials. One trial
is 1.85 percentage points within a seed.

| Seed | Static AGFL | Token AGFL | Feature AGFL | MHA | Linformer | Feature − token |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 48 | 49 | 47 | 49 | 51 | −2 |
| 1 | 49 | 46 | 48 | 45 | 50 | +2 |
| 2 | 41 | 41 | 40 | 41 | 48 | −1 |
| 3 | 46 | 46 | 48 | 50 | 39 | +2 |
| 4 | 44 | 48 | 44 | 46 | 44 | −4 |
| Total repeated occurrences | 228 | 230 | 227 | 231 | 232 | −3 |

Feature AGFL wins two seeds and loses three against token AGFL. At the
matched-trial level, it fixes 11 token-AGFL errors but loses 14 previously
correct predictions; two other changed predictions remain wrong. The four
retained control arms reproduce the per-seed test scores and selected epochs
recorded in the previous review. The earlier raw download is no longer at
its old path, so this is not a new byte-for-byte comparison with that download.

Selected feature-AGFL epochs: `[162,104,84,140,90]`. Validation correct counts
are `[51,50,47,49,48]`, versus token AGFL `[51,50,48,49,49]`. All 25 fits
record 250 epochs, and saved checkpoint epochs match the original first-best
validation-accuracy rule.

## Class tradeoffs

| Class | Token AGFL test recall | Feature AGFL test recall | Difference | Net correct occurrences |
|---|---:|---:|---:|---:|
| Left hand | 82.86% | 90.00% | +7.14 pp | +5 |
| Right hand | 94.29% | 91.43% | −2.86 pp | −2 |
| Feet | 83.08% | 78.46% | −4.62 pp | −3 |
| Tongue | 80.00% | 75.38% | −4.62 pp | −3 |

The extra capacity changed the class tradeoff without solving it. Left-hand
gains were outweighed by eight lost correct occurrences across the other
classes. On validation, feet recall also fell from 90.77% to 84.62%; that
part of the regression is visible independently of the test partition.

## Was it an execution problem?

The downloaded evidence does not indicate the earlier serialization failure,
an inactive router, or a checkpoint mismatch:

- All 25 results are completed; saved configurations match their result
  records. All methods use the original seven-token EEGNet and the same
  training, preprocessing, dataset fingerprint and per-seed 162/54/54 split.
  Partition IDs match the saved prediction IDs, with no within-seed overlap
  between training, validation and test.
- Accuracy, confusion matrices, macro F1 and macro one-vs-rest ROC-AUC were
  independently recalculated from all 50 saved validation/test prediction
  sets. They match the reported metrics. Probability arrays are finite and
  normalized.
- Every cluster diagnostic manifest records successful checkpoint prediction
  replay. The largest reported probability discrepancy is
  `2.98e-7`. Diagnostic inputs cover the 54 validation trials and CSP fitting
  IDs match the 162 training trials. No local checkpoint inference was run.
- All five feature routers have nonzero weights and variation across tokens
  and features. Gate norms range from 4.05 to 5.29. Mean within-token feature
  coefficient SD ranges from 0.089 to 0.116. These learned coordinates are
  not EEG electrodes or classes.
- Feature correction saturation averages 0.66% across seeds at the diagnostic
  threshold of 95% of the attainable bound. The largest zero-sum residual is
  `1.04e-7`, within the recorded tolerance; all coefficients are finite and
  within their configured correction bounds. Widening the correction bound
  has no clear support from these observations.

These checks establish that the implemented extension was active and the
reported outputs are internally consistent. They do not prove that every
implementation choice is optimal or identify a single causal reason for
the accuracy loss. More feature-specific parameters did not produce better
validation or test performance in this experiment.

## Plots now available

There are **208 result figure pairs and 380 checkpoint diagnostic figure
pairs**, totaling 588 PNG files and 588 corresponding PDFs. All manifest
file references exist and are nonempty; both analysis and diagnostic manifests
record no skipped figures. The class-recall and feature-coefficient plots
were also visually inspected.

- [Result tables and generated report](/Users/egor/Downloads/report/analysis/report.md).
- [All result plots](/Users/egor/Downloads/report/analysis/figures/index.md).
- [Class-recall comparison](/Users/egor/Downloads/report/analysis/figures/comparisons/fd0c2b32adf4f0aad648/test_class_recall.png).
- [New AGFL seed-0 checkpoint plot index](/Users/egor/Downloads/report/diagnostics/eeg/subject_A03/eegnet-agfl-624eb2f5203c3e2aea35/seed_0/index.md).
- Other checkpoints have their own `index.md` below `report/diagnostics/eeg/subject_A03/`.

Feature diagnostics save schema-3 coefficient JSON and NPZ arrays with
`[trial,head,token,feature,hop]` axes. Token diagnostics retain schema 2 and
their original axes. The previous float32 JSON serialization error did not
recur in these completed diagnostics.

## Interpretation and provenance

The immediate decision is to retain token AGFL and archive feature routing
as an unsuccessful ablation. The source and saved artifacts should remain
available for reproduction. No model or training changes were made during
this review.

The 270 repeated test occurrences cover 177 unique trials from A03, and
these splits have repeatedly informed development. Feature-versus-token
accuracy has exploratory paired-test p-values 0.634 (t-test) and 0.8125
(Wilcoxon), both Holm-adjusted to 1.0. That neither establishes a reliable
population-level regression nor supplies evidence of improvement or
equivalence. No broad claim that AGFL outperforms other attention methods
is supported by these accuracy results.

- Training source SHA256: `a72c18aa63d680f9944833d5afbd8fc7d7cfd863832cdf35700587b70222fac2`.
- Comparison ID: `fd0c2b32adf4f0aad648`.
- Dataset fingerprint: `ce4f9758ab17f5ea5cdb8bd730d4daec20e4870cfda03ef7b110bf4df20f5f66`.
- Recorded Python, CUDA/cuDNN, GPU model and core numerical package versions
  match across all runs. The broader package inventory is not byte-identical:
  some auxiliary packaging-library entries and the reported `platformdirs`
  version differ. Do not describe the entire recorded environment as identical.
- Review method: file inspection and standalone analysis of saved artifacts;
  no project imports, training, tests or checkpoint diagnostics executed locally.
