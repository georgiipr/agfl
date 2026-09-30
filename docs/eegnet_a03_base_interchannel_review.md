Review of `eegnet-A03-base-agfl-attentions-v1`, downloaded 28 September 2026.

Base AGFL has the highest mean test accuracy in this A03 experiment: **69.26%**, compared with **66.30% for MHA**. This is a promising result for the requested electrode attention setting, but it does not establish general superiority. The full reports also expose a substantial regression in the shared EEGNet architecture and unstable Nyströmformer training.

Source: [downloaded report](../../report/analysis/report.md). This review uses saved files only; no AGFL CLI, training, model inference, checkpoint reconstruction, or project tests were run locally. Model and experiment code were not changed for this review.

**Accuracy comparison**

BCI Competition IV 2a, A03 only, four classes, five matched seeds/splits (0–4). Each fit has 162 training, 54 validation and 54 test trials. All 25 fits completed 250 epochs. Attention operates on 22 electrode tokens in every fit.

| Attention | Mean test accuracy ± sample SD | AGFL advantage | Seeds where AGFL wins |
|---|---:|---:|---:|
| Base AGFL | **69.26 ± 3.84%** | — | — |
| MHA | 66.30 ± 7.34% | +2.96 percentage points | 3/5 |
| Performer | 65.19 ± 2.03% | +4.07 percentage points | 5/5 |
| Linformer, rank 8 | 63.70 ± 3.84% | +5.56 percentage points | 4/5 |
| Nyströmformer, 8 landmarks | 54.07 ± 7.90% | +15.19 percentage points | 5/5 |

AGFL also has the highest mean macro F1 (0.6874) and ROC-AUC (0.8913). Its five test accuracies are 74.07%, 68.52%, 66.67%, 72.22%, and 64.81%. MHA wins seeds 0 and 4; the AGFL advantage is not universal.

The 270 test trial occurrences per attention contain only **177 distinct trials** because the repeated splits overlap. They are not 270 independent observations. In the supplied exploratory seed-level analysis, all Holm-adjusted accuracy comparisons have p-values above 0.05: AGFL versus MHA has adjusted paired t-test p = 0.612 and adjusted Wilcoxon p = 0.750. One subject does not support a population-level superiority claim.

**Why absolute accuracy remains low**

There are two distinct findings.

First, the current architecture overfits. AGFL's mean training accuracy at epoch 250 is **99.26%**, while mean validation accuracy at that epoch is **66.30%**. Its selected checkpoints occur at epochs **32, 90, 20, 28, and 51**. Their mean validation accuracy is 78.89%, and validation loss subsequently increases in every seed. The reported 69.26% test accuracy comes from those validation-selected checkpoints, not the final weights. Additional epochs alone are therefore poorly motivated by these histories. Training accuracy is the logged training-mode quantity, with dropout active, rather than a separate evaluation-mode measurement.

Second, the earlier saved MHA results in [report_2](../../report_2) averaged **85.56%**, versus **66.30%** now: a **19.26 percentage-point decrease**. For all five MHA seeds, the saved data settings, dataset fingerprint, training settings, split settings, and full split records match. The old model used temporal attention; the new model preserves electrode tokens with a different encoder/projection/readout path. The current base MHA has neither the output gate nor temporal bias enabled.

This comparison makes the shared architecture transition a strong investigation priority. It does not isolate attention axis from the simultaneous feature compression and spatial processing changes, and it does not establish that inter-electrode attention inherently performs poorly. The requirement to preserve electrode identities remains valid. Changing AGFL alone would leave this shared regression unexplained.

Class recall also identifies a remaining weakness:

| Class | Base AGFL | MHA |
|---|---:|---:|
| Left hand | 64.29% | 60.00% |
| Right hand | 78.57% | 68.57% |
| Feet | 60.00% | 66.15% |
| Tongue | 73.85% | 70.77% |

AGFL's mean advantage comes with worse feet recall. These rates average the same five seed evaluations; their supports include repeated trials.

**AGFL and baseline diagnostics**

AGFL has learned nonzero coefficients in all 20 saved heads across the five checkpoints. Effective signed coefficients range from −0.3212 to +0.2674. Effective temperatures range from 0.7668 to 1.2656, with no head at a temperature clamp. These observations rule out all-zero saved filters or temperature clamp saturation as explanations for these results. They do not by themselves prove that each learned hop improves prediction.

Nyströmformer needs separate attention before using its large deficit as evidence for the paper. Its training losses spike to approximately 4.27–9.88 across seeds; mean final training accuracy is 31.23% and mean final validation accuracy is 28.15%. The four-class chance reference is 25%. Its reported test scores still use the selected checkpoints, but the trajectories show a struggling baseline under this training recipe. Saved histories cannot establish the cause of the instability.

Nyströmformer seed 1 also records a diagnostic probability consistency failure: maximum absolute difference **0.00004810**, with comparison tolerances `atol=0.00001` and `rtol=0.0001`. The recorded training and diagnostic source hashes match. This is a small numerical replay discrepancy; the files do not establish whether any class decisions changed. The other 24 replay checks pass. Independently recomputed metrics from the original saved predictions agree with all 25 reported results, including this run.

AGFL uses **12,788 total model parameters**, compared with **15,172 for MHA**, a 15.71% reduction in this configuration. Recorded mean training elapsed time is about **48.53 seconds for AGFL versus 29.95 seconds for MHA** (1.62×). Thus these runs support a parameter-count advantage, not a measured speed advantage. This is elapsed training time, not a controlled inference benchmark.

**Plots and completeness**

All **678 figures** listed in the downloaded manifests are present in both PNG and PDF: **198 result figures** and **480 checkpoint diagnostic figures**. The figure headers were checked, and representative training and electrode plots were visually inspected. The one manifest warning is the Nyströmformer prediction consistency flag described above; there are no missing listed figure files.

- [All result plots](../../report/analysis/figures/index.md): comparisons, training curves, confusion matrices and other saved-result figures.
- [AGFL seed 0 diagnostic index](../../report/diagnostics/eeg/subject_A03/eegnet-agfl-2ab4871977585a0729e4/seed_0/index.md): signal views, spectra, CSP, learned coefficients, temperatures, PCA embedding, electrode maps, and head statistics.
- [AGFL electrode heatmaps](../../report/diagnostics/eeg/subject_A03/eegnet-agfl-2ab4871977585a0729e4/seed_0/mixers/attn_blocks.0_heatmaps.png).
- [AGFL scalp connection view](../../report/diagnostics/eeg/subject_A03/eegnet-agfl-2ab4871977585a0729e4/seed_0/mixers/attn_blocks.0_electrode_connections.png).
- [AGFL seed 0 training curve](../../report/analysis/figures/runs/2ab4871977585a0729e4/seed_0/training.png).
- [Nyströmformer seed 0 training curve](../../report/analysis/figures/runs/78410e859a38adba6a06/seed_0/training.png).

The saved mixing matrices have shape **4 heads × 22 electrodes × 22 electrodes** and named electrode axes. Class maps and CSV edges use the same electrode ordering; CSV values agree with the saved matrices. Rows receive information from columns. AGFL heatmaps show the routing matrix before polynomial filtering, not the full filtered operator. They describe model routing, not demonstrated physiological or causal brain connectivity.

**What to do next**

Keep this base AGFL configuration as the reference. The next controlled comparison should examine the shared encoder and spatial readout while retaining 22 electrode tokens, and apply each proposed shared change equally to base AGFL and MHA. Select changes using validation data. Stabilize and recheck Nyströmformer before making broad claims against all attention families. The repeated A03 test evaluations are exploratory; eventual article claims need a fixed protocol evaluated across subjects.

The evidence supports investigating representation and generalization rather than merely extending training or assuming a new AGFL mechanism will restore the previous accuracy. It does not justify promising 90%.

**Verification record**

Saved validation and test probabilities were checked for finite values and normalization. Accuracy, confusion matrices, macro F1, and one-versus-rest macro ROC-AUC were independently recomputed for all **50 saved partitions** and agree with the reported values. Seed-level aggregates also agree. Splits are disjoint within each run and match across attentions for each seed. Checkpoint epochs match the first maximum of validation accuracy in the recorded histories. Diagnostic examples belong to the validation partition, and saved CSP training IDs are contained in the training partition. The downloaded report contains no checkpoints, so the diagnostic reconstruction checks cited above are the saved cluster checks.

Training source SHA-256: `97def5642d5867b3317e0fa815be756b55b6dd7f2dee23e10f340dca7d157646`.

Dataset fingerprint: `ce4f9758ab17f5ea5cdb8bd730d4daec20e4870cfda03ef7b110bf4df20f5f66`.
