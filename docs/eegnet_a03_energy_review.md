# EEGNet/A03: energy-routing results

Reviewed 25 September 2026 from `/Users/egor/Downloads/report`.
Cluster session: `results/eegnet-A03-token-energy-v1`.
Subsequent user-requested action: energy routing was rolled back; see the
[restoration record](eegnet_token_restore.md). Results and plots remain intact.
This review uses saved artifacts only; no project imports, training,
checkpoint inference or development tests ran locally.

**The added energy inputs did not improve AGFL in this study. Mean test
accuracy fell from 85.19% to 84.81% (−0.37 percentage points), and validation
fell from 91.48% to 90.74% (−0.74 pp). Retain original token-routing AGFL
with ordinary AdamW as the reference configuration.** This small difference
does not establish that energy routing is generally worse; it does fail to
demonstrate the requested improvement.

All 20 fits completed 250 epochs, and all **511 PNG/PDF figure pairs** are
present. No plot recovery or additional training is needed to review this
session. No model, attention, optimizer or preset was changed during review.

## Five-seed comparison

All arms use original seven-token EEGNet and subject A03T individually.

| Attention | Test accuracy mean ± seed SD | Validation mean | Test macro F1 | Test macro ROC-AUC | Parameters |
|---|---:|---:|---:|---:|---:|
| Original token AGFL | **85.19% ± 5.71 pp** | **91.48%** | **0.8496** | **0.9762** | 8,244 |
| Token AGFL with energy inputs | 84.81% ± 8.22 pp | 90.74% | 0.8437 | 0.9721 | 8,268 |
| MHA | 85.56% ± 6.60 pp | 92.96% | 0.8525 | 0.9625 | 8,036 |
| Linformer rank 4 | 85.93% ± 9.13 pp | 90.74% | 0.8535 | 0.9652 | 8,260 |

The new AGFL is 0.74 pp below MHA and 1.11 pp below Linformer. The original
AGFL control reproduces its previous per-seed accuracies and selected epochs;
MHA and Linformer also reproduce their previous aggregate scores. Original
token AGFL has the highest observed ROC-AUC here, but that does not establish
accuracy superiority or a broad superiority claim across metrics.

All five declared contrasts have five matched seeds for each of the three
metrics: 15 comparisons, no missing pairs. None survives the report's Holm
correction. Energy-versus-original accuracy has paired t-test p=0.854 and
Wilcoxon p=1.0; both adjusted p-values are 1.0. These are exploratory seed
comparisons on overlapping splits of one repeatedly examined development
subject, not five independent subjects or evidence of equivalence.

## What changed in predictions

Entries are correct predictions out of 54 test trials per seed.

| Seed | Original token AGFL | Energy AGFL | MHA | Linformer | Energy − original |
|---|---:|---:|---:|---:|---:|
| 0 | 49 | 48 | 49 | 51 | −1 |
| 1 | 46 | 49 | 45 | 50 | +3 |
| 2 | 41 | 38 | 41 | 48 | −3 |
| 3 | 46 | 47 | 50 | 39 | +1 |
| 4 | 48 | 47 | 46 | 44 | −1 |
| Repeated occurrences | 230 | 229 | 231 | 232 | −1 |

The new candidate wins two seeds and loses three against original AGFL.
It fixes nine test errors but loses ten previously correct predictions; two
additional changed predictions remain wrong. There are 177 unique test trials
across the five partitions, not 270 independent observations. The −0.37 pp
aggregate is one net lost correct occurrence across those repeated splits.

Validation correct counts change from `[51,50,48,49,49]` to
`[50,50,47,49,49]`: two losses and three ties, no winning seed. Selected
epochs change from `[153,104,160,180,130]` to `[109,115,57,180,172]`.
The first-best validation-accuracy checkpoint rule was followed in every run.
Neither test outcomes nor these retrospective comparisons justify selecting
different epochs for the completed study.

## Class tradeoffs

| Class | Original AGFL test recall | Energy AGFL test recall | Change | Net correct occurrences |
|---|---:|---:|---:|---:|
| Left hand | 82.86% | 84.29% | +1.43 pp | +1 |
| Right hand | 94.29% | 94.29% | 0.00 pp | 0 |
| Feet | **83.08%** | **80.00%** | **−3.08 pp** | **−2** |
| Tongue | 80.00% | 80.00% | 0.00 pp | 0 |

The largest seed regression is seed 2: test accuracy falls from 75.93% to
70.37%, and all three net lost correct predictions are feet trials. Feet
correct counts fall from 9/13 to 6/13, with feet→tongue errors rising from
three to five and feet→right-hand from one to two.

Validation also shows a tradeoff: feet recall falls from 90.77% to 87.69%
while tongue rises from 90.77% to 93.85%. Left- and right-hand recall each
lose 1.43 pp. These observations locate the changes; they do not identify
a proven physiological or optimization cause.

## The new branch was active

All five energy runs have schema-4 coefficient diagnostics with four heads,
seven tokens and three hops. The saved energy arrays contain 54 validation
trials per seed. All 20 head-specific energy gates have nonzero learned
weights; their norms range from **0.0879 to 0.6560**.

The two added inputs vary across the saved trials and tokens. At fixed saved
values and graphs, removing only the added `B*r` logits changes coefficients
by mean absolute **0.00946**, with maximum **0.10708** across these diagnostics.
Thus the new computation was used. These numbers describe coefficient
changes, not prediction differences or a separately retrained ablation.

The zero-sum correction remains satisfied within **9.64e-8**. Only about
**0.265%** of coefficient entries lie above 95% of the attainable correction
bound. Missing activation or pervasive correction saturation is not supported
as the explanation by these saved diagnostics. The files cannot establish
that stronger routing, rescaling or another optimizer would improve accuracy.

## Artifact and comparison checks

- All 20 results are complete, with 250 history entries each. Every saved
  selected epoch is the first maximum of validation accuracy, and its
  validation metrics equal the corresponding history entry. No skipped AMP
  updates are recorded; AMP was disabled.
- All 40 saved prediction sets independently reproduce accuracy, macro F1,
  macro one-vs-rest ROC-AUC and confusion matrices, to numerical precision.
  Probabilities are finite and normalized; prediction IDs match their splits.
- Within each seed, train/validation/test IDs are disjoint, with 162/54/54
  trials. All four arms share the full split, dataset fingerprint, data
  settings, backbone and ordinary-AdamW training configuration. Between the
  two AGFL arms, attention options differ only in `coefficient_conditioning`.
- All runs share source hash
  `3e5b06f48c9a65905b311c21cf0f59f0ff1f3e5860429842dbd48a0cf49f4979`.
  Python, numerical package versions, CUDA, cuDNN and GPU model match.
  The full environment inventory differs: later arms record extra packaging
  utilities and `platformdirs` 4.4.0 versus the control's 4.9.4. This repeats
  the inventory pattern in the SAM study. The artifacts do not establish
  its cause; the full environments must not be described as identical.
- All 20 cluster diagnostic manifests record successful prediction replay
  and matching training source. Maximum probability discrepancy is
  **2.09e-7**. These are saved cluster checks, not new local inference.
- The energy descriptor/logit/effect arrays have the expected shapes,
  finite values and bounds. Saved `B*r` logits match independent NumPy
  multiplication; coefficient effects equal actual minus saved
  without-energy coefficients.
- All manifest-referenced figures exist and are nonempty, with no skipped
  figures: **161 result pairs + 350 diagnostic pairs = 511 PNGs and 511 PDFs**.
  The paired-accuracy plot and seed-2 energy-effect plot were visually checked.

Recorded training/evaluation time totals 209.3 seconds for original token
AGFL and 236.6 seconds for the new candidate, about 13% more. This excludes
checkpoint diagnostics and is a session observation, not a controlled
throughput benchmark.

## Plots and decision

- [Generated analysis](/Users/egor/Downloads/report/analysis/report.md).
- [Result plot index](/Users/egor/Downloads/report/analysis/figures/index.md).
- [Energy versus original AGFL by seed](/Users/egor/Downloads/report/analysis/figures/paired/69d9118aee9e6e2c726e_53751b62e35411414d5f_accuracy.png).
- [Energy AGFL seed-2 diagnostics](/Users/egor/Downloads/report/diagnostics/eeg/subject_A03/eegnet-agfl-69d9118aee9e6e2c726e/seed_2/index.md).
- [Energy branch's coefficient effects, seed 2/head 0](/Users/egor/Downloads/report/diagnostics/eeg/subject_A03/eegnet-agfl-69d9118aee9e6e2c726e/seed_2/filters/attn_blocks.0_energy_effect_head_0.png).

Keep original token-routing AGFL as the reference and record this energy
extension as an unsuccessful ablation. The new inputs reached the router
and learned weights, but did not produce a validation or test improvement.
The evidence does not justify replacing the retained configuration, claiming
superiority over the baselines, or promising a gain from strengthening this
branch. No architectural change was made during the results review. The
subsequent requested rollback preserves the implementation/preset/tests in a
plain-text reproduction patch outside the active package.
