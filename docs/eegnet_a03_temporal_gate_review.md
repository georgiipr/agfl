# A03 temporal bias and output gate: completed results

26 September 2026. Source: the newly downloaded
`/Users/egor/Downloads/report`, session `eegnet-A03-temporal-gate-v1`.
This review independently checked saved predictions and diagnostics using
standalone NumPy analysis. No AGFL code, training, inference or project tests
ran locally. No model/configuration changes were made during this review.

## Outcome

**The combination reached 87.78%, up 2.59 percentage points from retained
token AGFL's reproduced 85.19%.** It wins four matched test seeds and ties
the fifth. Keep it as the leading candidate for further controlled evaluation.

| Arm | Validation accuracy | Test accuracy ± seed SD | Test correct occurrences | Parameters |
|---|---:|---:|---:|---:|
| Retained token AGFL | 91.48% | 85.19% ± 5.71 pp | 230/270 | 8,244 |
| Temporal bias | 91.85% | 85.93% ± 4.65 pp | 232/270 | 8,256 |
| Output gate | 90.74% | 85.56% ± 4.61 pp | 231/270 | 8,312 |
| Both | 91.85% | **87.78% ± 4.06 pp** | **237/270** | 8,324 |

The combination's macro F1 also improves, 0.8496 to 0.8745. ROC-AUC falls
slightly, 0.9762 to 0.9730; this is not improvement on every metric.
Its test interaction is +1.48 pp: combined gain exceeds the sum of the two
individual observed gains. This is descriptive, not proof of synergy.

| Seed | Control correct / 54 | Bias | Gate | Both | Both − control |
|---|---:|---:|---:|---:|---:|
| 0 | 49 | 47 | 50 | 50 | +1 |
| 1 | 46 | 48 | 47 | 48 | +2 |
| 2 | 41 | 42 | 44 | 44 | +3 |
| 3 | 46 | 48 | 44 | 47 | +1 |
| 4 | 48 | 47 | 46 | 48 | 0 |

Both fixes 17 previously wrong test predictions and loses 10 previously
correct ones, for seven net improvements. Four other changed predictions
remain wrong. The 270 occurrences cover only 177 unique test trials across
overlapping splits. To reach a 90% mean on these same partitions would
require 243/270, another six net correct occurrences (+2.22 pp).

The earlier recorded MHA and Linformer means were 85.56% and 85.93%.
The current combination is numerically higher by 2.22 and 1.85 pp,
respectively. Neither baseline was rerun in this four-AGFL-arm study;
these historical differences are not a new matched baseline comparison.

## Validation selection and uncertainty

Both and bias tie at 91.85185% mean validation accuracy. The declared
selected-checkpoint validation-loss tiebreaker selects both:
0.1331039324 versus 0.1331258357, a difference of only **0.0000219032**.
That is a very narrow tiebreak, not strong validation separation.
The control has lower mean validation loss (0.1262023091), but accuracy was
the prespecified primary selection criterion.

Combined validation correct counts are `[51,50,49,49,49]`, versus the
control's `[51,50,48,49,49]`: one winning seed, four ties, one net extra
correct occurrence. Its 10 fixed and 9 lost validation predictions show
that the small net change includes meaningful turnover.

No supplied comparison survives Holm correction. Both-minus-control test
accuracy has paired t p=0.0516, Holm p=0.7225; Wilcoxon p=0.125, Holm p=1.
The unadjusted F1 t-test is p=0.0324, but Holm p=0.4866. These tests concern
five dependent, repeatedly examined development splits, not independent
subjects. The result warrants retaining the candidate, not claiming confirmed
general superiority or choosing a new architecture using test labels.

## Class tradeoffs

| Class | Control test recall | Both test recall | Net correct occurrences |
|---|---:|---:|---:|
| Left hand | 82.86% | 88.57% | +4 |
| Right hand | 94.29% | 95.71% | +1 |
| Feet | 83.08% | **80.00%** | −2 |
| Tongue | 80.00% | 86.15% | +4 |

The improvement comes from hands and tongue; the feet limitation remains.
Bias alone and gate alone have still lower feet recalls (76.92% and 75.38%).
The combination partially avoids those regressions but does not eliminate
the class tradeoff. This does not justify fitting class-specific corrections
to these repeatedly inspected test examples.

## Evidence that the additions are active

In the combined model, 19 of 20 seed/head bias vectors penalize increasing
temporal distance and favor self edges. These are biases added to content
scores, not a statement that every resulting graph is local. Output gates
vary across trials and tokens: per-head standard deviations range from
0.184 to 0.680; all captured gate values lie between 0.0726 and 1.9454.
Neither branch remained at its neutral initialization.

Mean validation accuracy when temporarily disabling parts of each trained
combined checkpoint:

| Evaluation of trained combined checkpoint | Validation accuracy | Change |
|---|---:|---:|
| Original combined checkpoint | 91.85% | — |
| Bias disabled | 88.52% | −3.33 pp |
| Output gate disabled | 90.37% | −1.48 pp |
| Both disabled | 89.63% | −2.22 pp |

Removing bias loses nine net validation occurrences; removing gate loses
four; removing both loses six. These are saved cluster sensitivity checks
at jointly trained weights. They are not retrained controls and their
nonadditive effects do not establish a causal explanation of generalization.
The separately trained control remains the valid architectural comparison.

## Verification and figures

- All 20 fits completed 250 epochs; no failure files or skipped AMP steps.
  All selected epochs are the first maximum of validation accuracy. Control
  epochs `[153,104,160,180,130]` exactly reproduce the earlier control.
- Independently recomputed accuracy, macro F1, macro one-vs-rest ROC-AUC
  and confusion matrices from all 40 validation/test prediction arrays.
  Probabilities are finite and normalized; IDs and targets match splits.
- All four arms share each seed's complete, disjoint 162/54/54 split,
  dataset fingerprint, preprocessing, backbone and training settings. Their
  attention configurations differ only in the two experimental flags.
- All runs use source hash
  `5395266d7c92604334f51983faaf371b5eec88658d6f8451e22dead6e61dfba1`,
  matching the current local Python source. Python, numerical package
  versions, CUDA, cuDNN and GPU model match. Full environment inventories
  have two variants: packaging utilities and platformdirs 4.9.4/4.4.0 differ;
  they must not be described as entirely identical environments.
- All 20 diagnostic manifests report source agreement and successful
  cluster prediction replay; the largest discrepancy is 2.3842e-7.
- Gate/graph dimensions, bounds, row sums, sample IDs, temporal-bias
  equations and all saved validation-ablation accuracy/fixed/lost counts
  were independently checked from arrays. No model was loaded locally.
- Every manifest-referenced figure exists and is nonempty, with no skipped
  figures: **163 result pairs + 395 diagnostic pairs = 558 PNG/PDF pairs**.
  The four-arm accuracy plot and seed-2 combined bias/gate heatmap were
  visually inspected.

Useful entry points:

- [Generated study table](/Users/egor/Downloads/report/analysis/temporal_gate_study.md)
- [Four-arm accuracy plot](/Users/egor/Downloads/report/analysis/figures/temporal_gate/9b56b24f5cda7d6e89c1_accuracy.png)
- [All result plots](/Users/egor/Downloads/report/analysis/figures/index.md)
- [Combined seed-2 diagnostics](/Users/egor/Downloads/report/diagnostics/eeg/subject_A03/eegnet-agfl-e31369583cffbe0de724/seed_2/index.md)

The next evaluation should preserve the combined recipe and its control,
then compare against matched attention baselines. A generic temporal bias
or output gate may also help MHA, so that control matters for an AGFL-specific
claim. Independent confirmation requires untouched session/subject data
after freezing the recipe. No further architecture change or rollback was
performed as part of this review.
