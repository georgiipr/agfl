# Saved-result analysis

`agfl.analysis` reads completed `result.json` files recursively. It never uses
README numbers or manually entered metrics. Config/result identities are checked;
duplicate experiment/seed records are rejected instead of counted as additional
replicates. Model and attention are separate columns. Different actual backbones,
source, training, dataset, or architecture settings
cannot silently form one controlled comparison.

Each experiment group reports Accuracy, macro one-vs-rest ROC-AUC, and macro F1
for validation and test, with mean, sample standard deviation (`ddof=1`), and the
number of defined observations. A single observation has undefined standard
deviation. Undefined AUC remains null and its available count is shown; it is
not converted to zero. Full resolved settings remain beside each original run.
Run-level metrics pool the held-out samples. Subject-specific test metrics are
also saved; the across-seed standard deviation is not a standard deviation
across subjects.

New BCI experiments have an explicit subject ID and a separate model per subject.
`per_subject.csv` reports the mean and sample SD across that subject's seeds.
`across_subjects.csv` first averages seeds within each person, then gives the
equal-weight mean and sample SD of those subject means. It lists the subjects
and completed seeds explicitly. Unequal seed coverage suppresses the overall
score. These descriptive summaries are not a pooled subject-by-seed significance
test; within-subject paired tests continue to require identical split IDs.

AGFL is paired with vanilla attention, Performer, Linformer, and Nyströmformer
only within the same chosen model, with all model options and head count fixed,
when the comparison protocol matches and the tuple
`(seed, split_id, dataset_fingerprint)` is identical. Missing partners are counted
explicitly. Experiments with no eligible counterpart are listed in the aggregate
JSON. Every chosen backbone has its own attention comparison family.
Earlier v1 generic runs are labeled Signal Transformer with their attention,
without changing stored configuration or identity hashes.

For paired differences `d = metric_AGFL - metric_baseline`, output includes:

- Number of usable pairs and mean paired difference.
- Two-sided paired t statistic and p-value.
- Two-sided Wilcoxon signed-rank statistic and p-value, excluding zero
  differences from the ranks (`zero_method="wilcox"`). Differences are rounded
  to twelve decimal places only for rank ties, not for the t-test or means.
- Cohen's paired effect size `d_z = mean(d) / sample_std(d)`.
- Signed rank-biserial effect `(positive_rank_sum - negative_rank_sum) /
  total_rank_sum`.
- Holm-adjusted p-values, with one family containing every supplied
  model/ablation/metric comparison, separately for the t-test and Wilcoxon.

At least two usable pairs are required for the statistical tests. Identical
paired values produce p=1 and rank effect zero by an explicit reporting
convention; d_z is undefined when variance is zero. A constant nonzero difference
has no finite t statistic or d_z and is reported as null with a reason; Wilcoxon
can still be computed. No hard-coded significance labels are emitted.

The paired t-test assumes suitable independent paired differences; Wilcoxon
assumes symmetric paired differences for its location interpretation. Five
seeds provide limited power, and random subject splits can overlap across seeds.
The resulting seed-level tests are exploratory summaries, not independent
subject-level population inference. More seeds do not remove dependence from
overlapping subjects. Predeclare comparisons and use held-out subject/fold units
for confirmatory analysis where appropriate; do not select the best ablation
using test outcomes and then treat its p-value as preplanned evidence.

Outputs are `aggregation.json`, `per_model.csv`, `per_subject.csv`,
`across_subjects.csv`, `attention_comparison.csv`,
`agfl_ablations.csv`, `statistical_comparisons.csv`, and `report.md`.
`aggregation.json` includes paths to every consumed source result. The CSVs
include experiment IDs so different AGFL configurations cannot be mistaken for
one aggregated model. Per-subject test metrics and raw probabilities are saved
with each run for further analysis.

Implementation references:
[paired t-test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html),
[Wilcoxon signed-rank test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html).
Numerical checks against SciPy are supplied in `tests/test_analysis.py` and have
not been run locally.
