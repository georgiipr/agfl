# A01 augmentation comparison: results and next check

Reviewed `/Users/egor/Downloads/eegnet-A01-spatial-augmentation` against
`/Users/egor/Downloads/eegnet-A01-spatial-seeds` using saved files only.
No project training, inference or tests were executed locally.

## Result

The validation-selected procedure reaches **71.64% mean test accuracy**, with
a sample SD of **3.30 percentage points**, versus **68.00%** for the fixed
spatial control on the same five splits. The gain is **3.64 percentage points**.

| Seed | Selected setting | Control test | Selected test | Difference |
|---|---|---:|---:|---:|
| 0 | spatial_recombine | 63.64% | 67.27% | +3.64 pp |
| 1 | spatial_recombine | 72.73% | 74.55% | +1.82 pp |
| 2 | spatial_control | 69.09% | 69.09% | 0.00 pp |
| 3 | spatial_recombine | 67.27% | 74.55% | +7.27 pp |
| 4 | spatial_recombine | 67.27% | 72.73% | +5.45 pp |

Mean test macro F1 increases from **0.670 to 0.708**. Mean macro ROC-AUC
increases from **0.930 to 0.935**. The selected model gets 197 of 275 repeated
test predictions correct, versus 187 previously; these are not 275 unique
trials. The split assignments remain unchanged.

This is **not** the five-seed score of always enabling augmentation. Validation
chooses augmentation for four seeds and the control for seed 2. The augmented
seed-2 checkpoint has no test predictions, by design. Its unmeasured test
accuracy must not be inferred from the other seeds.

## What validation actually selected

| Seed | Control validation accuracy | Augmented validation accuracy | Selection reason |
|---|---:|---:|---|
| 0 | 83.64% | 90.91% | Higher accuracy |
| 1 | 81.82% | 85.45% | Higher accuracy |
| 2 | 83.64% | 81.82% | Control has higher accuracy |
| 3 | 70.91% | 74.55% | Higher accuracy |
| 4 | 81.82% | 81.82% | Augmented macro F1 0.817784 versus 0.816686 |

Augmentation strictly improves validation accuracy on three seeds, ties on
one and loses on one. The four selected augmented runs improve their test
scores over their matching controls. This supports retaining augmentation
as a validation-selected option, not assuming it always helps.

## Remaining limitations

Mean selected validation accuracy is **83.27%**, versus **71.64%** test
accuracy. The validation/test gap remains **11.64 percentage points**.
Seed 0 has a particularly large gap: 90.91% validation versus 67.27% test.
Validation is reused for epoch and candidate selection, and its peak is not
an independent estimate of test performance.

Mean class recall across the five test partitions changes as follows:

| Class | Control | Selected procedure |
|---|---:|---:|
| Left hand | 55.71% | 72.86% |
| Right hand | 87.14% | 81.43% |
| Feet | 48.57% | 51.43% |
| Tongue | 81.54% | 81.54% |

Most of the gain is in left-hand recognition. Feet remain the weakest class,
with frequent confusion with tongue. Some individual predictions regress:
across repeated seed evaluations, 25 previous errors become correct and 15
previously correct predictions become wrong. Net improvement does not imply
every trial improved.

The mean remains **3.36 percentage points below 75%**, and no selected seed
reaches 75% (the best is 74.55%). The five splits overlap and cover one subject;
their SD is not a confidence interval or a nine-subject result. Repeatedly
inspected test scores remain exploratory during development.

## Next check: keep the procedure fixed on A03 and A06

Keep the same two candidates and all model/preprocessing/training settings.
Run A03 and A06 separately, with five seeds each. These subjects provide
contrasting cases from the earlier study; this is a targeted pilot, not a
representative estimate of the full cohort. It checks whether A01-specific
development produced useful gains on other subjects before expanding further.
No new code or candidate is required.

From `AGFL` on the cluster, with `A03T.gdf` and `A06T.gdf` in `../ml`:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 3 6 \
  --seeds 0 1 2 3 4 \
  --candidate spatial_control \
  --candidate spatial_recombine \
  --output-dir results/eegnet-A03-A06-spatial-augmentation
```

This is **20 candidate fits and 10 selected test evaluations**. The model is
trained separately for each subject. All candidate and checkpoint choices
are made on validation within that subject/seed split. The new output folder
keeps this study separate from A01; an identical relaunch resumes completed work.

Inspect each subject's five scores and selected settings. If improvement
extends to both, the next cohort check can use the same frozen procedure on
the remaining subjects. If a subject remains weak, its histories and errors
will guide the next model change. The goal remains four-class accuracy across
all nine subjects, including the difficult ones.

## After training: all plots for these two subjects

Run in Bash after training succeeds:

```bash
task_results_root=results/eegnet-A03-A06-spatial-augmentation
for task_subject in A03 A06; do
  for task_seed in 0 1 2 3 4; do
    python main.py diagnose \
      "$task_results_root/selected/eeg/subject_${task_subject}/seed_${task_seed}" \
      --output-dir "$task_results_root/analysis/diagnostics/${task_subject}/seed_${task_seed}" \
      --device cuda --partition validation --embedding tsne || break 2
  done
done
```

After all ten diagnostics finish successfully:

```bash
python main.py analyze \
  results/eegnet-A03-A06-spatial-augmentation/selected \
  --output-dir results/eegnet-A03-A06-spatial-augmentation/analysis \
  --plots \
  --diagnostics-root results/eegnet-A03-A06-spatial-augmentation/analysis/diagnostics
```

Read `search_report.md` for the selected procedure's per-subject and overall
scores. Generic `analyze` tables group by configuration when selected settings
differ; they are not a substitute for the search report's combined score.
The plot index is `analysis/figures/index.md`. Download the whole new folder,
including `candidates/`, `selected/` and `analysis/`, for review.

## Saved-artifact verification

- All ten candidates and five selected runs completed, with no failure markers.
- The paired configurations differ only in enabling eight-segment training
  recombination; recording data, model, loss, learning rate and splits match.
- The repeated controls reproduce their earlier selected validation metrics
  and checkpoint epochs for every seed.
- Candidate selection follows accuracy, then macro F1; selected checkpoint
  hashes and checkpoint epochs match their records.
- All ten validation/test metric records reproduce from saved predictions;
  prediction IDs match split manifests, and within-seed partitions are disjoint.
- All **36 result figures and 90 diagnostic figures** exist in PNG and PDF,
  with no missing, empty or skipped files in their manifests. Diagnostic
  manifests report matching checkpoint predictions for all five selected runs.

The checks establish consistency of the saved experiment artifacts. They
do not establish a 75% result, causal explanations of class errors, or
optimality of every preprocessing choice.
