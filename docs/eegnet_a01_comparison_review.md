# Review of the A01 EEGNet comparison

Reviewed saved artifacts in `/Users/egor/Downloads/eegnet-A01-comparison`.
Scope: EEGNet + AGFL, BCI Competition IV-2a, four classes, subject A01,
seed 0. No project execution, training or inference was performed locally.

## Decision

Use **`spatial_control` as the working configuration** and repeat it on A01
across five seeds before making further architectural changes. The compact
recipe is not the strongest starting point in this comparison. The 75% test
accuracy target remains unmet.

| Candidate | Selected validation accuracy | Maximum validation accuracy anywhere | Selected epoch | Epochs trained |
|---|---:|---:|---:|---:|
| spatial_control | 83.64% | 83.64% | 76 | 250 |
| compact_train_channel | 61.82% | 63.64% | 74 | 134 |
| compact_recombine_slow | 58.18% | 58.18% | 112 | 172 |
| compact_recombine | 56.36% | 60.00% | 64 | 124 |
| compact_trialnorm | 52.73% | 56.36% | 74 | 134 |

The control selects checkpoints by validation accuracy; compact candidates
select by validation loss. Even their maximum accuracy across all recorded
epochs is below the control. Focal and cross-entropy loss values must not be
compared directly. The candidate bundles change several settings, so this
comparison cannot attribute the entire gap to learning rate, pooling, width,
BatchNorm, initialization or loss individually.

## What improved

The selected control obtains **35/55 correct test trials = 63.64%**, macro
F1 **0.632**, and macro one-versus-rest ROC-AUC **0.898**. The previous
electrode-attention A01/seed-0 model obtained **26/55 = 47.27%** on the same
trial IDs and the same recording bytes. That is nine additional correct
trials, or **16.36 percentage points**. The new model corrects 12 previously
wrong predictions and loses three previously correct ones.

The control uses early depthwise spatial filtering across all 22 electrodes,
followed by AGFL over seven temporal tokens with 32 features each. It has
5,652 parameters, compared with 91,892 in the earlier electrode-token model.
It retains F1=16, D=2, F2=32, kernel 32, pooling 8/16, dropout 0.5,
training-channel normalization, AdamW LR 0.005, weight decay 0.001,
weighted focal loss and PyTorch's original BatchNorm/initialization settings.
AGFL mathematical settings are unchanged.

The control's validation advantage is not confined to its single best epoch:
its final 25 epochs average **74.33%**, spanning **70.91–76.36%**. The original
argument that more temporal tokens or the entire compact training recipe
would necessarily improve this task is not supported by this run.

Within the compact recipe, training-channel normalization improves selected
validation accuracy from 52.73% to 61.82%. Segment recombination improves the
per-trial-normalized candidate modestly, but does not catch the control.
These are results for one subject/split, not general conclusions about either
normalization or augmentation. Augmentation has not yet been tested on the
winning control configuration.

## What remains unresolved

The selected validation score is **46/55 = 83.64%**, while the independent
test partition gives 35/55. The **20 percentage point gap** remains material.
Validation was reused to select among epochs and candidates, so its best
score is an optimistic estimate of generalization. One small split cannot
separate selection effects, trial difficulty and model variability.

Test class recall:

| Class | Correct / total | Recall |
|---|---:|---:|
| Left hand | 10/14 | 71.43% |
| Right hand | 11/14 | 78.57% |
| Feet | 6/14 | 42.86% |
| Tongue | 8/13 | 61.54% |

Feet trials account for eight of the 20 test errors. Training class counts
are almost equal (41/41/40/41), so class imbalance is not an evident explanation
for that weakness. Reaching at least 75% on these 55 test trials would require
42 correct predictions: seven more than the present result. Test errors are
descriptive evidence; they should not be used to tune class thresholds or
select another checkpoint on this already inspected test set.

No candidate test results are available except for the selected winner.
Consequently, the other four candidates cannot be ranked by test accuracy.

## Verification performed

- All five candidates completed, with correct selected epochs according to
  their declared validation criterion.
- Candidate checkpoint hashes match the saved selection records; the selected
  checkpoint hash also matches its completed result.
- All candidates use identical train/validation/test trial IDs and class counts.
  There are 163 training, 55 validation and 55 test trials, with no overlapping IDs.
- Recalculated accuracy, macro F1 and macro ROC-AUC from saved predictions match
  both validation and test metrics.
- No failure markers are present. Candidate folders contain no test predictions
  or completed test result files, as intended.
- All nine result figures and 18 diagnostic figures exist as nonempty PNG/PDF
  files. The diagnostic manifest reports reproduced validation probabilities
  within a maximum absolute difference of approximately 2.38e-7.

These checks validate saved-artifact consistency. They do not replace numerical
project tests or establish that every preprocessing/design choice is optimal.

## Next cluster run: one subject, one configuration, five seeds

From `AGFL`, with `A01T.gdf` in `../ml`:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 1 \
  --seeds 0 1 2 3 4 \
  --candidate spatial_control \
  --output-dir results/eegnet-A01-spatial-seeds
```

This is **five fits total**. Seed 0 is repeated as a reproducibility check;
seeds 1–4 measure variation under the existing repeated-split protocol. The
configuration is fixed and the compact candidates are not retrained. A new
output folder is required because the search scope has changed. Repeating
this exact command after interruption reuses matching completed work.

Keep all five scores and report their mean and spread. These random splits
overlap, so this is a stability check rather than five independent confirmation
cohorts. Do not report the best seed as the subject's accuracy.

If the improvement repeats, retain this EEGNet configuration as the baseline
for subsequent changes. A small controlled comparison of augmentation on
this winning configuration would then answer an unanswered question. If the
gap to 75% remains, inspect the persistent class errors and vary one factor
at a time with validation-only selection. The present evidence does not
justify replacing the winning recipe with another broad compact-model search.

## Plots for the five-seed follow-up

After successful training, generate diagnostics in Bash:

```bash
for task_seed in 0 1 2 3 4; do
  python main.py diagnose \
    "results/eegnet-A01-spatial-seeds/selected/eeg/subject_A01/seed_${task_seed}" \
    --output-dir "results/eegnet-A01-spatial-seeds/analysis/diagnostics/seed_${task_seed}" \
    --device cuda --partition validation --embedding tsne || break
done
```

After all five diagnostics finish successfully:

```bash
python main.py analyze \
  results/eegnet-A01-spatial-seeds/selected/eeg/subject_A01 \
  --output-dir results/eegnet-A01-spatial-seeds/analysis \
  --plots \
  --diagnostics-root results/eegnet-A01-spatial-seeds/analysis/diagnostics
```

Read `results/eegnet-A01-spatial-seeds/search_report.md` for all five scores
and their summary. Plot indexes are under `analysis/figures/index.md` and
`analysis/diagnostics/seed_N/index.md`.
