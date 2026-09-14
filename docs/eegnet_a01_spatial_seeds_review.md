# A01 spatial EEGNet: five-seed review and next experiment

Source: `/Users/egor/Downloads/eegnet-A01-spatial-seeds`.
The saved artifacts were inspected without local project execution, training
or inference. The configuration remains EEGNet + AGFL on four-class BCI IV-2a,
with subject A01 trained separately.

## Results

**Mean test accuracy is 68.00%, with a sample SD of 3.30 percentage points.**
The corresponding original electrode-token model averaged 49.82% on the same
five trial splits: an improvement of **18.18 percentage points**. Each new
seed outperforms its matching original run.

| Seed | Original test accuracy | Spatial test accuracy | Spatial validation accuracy | Selected epoch |
|---|---:|---:|---:|---:|
| 0 | 47.27% | 63.64% | 83.64% | 76 |
| 1 | 47.27% | 72.73% | 81.82% | 193 |
| 2 | 54.55% | 69.09% | 83.64% | 134 |
| 3 | 47.27% | 67.27% | 70.91% | 32 |
| 4 | 52.73% | 67.27% | 81.82% | 200 |

Mean test macro F1 is **0.670** and mean macro ROC-AUC is **0.930**. AUC is a
ranking measure; it is not 93% classification accuracy. Mean selected
validation accuracy is **80.36%**, leaving a **12.36 percentage point** gap
to mean test accuracy. No seed reaches 75% test accuracy.

Seed 0's saved predictions reproduce the earlier spatial-control run exactly.
The improvement therefore is not confined to one fortunate seed, but these
are still overlapping splits from one subject. Their 275 test predictions
cover only **182 distinct trials**. The SD measures variation across these
runs; it is not a confidence interval or evidence about all nine subjects.
The nine-subject 75% target remains unverified and unmet by the available study.

## Remaining weaknesses

Mean class recall across the five test partitions:

| Class | Mean recall |
|---|---:|
| Left hand | 55.71% |
| Right hand | 87.14% |
| Feet | 48.57% |
| Tongue | 81.54% |

The recurring confusions are left hand versus right hand and feet versus
tongue. These indicate where errors remain; they do not prove a cause or
justify changing labels or tuning thresholds using test predictions.

Every seed has only 163 training trials, with training class counts
41/41/40/41. Class imbalance is not an evident explanation. All runs use 250
epochs, and selected checkpoints range from epoch 32 to 200. Simply extending
the epoch budget is not supported as the next isolated change.

## Next experiment: augmentation on the established spatial configuration

Keep the spatial architecture, train-channel normalization, focal loss,
learning rate, batch size, pooling, dropout, initialization and checkpoint
criterion fixed. Compare two configurations on A01 with seeds 0–4:

- `spatial_control`: the established model, with no segment recombination.
- `spatial_recombine`: the same model, with eight-segment recombination on
  training examples at probability 0.5.

Donors come only from the same subject/class training partition. All channels
of a segment come from the same donor and retain their relative trial time.
Validation and test examples are unchanged. This is a hypothesis about better
generalization from limited training trials, not a predicted 75% result.
Previous augmentation candidates used the compact model, so they did not
answer this question for the established spatial model.

The optional `spatial_recombine` candidate has been added to
`agfl/models/eegnet/search.py`. Default search commands still use the original
five candidates; request this new candidate explicitly. Update the cluster
checkout before launching it. Local verification covers syntax only; the
target-machine test checks that the two resolved configurations differ only
in whether segment recombination is enabled.

On the cluster, from `AGFL`:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 1 \
  --seeds 0 1 2 3 4 \
  --candidate spatial_control \
  --candidate spatial_recombine \
  --output-dir results/eegnet-A01-spatial-augmentation
```

This performs **10 validation-only fits**, then **five selected checkpoint
test evaluations**. The control is repeated within this comparison. Each
seed's winner is selected independently by validation accuracy, with macro
F1 and then declared candidate order resolving ties. Test metrics from
unselected candidates are not produced. An identical relaunch resumes
completed work and restarts incomplete fits from epoch 1.

The resulting test mean describes the validation-selected procedure, whose
augmentation setting may vary across seeds. Read `selection_report.json` to
see both candidates' validation scores for each seed and `search_report.md`
for the selected test scores. Keep all five seeds; do not report only the
best one. Previously inspected test splits remain exploratory during this
development process.

If augmentation repeatedly helps on validation and the selected results
improve, retain it for a broader subject check. If not, keep the established
spatial configuration and investigate one further feature-extraction or
training change at a time. Do not assume that more augmentation, a smaller
model or a lower learning rate is automatically beneficial.

## After training: plots

Generate diagnostics for all selected seeds in Bash:

```bash
for task_seed in 0 1 2 3 4; do
  python main.py diagnose \
    "results/eegnet-A01-spatial-augmentation/selected/eeg/subject_A01/seed_${task_seed}" \
    --output-dir "results/eegnet-A01-spatial-augmentation/analysis/diagnostics/seed_${task_seed}" \
    --device cuda --partition validation --embedding tsne || break
done
```

After all diagnostics succeed:

```bash
python main.py analyze \
  results/eegnet-A01-spatial-augmentation/selected/eeg/subject_A01 \
  --output-dir results/eegnet-A01-spatial-augmentation/analysis \
  --plots \
  --diagnostics-root results/eegnet-A01-spatial-augmentation/analysis/diagnostics
```

Result plots are indexed at `analysis/figures/index.md`; each diagnostic
directory has its own `index.md`. If augmentation choices differ across seeds,
generic analysis groups runs by configuration. Use `search_report.md` for the
overall score of the selected procedure. Download the entire new result folder
for comparison, including `candidates/`, `selected/` and `analysis/`.

## Saved-artifact audit

- All five runs completed, with no failure markers.
- Each seed's train/validation/test trial IDs match its original comparison
  run, and no trial ID overlaps partitions within a seed.
- Selected checkpoint hashes and validation-selected epochs match the records.
- Accuracy, macro F1 and macro ROC-AUC recomputed from saved predictions match
  all ten validation/test metric records.
- Diagnostic manifests report reproduced probabilities for all five seeds.
- All **33 result figures and 90 diagnostic figures** exist as nonempty PNG
  and PDF files, with no skipped figures in their manifests.

These checks establish artifact consistency and completion, not proof that
every model/preprocessing choice is optimal. No project tests, training or
inference were executed locally for this review.
