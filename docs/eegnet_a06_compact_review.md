# A06 compact comparison: no improvement on seed 0

Reviewed `Downloads/eegnet-A06-compact-check` using saved artifacts only.
No project training, inference, dataset loading or tests ran locally.

## Result

Neither compact configuration beats `spatial_control` on validation.
The selected control gets **14/43 test trials correct: 32.56%**, exactly
matching its previous A06 seed-0 result. Its validation and test prediction
arrays are identical to the previous control's arrays.

| Candidate | Selected validation accuracy | Epoch selection | Checkpoint epoch | Epochs trained | Test accuracy |
|---|---:|---|---:|---:|---:|
| spatial_control | **46.51%** | Maximum validation accuracy | 38 | 250 | **32.56%** |
| compact_train_channel | 39.53% | Minimum validation loss | 56 | 116 | Not evaluated |
| compact_trialnorm | 37.21% | Minimum validation loss | 25 | 100 | Not evaluated |

The compact checkpoints were intentionally selected by validation loss.
`compact_trialnorm` briefly reached 44.19% validation accuracy at epoch 27,
but that checkpoint had a higher validation loss. Even its maximum recorded
accuracy remains below the control's 46.51%. `compact_train_channel` never
exceeded 39.53%. Changing the compact checkpoint criterion to accuracy would
therefore not change the candidate winner on this seed.

The unselected candidates have no test predictions, by design. Their test
accuracy must not be inferred from their validation scores. This is one
subject and one split; it does not show that compact EEGNet can never help
A06 or estimate nine-subject performance.

## What the histories show

| Candidate | Final training accuracy | Final validation accuracy |
|---|---:|---:|
| spatial_control | 75.94% | 30.23% |
| compact_train_channel | 87.22% | 30.23% |
| compact_trialnorm | 84.96% | 27.91% |

Training accuracy is measured during optimization with dropout, rather than
in a separate clean training-set evaluation. Still, the compact histories
show that fitting training examples more successfully did not improve
held-out performance. Their validation losses rose after their selected
epochs. Their 116- and 100-epoch stopping points follow the configured
patience and minimum epoch budget; these are completed runs, not crashes.

The control's test macro F1 is **0.324** and macro ROC-AUC is **0.593**.
Errors remain spread across the four classes. This result provides no
support for a larger compact search or a longer training budget yet.

## Data and artifact checks

All candidates use the same 219 retained A06 trials and the same partition
assignments: **133 training, 43 validation, 43 test**. All four classes are
present. Metadata records 69 artifact-marked exclusions and no nonfinite
or out-of-bounds exclusions, unchanged from the preceding study.

The per-trial-normalized candidate has a different dataset fingerprint and
split ID because its signal values and preprocessing metadata differ. Its
actual trial IDs and class counts in each partition match the other two
candidates. This is not a different random trial assignment.

Verified from saved files:

- All three candidates completed, with no failure markers or recorded
  AMP-skipped optimization steps.
- All candidate and selected checkpoint hashes match their own records.
- Selected epochs and metrics match their histories and configured criterion.
- Candidate selection follows validation accuracy and macro F1.
- The selected validation/test accuracy, macro F1 and macro ROC-AUC reproduce
  from saved prediction arrays.
- Within-seed partitions are disjoint; current and previous control splits,
  histories and predictions match.
- Recording source hashes agree across candidates. The diagnostic manifest
  reports matching source code and reconstructed checkpoint predictions.
- Diagnostic examples come from validation, and CSP fitting IDs match the
  training partition.

Checkpoint file hashes need not match across output folders: checkpoints
also contain saved configuration, including paths. Each checkpoint here
matches its own recorded hash; identical prediction arrays establish that
the repeated control's predictions did not change.

These checks establish consistency of the experiment artifacts. They do
not independently validate the original GDF annotations or identify the
cause of A06's weak performance. The existing figures show normalized model
inputs, so they cannot substitute for a raw-voltage and annotation audit.

## Plots in this download

All **18 diagnostic figures** exist in PNG and PDF, with no missing, empty
or skipped files. Their index is:

`analysis/diagnostics/A06/seed_0/index.md`

The downloaded folder contains no `analysis/figures/index.md`, result-figure
manifest or aggregate analysis tables. The download does not establish
whether those files were generated on the cluster and omitted during copying.

No retraining is needed to generate them. From `AGFL` on the cluster, run:

```bash
python main.py analyze \
  results/eegnet-A06-compact-check/selected \
  --output-dir results/eegnet-A06-compact-check/analysis \
  --plots \
  --diagnostics-root results/eegnet-A06-compact-check/analysis/diagnostics
```

Download the updated `analysis/` directory afterward. Result figures will
be indexed by `analysis/figures/index.md`; the existing diagnostic figures
remain indexed under `analysis/diagnostics/A06/seed_0/`.

## Next decision

Keep the spatial control as the reference and do not expand the compact
experiment to five seeds on the strength of this result. Before another
model change, the next diagnostic work should inspect A06's original
cue/label alignment, trial timing, raw channel quality and differences
between acquisition runs, using training and validation partitions.

The saved files do not contain the raw voltage information needed to finish
that audit. More hyperparameter runs alone would not resolve those questions.
The 75% nine-subject target remains unmet; this repeat adds no accuracy gain.

The subsequent [raw EEG audit workflow](eeg_raw_audit.md) now provides
`audit-eeg` and `plot-eeg-audit`, with cluster commands and a single-file
HTML report. It requires no further training.
