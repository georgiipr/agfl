# A03/A06 results: strong A03 improvement, unresolved A06 performance

Reviewed `Downloads/eegnet-A03-A06-spatial-augmentation` against the matching
subject/seed runs in `Downloads/results_1`. This review reads saved JSON,
predictions, checkpoint bytes and figures only. No project training,
inference, dataset loading or tests were run locally.

## Results

| Subject | Earlier EEGNet + AGFL | Current selected procedure | Change | Current seed SD |
|---|---:|---:|---:|---:|
| A03 | 69.63% | **86.67%** | +17.04 pp | 4.01 pp |
| A06 | 28.37% | **31.63%** | +3.26 pp | 7.09 pp |

The two-subject mean is **59.15%**. Together with the previously reviewed
A01 result of 71.64%, the equal-subject pilot mean is **63.31% across three
subjects**. Neither number establishes the nine-subject 75% target.

| Subject | Seed | Validation-selected candidate | Validation accuracy | Test accuracy |
|---|---:|---|---:|---:|
| A03 | 0 | spatial_recombine | 98.15% | 92.59% |
| A03 | 1 | spatial_control | 90.74% | 85.19% |
| A03 | 2 | spatial_recombine | 96.30% | 87.04% |
| A03 | 3 | spatial_control | 88.89% | 81.48% |
| A03 | 4 | spatial_control | 92.59% | 87.04% |
| A06 | 0 | spatial_control | 46.51% | 32.56% |
| A06 | 1 | spatial_recombine | 39.53% | 23.26% |
| A06 | 2 | spatial_recombine | 48.84% | 25.58% |
| A06 | 3 | spatial_recombine | 44.19% | 39.53% |
| A06 | 4 | spatial_control | 53.49% | 37.21% |

A03 exceeds 75% on every seed. Its mean macro F1 is **0.862** and mean
macro ROC-AUC is **0.968**. A06 remains weak on every seed: mean macro F1
**0.305**, ROC-AUC **0.602**, and test accuracy ranges from **23.26–39.53%**.
Uniform random guessing has 25% expected accuracy for four classes.

The current scores describe validation selection between two candidates,
not always enabling augmentation. Augmentation wins validation on two A03
seeds and three A06 seeds. The unselected checkpoints have no test evaluations.
Consequently this folder cannot establish the test gain caused by augmentation
alone. The comparison with `results_1` also includes architecture and training
changes, not just augmentation.

## What the A06 evidence supports

| Saved data property | A03 | A06 |
|---|---:|---:|
| Artifact-marked trials excluded | 18/288 (6.25%) | 69/288 (23.96%) |
| Retained trials | 270 | 219 |
| Training / validation / test trials per seed | 162 / 54 / 54 | 133 / 43 / 43 |
| Retained class counts: left, right, feet, tongue | 69, 68, 66, 67 | 56, 57, 49, 57 |
| Nonfinite / out-of-bounds trials skipped | 0 / 0 | 0 / 0 |

A06 has fewer training examples and substantially more recorded artifacts.
Its final control training accuracy averages **73.38%**, while final validation
accuracy averages **36.28%**. With augmentation these are **69.47%** and
**35.35%**. Training accuracy is measured during optimization with dropout and,
when enabled, augmentation; it is not a separate clean training-set evaluation.
Nevertheless, the histories show learning on training examples without strong
held-out performance. Simply extending the 250-epoch budget is not supported.

Mean selected validation accuracy is **46.51%** on A06, versus **31.63%**
on test. Validation has been reused to choose the epoch and candidate, so its
peak is optimistic. One test prediction changes an A06 seed's accuracy by
**2.33 percentage points**.

| Class | A03 mean test recall | A06 mean test recall |
|---|---:|---:|
| Left hand | 87.14% | 29.09% |
| Right hand | 94.29% | 36.36% |
| Feet | 73.85% | 18.00% |
| Tongue | 90.77% | 41.82% |

A06 errors affect all four classes. Feet are weakest; class imbalance is
modest, so the counts do not support treating imbalance as the sole cause.

Saved preprocessing uses 22 EEG channels, 250 Hz, a four-second window from
each cue, 2–30 Hz trial-local filtering, and training-fitted channel
normalization. The artifact records, source hashes, split IDs and prediction
labels agree with the previous experiment. Both subjects retain trials from
all six task runs. The examined spectra show the expected high-frequency
attenuation; the example traces are not flat. These are normalized model-input
plots, not raw-voltage or EOG quality measurements.

This audit found no inconsistent saved labels, mismatched checkpoints or
overlapping partitions within a seed. It does **not** independently verify
the GDF annotations or prove that every preprocessing choice is optimal.
The available evidence does not identify a specific loader bug or establish
that artifacts alone caused the low accuracy. A06 remains part of the study.

## Plots are already present

All **72 result figures and 180 diagnostic figures** are present in both PNG
and PDF: **252 figures, 504 files**, with none missing, empty or marked skipped.

In `Downloads/eegnet-A03-A06-spatial-augmentation/`:

- `analysis/figures/index.md`: training curves, validation/test confusion
  matrices, ROC curves and aggregate result figures.
- `analysis/diagnostics/A03/seed_0/index.md`: A03 signals, spectra, scalp maps,
  C3/C4 time-frequency maps, CSP, embeddings and AGFL plots.
- `analysis/diagnostics/A06/seed_0/index.md`: the corresponding A06 diagnostics.
- The same diagnostic paths exist for seeds 1–4.

Use `search_report.md` for the combined validation-selected procedure.
Generic `analyze` tables group different configurations separately.

## Next small experiment: A06 only, seed 0

Compare the current control with the existing compact EEGNet settings, with
and without per-trial normalization. This costs **three candidate fits and
one selected test evaluation**, instead of another cohort-wide search.
It retains EEGNet + AGFL, all four classes and the same subject/split.

- `spatial_control`: repeat the current reference; its saved seed-0 validation
  accuracy is 46.51%.
- `compact_train_channel`: the existing compact recipe, with training-fitted
  channel normalization. It uses longer temporal filters, 31 time tokens,
  cross-entropy, a lower learning rate and validation-loss early stopping.
- `compact_trialnorm`: the same compact recipe with per-trial normalization.
  This pair isolates normalization; the comparison with `spatial_control`
  changes several model/training settings and cannot isolate their causes.

This is a diagnostic comparison, not a promised accuracy improvement. Compact
settings were weaker on A01; A06 has not yet been assessed with this comparison.
Use validation and histories to decide whether a candidate deserves a five-seed
check. A single seed or a favorable test score is insufficient to establish
success. Further raw-data quality checks require the cluster recordings;
the current figures cannot replace an annotation and voltage audit.

From `AGFL` on the cluster:

```bash
python main.py tune-eegnet \
  --data-dir ../ml \
  --subjects 6 \
  --seeds 0 \
  --candidate spatial_control \
  --candidate compact_train_channel \
  --candidate compact_trialnorm \
  --output-dir results/eegnet-A06-compact-check
```

After training succeeds, generate checkpoint diagnostics:

```bash
python main.py diagnose \
  results/eegnet-A06-compact-check/selected/eeg/subject_A06/seed_0 \
  --output-dir results/eegnet-A06-compact-check/analysis/diagnostics/A06/seed_0 \
  --device cuda --partition validation --embedding tsne
```

After diagnostics succeed, generate result plots:

```bash
python main.py analyze \
  results/eegnet-A06-compact-check/selected \
  --output-dir results/eegnet-A06-compact-check/analysis \
  --plots \
  --diagnostics-root results/eegnet-A06-compact-check/analysis/diagnostics
```

Download the complete `eegnet-A06-compact-check` folder, including `candidates/`,
`selected/`, `analysis/`, `selection_report.json` and `search_report.md`.
An identical relaunch resumes completed candidates. Use a new output folder
if changing the subjects, seeds or candidate list.

## Verification and interpretation limits

All 20 candidate fits finished 250 epochs without recorded AMP-skipped steps.
Candidate/selected checkpoint hashes match their records. Every selected
winner follows the recorded validation accuracy/F1 ranking, and the selected
epochs match their candidate records. All 20 selected validation/test accuracy,
macro F1 and macro ROC-AUC records reproduce from saved predictions.

Control and recombination configurations differ only in enabling eight-segment
training augmentation. Within each seed their trial IDs match. New and earlier
test/validation IDs and targets match, and recording source hashes agree.
The ten diagnostic manifests report matching checkpoint predictions and source
code, and their CSP training IDs match the training partitions.

The five test splits overlap: A03's 270 repeated test predictions cover 177
unique trials, and A06's 215 cover 139. Seed SDs are descriptive, not confidence
intervals for independent cohorts. These repeatedly inspected T-session splits
remain exploratory; they are not an untouched official T-to-E evaluation.
