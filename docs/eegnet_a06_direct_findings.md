# A06: findings from the local recording bytes and saved results

Numerical measurements and source hashes are saved in
[the accompanying JSON](eegnet_a06_direct_measurements.json).

The raw files were available in `Downloads/ml`. The essential event and
signal checks could therefore be performed immediately; another cluster
audit is not required to establish the findings below.

This inspection used standalone file analysis with Python's standard library
and NumPy. It decoded GDF headers, event tables and calibrated sample values
directly. It did not import the AGFL project, run its loaders or audit command,
load a model, perform inference, or train anything. The parser was restricted
to the actual GDF 1.99, 25-channel, uniform int16 files and checked record and
event-table lengths. Field interpretation was checked against the
[MNE GDF reader source](https://github.com/mne-tools/mne-python/blob/maint/1.12/mne/io/edf/edf.py).

## What is now verified

For A01, A03 and A06, the local recording SHA256 matches the source identity
saved in the corresponding cluster runs. Direct event-table inspection found:

- 288 trial starts and 288 labeled cues, with 72 cues per class.
- Six task runs with 12 cues per class per run.
- Every cue is exactly 500 samples (2 seconds at 250 Hz) after trial onset.
- Every configured four-second cue window fits within its trial/run.
- Header channel positions identify Fz, C3, Cz, C4 and Pz consistently with
  the loader's channel ordering; the final three channels are explicitly
  named EOG-left, EOG-central and EOG-right.
- Expert artifact events account for the recorded exclusions: 15 on A01,
  18 on A03 and 69 on A06. A06 exclusions by class are 16, 15, 23 and 15.
- The retained cue IDs exactly match the union of saved partitions, and
  all saved partition class counts match the original event labels, for
  all five seeds of each inspected subject.
- Original cue labels match every saved validation target in those 15 runs.

These findings agree with the
[official data description](https://www.bbci.de/competition/iv/desc_2a.pdf).
They rule out the suspected file mismatch, cue offset, shifted validation
labels and incorrect artifact counts in these runs. They do not certify the
underlying physiological annotations against an independent measurement.

## The learning problem remains visible in the signal

Signal measurements below use each subject's seed-0 training/validation
windows only. Test signal quality and test predictions were not evaluated.
Physical values were reconstructed from the GDF digital/physical calibration.

The A06 training data contain much weaker class-associated differences in
simple motor-area alpha power than A03's training data:

| Training measurement | A01 | A03 | A06 |
|---|---:|---:|---:|
| Number of training trials | 163 | 162 | 133 |
| Class-associated variance of log alpha power, C3 | 30.54% | 39.78% | 2.07% |
| Class-associated variance of log alpha power, Cz | 16.15% | 20.04% | 2.68% |
| Class-associated variance of log alpha power, C4 | 21.05% | 35.13% | 4.64% |

These values are descriptive eta-squared: between-class sum of squares
divided by total sum of squares, using 8–13 Hz power from a Hann-windowed,
one-sided periodogram of each raw four-second trial. They are **not accuracy
scores**, significance tests or evidence that the subject is unclassifiable.
They examine three electrodes and one frequency band, not every spatial or
temporal feature available to EEGNet. On A06 validation, the corresponding
values are 28.65%, 20.18% and 6.12%; the small partitions differ substantially.

Raw EEG amplitude alone does not distinguish the failure: median within-trial
EEG SD is 11.95 µV on A03 training and 12.34 µV on A06 training. Neither subject
has a constant channel or a near-duplicate EEG channel pair in the measured
windows. Each has one training trial reaching the calibrated Fz limit, with
no such validation trials. There is no evidence here of widespread clipping.

Median maximum absolute within-trial EEG/EOG correlation is 0.829 for A03
training and 0.688 for A06 training. The successful A03 model has stronger
raw EEG/EOG coupling under this descriptive measure, so these numbers do not
support attributing A06's failure specifically to unusually high EOG coupling.
Correlation alone neither identifies nor rules out an artifact.

Together with the compact models' 85–87% final training accuracy and 28–30%
final validation accuracy, the evidence supports a generalization problem
on a small training set with weak class separation in these simple features.
It does not identify one proven code defect that explains the 32.56% score.

## A concrete training concern found in the code

All three A06 configurations end every epoch with a five-trial batch:

| Configuration | Batch size | Training batch sizes |
|---|---:|---|
| spatial_control | 64 | 64, 64, 5 |
| compact_train_channel | 32 | 32, 32, 32, 32, 5 |
| compact_trialnorm | 32 | 32, 32, 32, 32, 5 |

The loader uses `drop_last=False`, and EEGNet uses
[PyTorch BatchNorm](https://github.com/pytorch/pytorch/blob/v2.6.0/torch/nn/modules/batchnorm.py).
Each training
batch updates its running statistics with the same configured momentum,
regardless of batch size. Thus the final five-trial batch has disproportionate
influence per trial on the statistics used for validation. Compact BatchNorm
momentum is 0.01 and control momentum is 0.1; the magnitude of the effect can
differ. By comparison, A03's spatial control has batches 64, 64 and 34.

This is an avoidable source of statistical instability, **not yet a proven
cause of the low accuracy**. A focused next comparison can keep EEGNet, AGFL,
preprocessing and the split fixed and change only the control batch size to
48, giving batches 48, 48 and 37. Training and held-out evaluation would be
needed to establish whether that change helps. No such comparison was run
during this inspection, and no model defaults were changed.

The raw audit commands remain available for a repeatable report, but the
basic data-integrity questions above no longer need another cluster run.
The nine-subject 75% target remains unproven.

The batch-size comparison is now available as the opt-in `spatial_batch48`
candidate. See [the README's A06 training and plot commands](../README.md#improve-eegnet--agfl-on-bci-iv-2a).
It compares `spatial_control` and `spatial_batch48` for A06 seed 0, using
validation to select the checkpoint before testing the winner. No training
has been run locally. A change in performance would assess the batching
choice, not isolate BatchNorm from the other effects of batch size.

**Follow-up result:** the downloaded batch comparison selected batch 48 by
one validation trial, but its test score was 25.58%, versus the previously
evaluated control's 32.56% on identical test IDs. Removing the tiny final
batch was not sufficient to resolve A06. See the
[single-file review and all 27 plots](eegnet_a06_batch_review.html).
