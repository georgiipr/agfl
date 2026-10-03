# The SI_Hom dataset in this project

Status on 2026-10-01: the archive was built and checked by standalone inspection
of the files. The loader, the splits and everything else in this project have
not been executed yet; `python -m agfl_speech check` on the cluster is the first
real run.

## Where the data comes from

The trials are the dataset authors' preprocessed single-trial MATLAB files from
their thesis material (folder `Data_voices`). Nothing was re-filtered or
re-epoched here.

| Item | Source file | Status |
|---|---|---|
| Signals | `oneTrailMat1s/1.mat … 2640.mat`, variable `singleMatrix`, 19 × 500 | Verified |
| Subject and class of trial *i* | row *i* of `label_merged` in `label_merged_two_splited_subject_and_recorrect_theLabel.mat` | Verified |
| Recording blocks | `subjectTrails` in the same file | Verified |
| Electrode order | Raw NEDF recording header and the authors' BrainNetViewer node file | Verified, see below |
| Authors' test trials | `dataset.mat` (`data_test`) | Verified |
| Which word each class is | Word order in the authors' semantic-distance script | **Inferred** |
| Trial order within a subject is recording order | File numbering follows the epoched recordings | **Inferred** |

What "verified" means:

- **Labels.** All 2640 single-trial files were matched one-to-one against the
  authors' own `dataset.mat` (train and test arrays). Every trial occurs there
  exactly once, with identical values, and its class equals `label_merged`.
- **A trap in the source folder.** `label_Cell.mat` has the same counts but a
  different, word-sorted trial order. It does not line up with the single-trial
  files and is not used.
- **Electrode order.** The NEDF header and the node file agree. Independently,
  the signals agree: each electrode's most correlated partners are its scalp
  neighbours (Pz with P3/P4, Fz with F3/F4, O1 with O2, …), and inter-electrode
  distance against signal correlation gives r = −0.92, far outside 2000 random
  re-labellings (most extreme −0.34).

## What the trials are

| | |
|---|---|
| Trials | 2640 |
| Electrodes | P7 P4 Cz Pz P3 P8 O1 O2 T8 F8 C4 F4 Fp2 Fz C3 F3 Fp1 T7 F7 (in this order) |
| Samples | 500 = 1 s at 500 Hz |
| Subjects | S01 431, S02 218, S03 446, S04 381, S05 253, S06 506, S07 405 trials |
| Recording blocks | B01–B09: 431, 218, 446, 381, 102, 151, 361, 145, 405 trials |
| Classes | 0–7 with 331, 322, 310, 354, 347, 339, 309, 328 trials |

Blocks B05+B06 are one participant recorded in two files (S05); B07+B08 likewise
(S06). Two further participants were discarded by the authors for artifacts.

The classes are the stimulus event codes 21, 22, 31, 32, 41, 42, 51, 52: four
pairs of Chinese homophones, named `P1a, P1b, P2a, P2b, P3a, P3b, P4a, P4b`
here. The words are probably 助手/住手, 报复/暴富, 制服/制伏, 初中/初衷 in that
order; this is taken from the word order in one of the authors' scripts and was
not confirmed against the stimulus program, so figures use the neutral names.

Processing by the authors, before these files were written:

1. EEGLAB: unused channel removed, 49–51 Hz notch and band-pass filter, epochs of
   2 s, manual rejection of bad epochs.
2. Every channel z-scored within every 2 s trial.
3. The first 500 samples (1 s) kept.

Consequences: original amplitudes are gone, and because the z-score used the
2 s epoch, mean and standard deviation over the stored second are close to but
not exactly 0 and 1 (overall mean 0.117, standard deviation 0.977).

## The archive

`AGFL_speech_data/si_hom_trials.npz` holds the trials as float32 (the largest
rounding difference from the float64 sources is 2.4e-7) with labels, subjects,
blocks, source trial numbers, positions within blocks, the authors' test flag,
electrode names, class names, event codes and the sampling rate.
`si_hom_manifest.json` records checksums and counts. Array-by-array description:
`AGFL_speech_data/README.md`. The converter is `scripts/prepare_si_hom.py`.

The loader refuses to load when the archive's checksum or shape disagrees with
the manifest, when an array is missing, when the electrode order or sampling
rate differs, or when blocks and subjects are inconsistent. `check` additionally
compares the trial and subject counts with the released recordings.

## Data options

Set with `--set data.<key>=<json>` or in a config/preset.

| Key | Default | Meaning |
|---|---|---|
| `data_dir` | `"../AGFL_speech_data"` | Data folder. A relative path is resolved against the project folder, not the launch directory. |
| `subjects` | `[1,2,3,4,5,6,7]` | Subjects to load. |
| `classes` | `[0,…,7]` | Classes to load (at least two). They are renumbered 0…k−1 in their original order. `[0,1]` is the first homophone pair. |
| `cohort` | `"pooled"` | `pooled`: one model on all selected subjects. `individual`: one experiment per subject. |
| `start`, `window` | `0`, `null` | First sample and number of samples of each stored trial; `null` keeps the rest. Presets state `500`. |
| `normalization` | `"train_channel"` | See below. |

`normalization`:

- `train_channel`: each electrode is standardized with the mean and standard
  deviation of the **training** trials of that run; fitted by the engine, saved
  in the checkpoint. Given the authors' z-scoring this is a small correction.
- `per_sample`: every channel of every trial is z-scored again over the selected
  window.
- `none`: the stored values as they are (what the authors fed to their models).

## Identity and portability

Every run records a dataset fingerprint: a hash of the selected signals, labels,
subject ids, sample ids and metadata, including the checksum of the archive
file (not of the manifest). It does not include the data folder's location or the cohort
setting. With the default relative `data_dir`, configurations, run folders and
identities are the same on the Mac and on the cluster. Passing an explicit
`--data-dir` puts that string into the run identity.

A rebuild on the Mac reproduced the archive byte for byte, so the fingerprint is
stable there. Another numpy version may write different bytes and so a
different fingerprint, even though the arrays are identical. Keep one archive
for the lifetime of a result set.

## Cohorts and splits

Every protocol produces train, validation and test partitions. Checkpoints are
selected on validation data only; the test partition is evaluated once.

| Protocol | How trials are assigned | Subject-independent | Random |
|---|---|---|---|
| `stratified` | Within every subject and class, 60/20/20 at random | no | yes, per seed |
| `chronological` | Within every subject, the first 60% of trials (in file order) train, the next 20% validate, the last 20% test | no | no |
| `group` | Whole subjects: 5 train, 1 validation, 1 test | yes | yes, per seed |
| `authors` | Test = the authors' 528 test trials; the rest is split into train and validation within every subject and class (`split.validation_within_train`, default 0.2) | no | validation only |

Sizes: pooled `stratified` is about 1584/528/528 trials. For individual
subjects the test partition is about 43 (S02) to 101 (S06) trials.

What each protocol can and cannot show:

- `stratified` and `authors` draw train and test trials from the same recording
  at random. Slow changes over a recording are then shared by both partitions,
  which can help any model. This is the dataset authors' setting.
- `chronological` tests on the latest trials of each subject, so it is stricter
  against such drifts. It relies on the inferred trial order, and partitions are
  not class-balanced (a small subject's validation or test part can hold as
  few as 2–3 trials of a class).
- `group` is the only subject-independent protocol. With 7 subjects each seed
  tests on one subject, so seed-to-seed spread mostly reflects which subject was
  drawn. It cannot be combined with `cohort=individual`.

Splits are stored in `splits/` and reused by every attention arm, so arms of one
comparison always see identical partitions.

## Statistical cautions

- Chance is 12.5% for 8 classes. Near 22% accuracy, one standard error is about
  1.8 points on 528 test trials and 4–6 points on one subject's test trials.
- The authors' note reports about 22% test and 30% train accuracy for a plain
  EEGNet on a pooled random 80/20 split without a validation set, keeping the
  epoch with the best test accuracy. Validation-selected results from this
  project are not comparable to that number.
- Seeds re-draw the split (except `chronological`) and the initialization; they
  are not independent samples of subjects.
