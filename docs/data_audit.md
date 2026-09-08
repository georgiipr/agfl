# Dataset audit and reproducibility contract

The initial inspection found all eighteen BCI IV 2a GDF recordings in `../ml`
and all 48 MIT-BIH WFDB recordings in `../mit-bih-arrhythmia-database-1.0.0`.
No official BCI evaluation `.mat` label files were present in `../ml`.
Dataset roots are configuration values; the resolved run configuration records
their locations. Raw recordings are never modified.

| Finding in the original pipeline | Correction or explicit protocol | Comparability |
| --- | --- | --- |
| EEG used an unstratified 80/20 split of each subject's T session with split seed fixed at 42; the validation partition was also the reported final evaluation. | Separate persisted train/validation/test partitions. `group` holds out people; `stratified` explicitly denotes a subject-dependent trial split. `session` uses E for test and splits T for train/validation. | These are different evaluation protocols; historical validation scores cannot be interpreted as new test scores. |
| E-session loading expected the labeled cue events 769–772, although evaluation GDF contains unknown cue 783. | Loading unknown cues requires official matching `AxxE.mat` files containing `classlabel`; fail with a clear error if labels are absent. | Correctly labeled E results are a new benchmark; missing labels are never guessed. |
| Whole EEG recordings were filtered across run boundaries without a missing-data policy. | The default isolates runs and finite spans; sustained negative-limit GDF separators are marked unusable. Trials touching unusable samples are excluded and counted. A trial-local filter and explicit `continuous` mode are available. | SOS filtering and run separation can change preprocessed signals; settings and source checksums are part of dataset identity. |
| Artifact markers were implicitly ignored. | `artifact_policy` is explicitly `include` (historical default) or `exclude`; counts are stored. | Exclusion changes the cohort and split fingerprint. |
| ECG record 201 was in training and 202 in validation. These records are from the same person. | Both records have group `mitdb:201_202`; group splitting cannot separate them. | Corrected subject-independent results are not comparable to the old leaking split. |
| ECG always chose signal column zero. Record 114 has V5 first and MLII second; record 104 has no MLII. | Default selects `MLII` by name. Records without that lead are explicitly excluded and recorded (`missing_lead=skip`) or rejected (`error`). `lead=first` restores the historical selection for parity studies. | Default old cohort requests 44 records and loads 43 after excluding 104. The correction changes record 114's lead and removes 104. |
| ECG assumed a 360 Hz sample rate and silently mishandled odd window lengths. | Read sampling rate from WFDB metadata; mixed rates require explicit resampling. Left/right window sizes sum to the configured length. | The actual MIT-BIH 360 Hz, 256-sample default is preserved. |
| ECG binary labels included N versus V/A/L/R/F and discarded other symbols. | Preserve and record this exact configurable label mapping and discarded-annotation counts. | This remains the project's binary beat task; it is not an AAMI five-class experiment. |
| Dataset paths were resolved from launch cwd and missing EEG subjects silently skipped. | Explicit paths and required-file errors; no silent missing-subject cohort changes. | Files intentionally excluded by lead/artifact policy remain documented. |
| Per-trial EEG channel and per-beat ECG standardization removed within-sample mean and scale. | Keep `normalization=per_sample`; `none` is available. `train_channel` leaves data unnormalized until the engine fits channel statistics on training samples only. | A normalization change creates a different dataset fingerprint and experiment setting. |

All signals use `float32 [N,C,T]`, labels use `int64 [N]`, and every sample has
a stable ID plus a subject group. EEG retains 22 physical channels; ECG retains
one selected lead. There is no implicit transposition or variable-length padding.
Dataset loaders do not augment validation or test samples. Any training
augmentation belongs in the training data view and uses the seeded engine.

Filtering is offline and zero phase. A whole beat or trial is the inference
sample, including its own normalization statistics. These choices do not claim
causal online inference or compliance with the original competition's continuous
causal evaluation protocol. The historical EEG interval remains 0.5–4.5 seconds
after cue onset; this includes approximately 0.5 seconds after the instructed
imagery interval. It is explicit and can be evaluated as a controlled window
ablation rather than silently changed to improve accuracy.

`get_split` hashes the complete preprocessed signal/label content, sample and
group IDs, resolved preprocessing, raw source SHA256 checksums, split algorithm
version, protocol, fractions, and split seed. The model key is deliberately
absent. The saved JSON contains all three index lists, corresponding sample IDs,
subject IDs, class counts, and the identity fields. Existing files are checked for
identity, coverage, overlap, bounds, class coverage in training, and subject
independence before reuse. Editing a persisted sample identity is an error.

`group` defaults to fractions `train=0.6`, `validation=0.2`, `test=0.2`.
With nine EEG subjects, rounding gives five/two/two people. `stratified` allocates
each class across all partitions and is explicitly subject dependent. ECG beat
splitting requires `allow_subject_overlap=true` because overlapping windows and
repeated subjects make that protocol unsuitable for subject-independent claims.
`session` requires labeled T and E sessions and uses `validation_within_train`
(default 0.2) to reserve validation trials from T. `diagnostic_eeg` exactly recreates
the historical seed-42 torch 80/20 indices, returns no test set, and sets
`diagnostic_only=true`; the production engine must reject it.

Optional diagnostic setting names are now `continuous` and `diagnostic_eeg`.
Default run filtering and group splits are unchanged. A v1 checkpoint using an
obsolete diagnostic preprocessing name requires its training checkout; changing
preprocessing names in a saved configuration invalidates its identity.

For extension, add a module under `agfl/datasets` and decorate its loader with
`register_dataset`. Discovery imports registered modules automatically. Dataset
plugins return `SignalDataset`; model families own their modality-specific input
adapters while sharing this dataset identity/split contract. `npz_eeg` and
`npz_ecg` accept explicit `x`, `y`, `groups`, and optional `sample_ids` arrays
without unsafe pickle loading. Synthetic datasets are marked as fixtures and
must never be reported as evidence about EEG/ECG classification quality.

Primary references: the [BCI IV 2a data description](https://www.bbci.de/competition/iv/desc_2a.pdf)
documents cue codes, channel order, run separators and artifacts; the
[MIT-BIH database](https://physionet.org/content/mitdb/1.0.0/) documents recording
and subject counts. The local MIT-BIH `mitdbdir/intro.htm` explicitly states
201/202 are the same subject and record 114 has reversed signal order; `202.hea`
also identifies its shared source with 201.
