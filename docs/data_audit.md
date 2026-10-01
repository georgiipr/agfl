# Dataset protocols and extension contract

Dataset roots are configuration values, resolved from the launch directory.
Raw recordings are read without modification. Each run saves preprocessing
settings, source checksums, sample IDs, exclusion counts and split identities.

## BCI Competition IV 2a

The `eeg` adapter trains each requested participant separately. No weights or
normalization statistics are shared between participants. Its default protocol
uses class-stratified 60/20/20 training/validation/test partitions within the
participant's T session, rounded within each class.

| Setting | Default |
|---|---|
| Files | `A01T.gdf` through `A09T.gdf`, for the requested subjects |
| EEG channels | First 22 physical EEG channels; EOG excluded |
| Labels | Cue codes 769–772: left hand, right hand, feet, tongue |
| Epoch | 1,000 samples starting at the cue; 4 seconds at 250 Hz |
| Filtering | 2–30 Hz Butterworth SOS, zero phase, separately within each recording run |
| Artifact policy | `include`; marks are counted, `exclude` is available explicitly |
| Normalization | `train_channel`: each channel's mean/SD fitted on training trials only |

An artifact policy of `include` retains marked cued trials; it does not make
invalid windows usable. Boundary-crossing, nonfinite and otherwise unusable
windows are still excluded and counted. Finite spans and sustained GDF missing
markers are handled before filtering. The same loaded samples and persisted
split are shared across compared models and attentions.

`filter_scope=trial` provides independent trial filtering. Choosing run versus
trial filtering changes edge handling and dataset identity. Run-level filtering
uses surrounding recording samples and is not strictly trial-local processing.
All supported filtering here is offline and zero phase; it is not a causal
online preprocessing claim. Continuous filtering across run boundaries is
retained for saved-checkpoint compatibility, not new within-subject training.

The `eeg-session` preset trains and validates on T and tests on E separately
for each participant. By default 20% of T trials are reserved for validation.
E-session unknown cues (783) require official matching `AxxE.mat` files with
`classlabel`; missing labels are rejected. Set `data.labels_dir` if label files
are stored separately. Within-T and T-to-E scores describe different protocols.

See the [common launch instructions](../README.md) and
[EEG electrode-token contract](eeg_interchannel.md).

## MIT-BIH ECG

The `ecg` adapter reads WFDB `.hea`, `.dat` and `.atr` files. The default task
maps `N` to class 0 and `V`, `A`, `L`, `R`, `F` to class 1. Other annotations
are discarded and counted. This is a binary beat task, not the AAMI five-class
benchmark.

The loader selects MLII by name, filters each recording independently at
0.5–45 Hz, extracts centered 256-sample beat windows, and standardizes each
beat/channel. Sampling rates come from recording metadata; mixed rates require
explicit resampling. Requested records without MLII are recorded and skipped
by default, or rejected with `missing_lead=error`.

Training uses patient-group splits. Records 201 and 202 share one patient group
and cannot be separated between partitions. A random beat split is not a
subject-independent evaluation and requires an explicit overlap override.

## Split and normalization provenance

Dataset identity includes preprocessed signal/label content, stable sample and
group IDs, preprocessing settings and raw-source checksums. Split identity adds
the algorithm version, protocol, fractions and seed. The model key is excluded
so matched comparisons can reuse identical samples.

Persisted splits record all partition indices, sample IDs, groups and class
counts. Reuse validates identity, bounds, coverage, overlap and required class
coverage. Validation/test samples are never used to fit training statistics or
training augmentation donors. Changing normalization, filtering, artifact
policy or source data changes the experiment rather than updating old results.

## Preprocessed arrays and new datasets

`npz_eeg` and `npz_ecg` accept NPZ files with:

- `x`: finite real signals with shape `[samples, channels, time]`.
- `y`: integer class labels with shape `[samples]`.
- `groups`: subject/group IDs with shape `[samples]`.
- Optional `sample_ids`: distinct identifiers in signal order.

Use `data.path` for the file and `data.normalization=none` for inputs that should
not be normalized again. NPZ loading disables pickle. The adapter does not
silently transpose axes, pad variable lengths or infer physical channel names.
Generic NPZ EEG does not automatically acquire BCI's subject-by-subject expansion;
declare the intended split protocol and groups explicitly.

For a new adapter, register a loader under `agfl/datasets` that returns
`SignalDataset`. It supplies finite `float32 [N,C,T]` signals, `int64 [N]`
labels, stable IDs, groups, modality and appropriate physical metadata. Existing
backbones consume this contract through their modality-specific constructors.
