# Raw EEG audit for the unresolved A06 result

Subsequent [direct inspection of the locally available GDF files](eegnet_a06_direct_findings.md)
completed the basic source/event/label checks and raw signal comparisons.
The commands below remain available to produce a repeatable, full audit
report; they are not required to establish those already documented findings.

`audit-eeg` compares a saved subject/seed run with its original BCI IV 2a
T-session recording. It runs on CPU and does not load a model checkpoint,
train a classifier, change preprocessing or regenerate a split.

This is the next diagnostic step after the compact A06 comparison failed
to improve validation performance. It does not promise an accuracy gain.
The original recording is needed on the cluster; saved normalized-input
figures alone cannot answer raw-signal quality questions.

## Launch on the cluster

Update the cluster checkout with these changes first. From `AGFL`, with the
existing environment active and `A06T.gdf` in `../ml`, check the new audit code:

```bash
python -m pytest tests/test_eeg_audit.py
```

These tests use synthetic recording fixtures. They check artifact/cue
alignment, boundary crossings, corrupt saved labels, overlapping splits,
undefined correlations, exclusion of test windows from signal summaries,
repeat-run behavior, and plotting without reopening recordings. They are
not a real-data validation or evidence of classification performance.

Generate numerical measurements and the text report:

```bash
python main.py audit-eeg \
  results/eegnet-A06-compact-check/selected/eeg/subject_A06/seed_0 \
  --data-dir ../ml \
  --output-dir results/eegnet-A06-data-audit
```

Then generate the audit plots and the single-file HTML report:

```bash
python main.py plot-eeg-audit results/eegnet-A06-data-audit
```

Download **`results/eegnet-A06-data-audit/report.html`** for one self-contained
report with embedded figures. Download the whole audit directory for review
of numerical measurements and source identities.

An identical audit command overwrites its own outputs. Use a new output
directory for a different saved run. After rerunning the audit, rerun the
plot command so figures correspond to the new measurements. A failed audit
records `failure.json` and removes the previous completion marker.

## What it checks

- Original GDF bytes must match the recording SHA256 saved during training.
- Integer-coded events are interpreted separately from the loader's automatic
  annotation-ID mapping, then compared with saved trial IDs and validation
  prediction labels.
- The event inventory checks cue counts, class counts per task run, one cue
  per trial, cue timing, imagery-window bounds and artifact flags.
- Reconstructing the saved preprocessing checks the dataset fingerprint,
  retained trial order, class labels and split assignments.
- Raw EEG amplitudes, constant/nonfinite channels, EEG/EOG correlations,
  spectra before/after the saved bandpass, class band power and acquisition
  run differences are measured on training and validation windows.

The first 22 recording channels are treated as EEG and the final three as
EOG according to the data format. Original channel names and MNE channel
types are saved, along with the assumed physical EEG names. Amplitudes use
MNE's calibrated voltage output converted to microvolts. No EOG correction,
new artifact threshold or extra trial rejection is applied.

The full recording and preprocessed dataset are loaded for identity checks.
Test IDs and class counts are checked structurally, but test predictions
and test signal measurements are omitted. No performance-based selection
occurs during the audit. The example traces use the first chronological
measured training trial of each class.

The [official BCI IV 2a description](https://www.bbci.de/competition/iv/desc_2a.pdf)
specifies the 25-channel layout, 288 trials in six task runs, cue/class codes,
artifact marker 1023, and imagery timing. The implementation uses MNE's
[GDF reader](https://mne.tools/stable/generated/mne.io.read_raw_gdf.html) and
[integer event mapping](https://mne.tools/stable/generated/mne.events_from_annotations.html).
Both annotation interpretations use MNE and the same GDF; agreement does
not establish independent ground truth for the original annotations.

## Outputs and interpretation

| File | Contents |
|---|---|
| `report.html` | Single-file report with embedded figures, produced by the plotting command |
| `report.md` | Check results and text summary |
| `audit.json` | Source identities, package versions, event inventory and numerical signal measurements |
| `trials.csv` | Original cue IDs, labels, timing, artifact flags and saved partition assignments |
| `figures/index.md` | Figure index with PNG previews and PDF links |
| `figures/manifest.json` | Figure inventory linked to the exact `audit.json` hash |

For complete usable data, plotting produces **eight figures in both PNG and
PDF**: trial inventory, timing/artifacts, channel amplitudes, trial quality,
raw/filtered spectra, EEG/EOG correlations, class band power and raw examples.
Missing inputs are explicitly recorded as unavailable rather than replaced
with fabricated plots.

Resolve failed source, label or timing checks before a model change. Raw
amplitude shifts or high EEG/EOG correlations are inspection leads, not proof
of a particular artifact or automatic reasons to exclude trials. Class power
maps are descriptive and are not baseline-relative ERD. These measurements
cannot establish a 75% accuracy result.

No project code or tests were executed locally while implementing this
command. Local validation was limited to static source and command syntax
checks. Run the target-machine tests above before the real-data audit.

## Missing result plots from the compact comparison

The raw-audit plots are separate from the classification result plots. To
complete the latter without retraining:

```bash
python main.py analyze \
  results/eegnet-A06-compact-check/selected \
  --output-dir results/eegnet-A06-compact-check/analysis \
  --plots \
  --diagnostics-root results/eegnet-A06-compact-check/analysis/diagnostics
```
