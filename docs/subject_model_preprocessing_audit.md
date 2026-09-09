# Corrections after the five-seed EEG study

The downloaded study trained pooled EEGNet+AGFL models on five subjects and
tested on two unseen subjects. Test accuracy was 29.24% ± 2.48 percentage points;
saved probabilities independently reproduced its reported metrics. This did not
implement the requested individual-subject methodology. The changes below are
source corrections and explicit model/protocol revisions. No local training,
inference or test suite was run; improved accuracy is not yet established.

## Subject protocol and outputs

The BCI adapter expands subjects 1–9 into nine configurations with singleton
`data.subjects` and an explicit `subject_id`. CLI previews and Python launches
use this expansion. Each subject/seed gets fresh weights, optimizer, loader RNG
and training-only normalization; no state passes between subjects. Splits are
persisted by data fingerprint and seed, independently of model/attention.

The default T-only study uses class-stratified 60/20/20 trial partitions. The
separate `eeg-session` preset splits T into 80/20 train/validation and reserves
officially labeled E for test. Group-split BCI training is rejected. Paths include
`subject_Axx`. Reports show seed means/SD for each subject, then equal-weight
subject means and between-subject SD. Seed runs are not counted as new people.
Unequal seed sets are disclosed and suppress overall aggregate scores.

## Model review

All public models were inspected through their EEG/ECG input transformations,
convolutions, residual paths, attention insertion, token counts and classifiers.
All return unnormalized `[batch, classes]` logits. The shared loss consumes those
logits; softmax occurs only when producing probabilities for metrics.

| Model | Finding/change | Retained checks and limits |
|---|---|---|
| EEGNet | The electrode variant had replaced learned spatial aggregation with an equal mean. It now uses signed per-feature electrode filters, shared across pooled time bins and max-norm constrained. A residual around attention in EEG and ECG prevents zero AGFL taps from blocking feature-extractor gradients at initialization. | Temporal/depthwise/separable convolutions and original temporal-layout option remain. This electrode variant is not claimed to reproduce canonical EEGNet's early spatial convolution. Native-layout frozen-weight comparison explicitly disables the new residual. |
| EEGEncoder | Local TCN and electrode-attention features now share a learned spatial readout instead of equal means. | Causal blocks execute once; feature-width projection, residual shape and temporal attention for ECG reviewed. |
| DSTS EEGEncoder | Each EEG branch now has learned spatial filters before its classifier. | Temporal convolution padding, branch logit averaging, RMSNorm/SwiGLU residuals and correct layer/depth scheduling reviewed. ECG retains temporal branches. |
| Conformer | EEG uses learned spatial pooling plus its existing max readout. | EEG kernel-1 block convolution avoids treating electrode order as a time neighborhood; ECG keeps temporal depthwise convolution and mean+max readout. Half-step FFNs and final normalization retained. |
| Signal Transformer | EEG uses a learned spatial readout after residual mixing. | EEG electrode tokenizer and deterministic pooling retained; ECG uses temporal patches with right padding only in the last patch. |

Learned spatial readouts are a justified alternative to information-losing uniform
pooling, not proof that mean pooling was the sole cause of poor performance.
Every selected attention receives the same surrounding spatial/residual structure.
Attention initialization remains isolated so changing attention capacity does not
shift common backbone weights. AGFL equations, Top-k, coefficients, projections,
optimizer, learning rate, loss and epoch defaults were not changed in this audit.

Model options distinguish these revisions in result/comparison identities.
`spatial_readout=mean` and EEGNet `attention_residual=false` reconstruct previous
saved behavior for checkpoint diagnostics. Old artifacts are never rewritten.
New train accuracy is logged from train-mode minibatches, alongside train loss;
validation metrics remain eval-mode. This helps diagnose generalization but is
not a substitute for a separate eval-mode training-cohort measurement.

## Preprocessing review

| Area | Correction or verified source behavior |
|---|---|
| EEG timing | Default cue-relative window changed from 0.5–4.5 s to 0–4 s, matching the instructed imagery period. Cue labels remain 769–772 mapped to 0–3. |
| EEG filtering | Default changed to trial-local 2–30 Hz fourth-order SOS zero-phase filtering. New within-subject training rejects run/continuous filtering, preventing neighboring held-out trials from influencing training samples. |
| EEG normalization | Default changed from independent per-trial/channel standardization to channel mean/SD fitted only on training trials and reused for validation/test. Relative amplitude variation is preserved. |
| EEG artifacts | Expert-marked trials are excluded by default and counted; labels are indexed before exclusion, including external E labels. Inclusion remains an explicit cohort choice. |
| EEG boundaries | Windows cannot cross the next trial or run boundary. Nonfinite samples are rejected and counted. |
| ECG missing values | The GDF-specific sustained-minimum detector no longer runs on ECG; a flat baseline is not automatically a missing-data marker. |
| ECG lead/cohort | MLII is selected by name, missing leads are recorded, and 201/202 remain one patient. Patient splits keep record filtering and overlapping beat windows within the same partition. |
| ECG filtering/windows/labels | 0.5–45 Hz per-record filtering at the recorded rate; centered 256-sample windows; correct odd-window lengths. N versus V/A/L/R/F remains the declared binary task. Integer window and label-map validation tightened. |
| NPZ | Real finite nonempty `[N,C,T]` signals required; explicit groups and labels, no pickle loading, integer class count validation. No sample-rate assumptions or extra filtering is silently applied. |
| Session splits | Validation explicitly checks that training/validation contain T and test contains only E. Official E labels remain mandatory. |

This remains offline trial/beat classification, not a causal continuous decoder.
The [official BCI IV 2a description](https://www.bbci.de/competition/iv/desc_2a.pdf)
documents cue timing, event codes, channel order and expert artifact annotations.
Defaults alter cohort/preprocessing fingerprints. Old cohort accuracy must not be
compared with new per-subject test scores as an isolated AGFL improvement.

## Target-machine acceptance

Regression checks cover subject expansion without loading data, 9 × 5 attention
configurations, singleton replay, split isolation, cue/window alignment, artifact
label alignment, train-only statistics, signed spatial selectivity, EEGNet initial
feature gradients, all model/attention/modality shapes, and preserved reference
layouts. They are supplied for the cluster and have not been executed locally:

```bash
python -m pytest -m 'not real_data'
AGFL_REAL_DATA=1 python -m pytest -m real_data
```

Then run a single-subject execution check before the 45-run EEG study. Use the
README's plot commands afterward. Changing subject protocol, preprocessing and
architecture together corrects the requested study, but cannot measure each
change's separate contribution; that requires controlled follow-up experiments.
