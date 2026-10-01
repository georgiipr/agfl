# Model and attention structure

`model` selects a backbone. `attention` selects the mechanism inserted into that
backbone. Both identities are stored in configurations, results and plot labels.
`model_options` controls architecture; `attention_options` controls the mechanism.

| Model | EEG attention path | ECG attention path |
|---|---|---|
| EEGNet | Electrode tokens; selectable `compact`, `spatial_fusion` and `pre_spatial` layouts | Lead-spanning convolutions, residual temporal attention, flattened readout |
| EEGEncoder | Per-electrode convolution/TCN and attention, learned spatial readout | Convolution across leads, temporal TCN/attention branches, fused readout |
| DSTS EEGEncoder | Per-electrode features, branch-specific TCN/attention, spatial readout and logit ensemble | Lead projection, temporal branches and logit ensemble |
| Conformer | Per-electrode temporal stem, electrode attention, pointwise convolution and spatial readout | Lead projection, temporal attention/convolution and mean/max readout |
| Signal Transformer | Per-electrode encoding, attention/FFN blocks and spatial readout | Temporal patch embedding, attention/FFN blocks and temporal readout |

Each attention maps `[batch, tokens, features]` to the same shape. Head count
must divide the actual feature dimension. New EEG runs require one node per
electrode; ECG uses time or time-patch nodes. Electrode indices represent channel
identity, not physical distance, so temporal-distance biases are rejected for EEG.

## EEGNet layouts

- `compact`: independently encoded electrode features are projected to
  `electrode_dim`, mixed by attention and combined by a learned spatial readout.
- `spatial_fusion` (default): a shared temporal stem feeds both a conventional
  full-channel spatial-convolution path and an electrode-local attention path.
  Their readouts are concatenated. Attention never receives spatially collapsed
  features as if they were individual electrodes.
- `pre_spatial`: attention mixes all electrodes at each time step of the shared
  temporal stem. Its residual output feeds the full-channel spatial convolution,
  pooling and separable convolution. There is no parallel classifier branch.

For `pre_spatial`, tokens have shape `[batch * samples, channels, f1]`, and heads
must divide `f1`; `electrode_dim` is unused. Other compact electrode paths use
`electrode_dim` as their attention width. With 1,000 samples, `f2=32` and pooling
factors 8 and 16, the conventional flattened readout has 224 features.
See the [pre-spatial layout guide](eegnet_pre_spatial.md).

These electrode-preserving variants adapt the surrounding feature extraction
and readout. They do not imply numerical parity with every original backbone
or an accuracy advantage. ECG variants have their own input and pooling shapes.

## Registry and initialization

`agfl/models` discovers packages exposing `ModelSpec`; each provides defaults
and EEG/ECG constructors. `agfl/attention` independently discovers `AttentionSpec`
packages. Models receive a factory that constructs their selected attention.
Shared methods handle data preparation, training and evaluation.

The factory isolates attention initialization from the backbone RNG stream.
For matching seeds/model settings, changing attention preserves common initial
convolutional and classifier weights. Attention capacities can differ; parameter
counts are reported. Normalization, residuals, dropout and readout surrounding
attention remain properties of the backbone.

Comparison identities fix the model, model options, heads, data, splits,
training settings and relevant source/environment provenance. Attention identity
and mechanism options vary explicitly. Comparing different backbone layouts
is a separate architectural experiment.

## Dataset and checkpoint contracts

Backbones consume `[batch, channels, time]` inputs with explicit modality and
dimensions. Dataset adapters own samples, labels, grouping, preprocessing and
split identities; see [dataset protocols](data_audit.md).

Checkpoint reconstruction uses saved settings to retain the original node
axis, readout and parameter shapes. Compatibility paths do not make temporal
EEG a selectable new-training configuration and do not relabel old heatmaps
as electrode connectivity. Independent reference sources under
`tests/references` support development checks and are excluded from installation.
