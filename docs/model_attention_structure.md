# Models and attention are independent

A model key selects the complete backbone. An attention key selects the
mechanism inserted into that backbone. Both are stored in configs, results,
paths, tables and plot labels. Changing attention does not substitute a generic
Signal Transformer for EEGNet or another selected model.

| Model | EEG default | ECG default |
|---|---|---|
| EEGNet | Per-electrode convolutions; residual electrode attention; learned spatial filters shared across time bins. | Lead-spanning convolutions; residual temporal attention; flattened readout. |
| EEGEncoder | Per-electrode convolution/TCN; local and attention features share a learned spatial readout. | Convolution across leads, parallel causal TCN and temporal attention; fused readout. |
| DSTS EEGEncoder | Per-electrode down-projection; branch-specific TCN, electrode attention and learned spatial readout; logit ensemble. | Down-projection across leads; temporal/attention branches; logit ensemble. |
| Conformer | Per-electrode temporal stem; electrode attention and pointwise convolution; learned spatial readout plus max. | Lead projection; temporal attention/depthwise convolution; mean+max readout. |
| Signal Transformer | Per-electrode temporal encoding; residual attention/FFN; learned spatial readout. | Temporal patch embedding; residual attention/FFN; mean temporal readout. |

Every attention module consumes and returns [batch, tokens, features].
The same five attention implementations work at each location. Heads must divide
the actual feature dimension; this is checked when the model is built.
EEGNet electrode features have width f2 × pooled_time, unlike its temporal
attention width f2. Linformer sequence projections use the actual token count.

## Changes relative to the original sources

Electrode EEG variants require different spatial aggregation: the original
EEG models collapsed channels before attention. Per-electrode feature extraction
and readout adaptations are intentional new variants. ECG variants of the three
EEG families also introduce explicit lead/time shapes and suitable pooling.
EEGNet, EEGEncoder and DSTS add fixed positional encoding to electrode tokens;
this preserves channel identity through shared per-electrode extraction and
spatial readout. Conformer and Signal Transformer use learned position embeddings.

All electrode variants now use feature-specific signed spatial filters
(`model_options.spatial_readout=learned`). EEGNet shares electrode weights across
pooled time bins and constrains their max norm. Its `attention_residual=true`
preserves the local signal and its gradients when AGFL taps are zero, for EEG
and ECG. AGFL equations are unchanged. These are architecture revisions, not
demonstrated accuracy gains. Older checkpoints reconstruct the previous mean
readout and EEGNet non-residual path using their saved configuration.

EEGNet no longer executes a training-mode dummy forward to calculate classifier
size. Nondefault f2 uses a valid depthwise/separable convolution. Max-norm
constraints are still applied. EEGEncoder uses one evaluation of each causal
residual block and supports differing input/TCN feature widths. Its dynamic
positional encoding uses the selected token count. DSTS requires explicit
[B,C,T] shape and passes the real block index/depth to scheduled AGFL.
Conformer EEG performs temporal convolution in its electrode stem; block
convolution uses kernel 1 because electrode order is not a temporal neighborhood.

EEGNet, EEGEncoder and DSTS EEG variants expose
`model_options.attention_axis=time` for temporal placement. This changes
the comparison identity and must be treated as a different architecture.
It does not promise original behavior for every corrected implementation.
Frozen-weight parity checks specifically cover EEGNet with its original default
dimensions, temporal placement and `attention_residual=false`, and Conformer ECG. They avoid conflating
controlled forward behavior with changed initialization/training policies.

Original independent sources are retained under tests/references. AGFL layer
equations retain their own independent forward/gradient reference. No local
execution has established numerical parity or training performance.

## Registry and comparison contract

`agfl/models` discovers packages exposing ModelSpec. Each model package owns
defaults and EEG/ECG constructors. Shared methods implement data preparation,
training and evaluation. `agfl/attention` independently discovers AttentionSpec
packages; the model receives a factory containing the selected mechanism/options.

Attention initialization is isolated from the backbone RNG stream. For matching
seed/model/options, convolutional and classifier initial weights are unchanged
when attention changes. Mechanism parameters and capacity can differ.
Dropout surrounding attention belongs to the backbone; mechanisms do not add
their own attention-probability dropout. No equality with the original
StandardAttention training dropout is claimed.

Configuration schema 2 uses model_options for architecture and attention_options
for mechanism settings. Comparison identity fixes model key, full model options,
head count, data/split/training settings and source/packages. Attention key and
mechanism options are excluded so declared attentions/ablations can be compared.
Cross-backbone scores are reported, but not paired as an attention-only effect.

## Earlier runs

Schema 1 stored mechanism names in the model field for a generic shared
backbone. The reader labels those runs Signal Transformer plus the corresponding
attention (transformer becomes mha), preserving original stored config and
identity hashes. The explicit signal_transformer model remains available,
with both variants, to keep this architecture identifiable.

New experiments require schema 2. Original-backbone v1 checkpoints that depended
on removed adapters need their original checkout for inference; saved-metric
analysis remains available. This compatibility reader does not create public
archived model keys. Use a fresh output root after copying the corrected source
to the experiment machine and keep one source version throughout each study.
