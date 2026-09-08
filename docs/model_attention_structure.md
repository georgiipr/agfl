# Models and attention are independent

A model key selects the complete backbone. An attention key selects the
mechanism inserted into that backbone. Both are stored in configs, results,
paths, tables and plot labels. Changing attention does not substitute a generic
Signal Transformer for EEGNet or another selected model.

| Model | EEG default | ECG default |
|---|---|---|
| EEGNet | Temporal/depthwise/separable convolutions per electrode; attention over flattened electrode features; mean electrode readout. | Convolutions combine leads and pool time; attention over pooled time tokens; flattened readout. |
| EEGEncoder | Convolution and causal TCN per electrode; electrode attention in parallel with local features; fused mean readout. | Convolution across leads, parallel causal TCN and positional temporal attention; fused readout. |
| DSTS EEGEncoder | Down-projection and branch-specific causal features per electrode; each branch's transformer mixes electrodes; ensemble of branch logits. | Down-projection across leads; branch-specific causal and temporal-attention paths; ensemble of logits. |
| Conformer | Per-electrode temporal stem, pooling and embedding; Conformer blocks attend over electrodes with pointwise convolution; mean+max readout. | Input lead projection and Conformer temporal attention/depthwise convolution; mean+max readout. |
| Signal Transformer | Per-electrode temporal convolution/pooling; residual attention/FFN blocks. | Strided temporal patch embedding followed by residual attention/FFN blocks. |

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
mean readout. Conformer and Signal Transformer use learned position embeddings.

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
dimensions and temporal placement, and Conformer ECG. They avoid conflating
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
