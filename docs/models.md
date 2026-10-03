# Backbones and attentions

A model is a backbone; the selected attention runs inside it. In every backbone
the attention has **one node per electrode (19 nodes)** and mixes electrodes.
Nothing attends over time, and there is no backbone without attention.

Shapes below are for one batch of `B` trials, 19 electrodes, 500 samples,
8 classes. They follow from reading the code; no model has been run yet.

## `eegnet`: the AGFL project's EEGNet

Defaults ([agfl_speech/models/eegnet/config.py](../agfl_speech/models/eegnet/config.py)):
`temp_kernel=250`, `f1=16`, `d=2`, `f2=32`, `pk1=8`, `pk2=16`,
`dropout_rate=0.5`, `max_norm1=1.0`, `max_norm2=0.25`,
`electrode_architecture="spatial_fusion"`, `electrode_dim=32`.

`temp_kernel=250` is half the sampling rate (0.5 s at 500 Hz), the same rule
that gave 125 at 250 Hz in the AGFL project. The max-norm constraint is
re-applied after every optimizer step.

Three layouts (`model_options.electrode_architecture`):

**`pre_spatial`** (used by all `eegnet` presets): attention sits between the
temporal filters and EEGNet's own spatial filter.

```text
input                                   [B, 19, 500]
temporal convolution + BatchNorm        [B, 16, 19, 500]
electrode identity + attention          [B*500, 19, 16]   19 nodes, width f1, at every time step
residual, back to                       [B, 16, 19, 500]
depthwise spatial convolution (19×1)    [B, 32, 1, 500]
BatchNorm, ELU, pool 8, dropout         [B, 32, 1, 62]
separable convolution, pool 16          [B, 32, 1, 3]
flatten → classifier                    [B, 96] → [B, 8]
```

The attention width is `f1`, so the number of heads must divide `f1`.

**`spatial_fusion`** (code default): the original spatial path (as above, without
attention) gives 96 features. In parallel every electrode is encoded on its
own, projected to `electrode_dim=32`, mixed by the attention across the 19
electrodes and read out to 32 features. The classifier sees 128 features.

**`compact`**: only the per-electrode path; the classifier sees 32 features.

In all layouts pooling by 8 and then 16 keeps 3 time steps of a 500-sample
trial (500 → 62 → 3). The last 116 samples reach the classifier only through
convolution overlap.

## `eegnet_original`: the dataset authors' EEGNet

The authors' `EEGNetModel` layer for layer: temporal kernel 32, `f1=16`,
`d=2`, `f2=32`, pooling 8 and 16, dropout 0.5. The attention is hosted exactly
as in `pre_spatial` above, between their temporal and spatial filters, so the
two EEGNets differ only in settings:

| | `eegnet` (pre_spatial) | `eegnet_original` |
|---|---|---|
| Temporal kernel | 250 samples (0.5 s) | 32 samples (64 ms) |
| Max-norm (1.0 spatial filter, 0.25 classifier) | after every optimizer step | once, when the model is built (`max_norm_schedule="init"`) |
| Layout | selectable | fixed `pre_spatial` |

`max_norm_schedule="every_step"` makes it behave like `eegnet` in that respect.
`f2` must equal `f1*d`; for other widths the authors' separable convolution is a
different layer. The authors' own model has no attention; this project does not
offer a no-attention variant.

## `signal_transformer`

Defaults: `dim=64`, `depth=2`, `dropout=0.1`, `mlp_ratio=4.0`,
`eeg_temporal_bins=8`, `eeg_kernel_size=31`, `spatial_readout="learned"`.

```text
input                                         [B, 19, 500]
per electrode: Conv1d(kernel 31) → GELU →
  average into 8 time bins → Linear           [B, 19, 64]    one token per electrode
+ learned position, then 2 blocks of
  LayerNorm → attention → residual → MLP      [B, 19, 64]
LayerNorm → learned electrode readout         [B, 64] → [B, 8]
```

`eeg_kernel_size=31` is 62 ms at 500 Hz (the AGFL project used 15 samples at
250 Hz). There is no max-norm constraint in this backbone.

## Attentions

All three receive `[batch, 19, width]` and return the same shape. The backbone
adds the residual connection and any dropout around them.

**`agfl`**: unchanged from the AGFL project. Defaults: 4 heads, polynomial
variant, `K=2`, scheduled top-k, softmax graph normalization, static
coefficients initialized at zero, split-input projection. Equations:
[mathematics.md](mathematics.md). With 19 nodes the scheduled top-k keeps 15
neighbours per node in a single-layer backbone (both EEGNets); in the two-layer
Signal Transformer it keeps 15 in the first block and 6 in the second.

**`mha`**: standard multi-head self-attention with learned Q/K/V (with bias) and
output projections; 4 heads.

**`hcann`**: the `Attention` module of the dataset authors' HCANN model: joint
Q/K/V projection **without bias**, softmax of scaled dot products, output
projection. In their model it is applied to the electrode rows of each
convolutional feature map, so it also mixes electrodes. Options:

| Option | Default | Meaning |
|---|---|---|
| `heads` | 4 | Must divide the token width. |
| `dropout` | 0.0 | Dropout after the output projection (HCANN used its model dropout here). |
| `pre_norm` | false | LayerNorm before the attention (HCANN's transformer block has it). |

Both extras are off by default, so `agfl`, `mha` and `hcann` are compared at the
same place with the same surrounding layers. With the defaults `hcann` differs
from `mha` only in the missing Q/K/V bias; treat it as a second plain-attention
baseline rather than as a different mechanism.

## Fair comparisons

`analyze` pairs AGFL with `mha` and with `hcann` only when backbone, model
options, training settings, data settings, split and head count are identical
(same `comparison_id`) and the seed, split and data fingerprint match. The
presets guarantee this within each backbone. Results of different backbones are
never paired.

## Training recipe

Project defaults for any run without a `training` block: 500 epochs, batch 64,
AdamW, learning rate 1e-3, weight decay 1e-3, class-balanced cross-entropy,
warm-up cosine schedule (10 warm-up epochs), checkpoint at the lowest
validation loss, no augmentation, no EMA.

| Preset | Changes from the defaults |
|---|---|
| pooled presets (`si-hom-attentions`, `si-hom-eegnet`, `si-hom-eegnet-original`, `si-hom-signal-transformer`, `si-hom`) | 200 epochs (about 25 batches per epoch on 1584 training trials) |
| `si-hom-individual` | 500 epochs (2–5 batches per epoch per subject) |
| `si-hom-authors-recipe` | The authors' settings: Adam, batch 16, 50 epochs, constant learning rate 1e-3, no weight decay, unweighted cross-entropy; split protocol `authors`. The checkpoint is still chosen on validation loss, not on test accuracy as in their script. |

The epoch budgets are choices, not measurements; change them with
`--set training.epochs=N` and a separate `--output-dir`.

## Known limitations

- `pre_spatial` layouts form one electrode graph per time step
  (`B*500` graphs per batch). Checkpoint diagnostics average them back to one
  map per trial. For AGFL with non-static coefficient conditioning, or AGFL/MHA
  with `output_gate=true`, the per-trial coefficient and gate figures are
  skipped in these layouts (they need one attention row per trial); training and
  the other figures are unaffected. All presets use static AGFL and ungated MHA.
- Kernel lengths were scaled to 500 Hz by rule (same duration as in the AGFL
  project); EEGNet's second kernel (16 samples after pooling) and the pooling
  factors are unchanged.
