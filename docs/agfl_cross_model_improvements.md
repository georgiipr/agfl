# Improve AGFL across backbones while retaining strong accuracy

The target is high absolute accuracy and an AGFL improvement over each attention
baseline **within each backbone**. For four-class BCI IV 2a, the earlier 75%
target refers to the equal-weight mean across nine separately trained subjects.
A03 alone cannot establish that target or an advantage across all models. Its
latest completed EEGNet comparison is 84.81% AGFL versus 85.56% MHA, averaged
over seeds 0–4. The new QKV candidates have not yet supplied measured results.

## Shared accuracy improvement already prepared

The learned Q/K/V, dense polynomial, one-hop initialization is implemented in
the shared AGFL attention module, so it can be used by every model, in both EEG
and ECG. It starts with the MHA computation, while trainable zero-hop and higher
hop coefficients can subsequently change the filter. This is an initialization
property, not a guarantee that training retains or improves MHA's accuracy.

The attention options are:

```json
{
  "projection": "qkv",
  "qkv_bias": true,
  "filter_projection": "none",
  "score_scaling": "sqrt_dim",
  "coefficient_init": "one_hop",
  "coefficient_activation": "identity",
  "learnable_coefficients": true,
  "agfl_variant": "polynomial",
  "top_k": null,
  "graph_normalization": "softmax"
}
```

Place these under `attention_options` in a model's configuration, alongside
`attention: agfl`. Keep its model settings and training recipe explicit. Heads
and maximum hop order are deliberately omitted above, retaining the selected
model's defaults:

| Model | Heads | Maximum hop order K | Existing local feature path |
|---|---:|---:|---|
| EEGNet | 4 | 2 | Residual around attention |
| EEGEncoder | 4 | 3 | Local TCN features added to attention features |
| DSTS EEGEncoder | 2 | 1 | Residual attention blocks and a temporal branch |
| Conformer | 4 | 2 | Residual attention, FFN and convolution blocks |
| Signal Transformer | 4 | 2 | Residual attention and FFN blocks |

K is the maximum hop, so there are K+1 coefficients. With K=1 the extra
learnable contribution is the zero-hop term; there is no two-hop term.
The sparse and norm-preserving candidates are separate hypotheses and do not
start with the MHA computation. Their usefulness can depend on token count,
placement and depth. EEGNet's strong A03 recipe uses temporal attention after
spatial convolution. Its convolution, pooling or optimizer settings should not
be copied indiscriminately into other models.

## Additional implementation changes

- Reject zero coefficients with ReLU activation: the zero derivative prevents
  those coefficients from ever activating. Also reject frozen zero effective
  coefficients. Trainable identity-zero coefficients remain supported for
  earlier configurations; unlike ReLU-zero, they can learn.
- Use float32 intermediate normalization for half-precision graph normalization
  paths that divide by row sums or degrees. Their previous `1e-12` denominator
  floor can round to zero in float16 and produce NaNs on empty rows. This does
  not change the existing float32/float64 graph formulas.
- Reduce gradient-finiteness flags on the device before reading one boolean
  in the training loop. Previously the non-AMP, unclipped path synchronized
  separately for every parameter with a gradient. Detection of nonfinite
  gradients remains in place. No runtime speedup has yet been measured.
- Add whole-model checks of dense one-hop AGFL versus MHA initialization,
  logits, input gradients and shared parameter gradients. Coverage includes
  all five backbones in EEG/ECG and the alternate temporal EEG layouts of
  EEGNet, EEGEncoder and DSTS.

The default coefficient initialization and existing experiment candidates are
unchanged. The graph implementation still uses dense matrix operations after
masking; graph sparsity is not evidence of a sparse runtime or lower complexity.

## Verification and next experiment

No project code, training or tests were executed locally. On the cluster, run:

```bash
python -m pytest \
  tests/test_agfl_initialization.py \
  tests/test_agfl_backbone_initialization.py \
  tests/test_agfl_safeguards.py \
  tests/test_gradient_checks.py \
  tests/test_models.py \
  tests/test_reference.py \
  -m 'not real_data' -q
```

The next accuracy experiment remains the bounded
[A03 QKV search, with separate diagnostic and result-plot commands](eegnet_agfl_qkv_improvement.md#cluster-commands).
Do not expand the candidate grid before examining these results. If that search
is already running, finish it with its original checkout. Source changes are
part of experiment provenance: use a new output root for a new checkout and
retain the matching checkout for checkpoint diagnostics.

After the development result, fix the method or validation-selection procedure
and compare against MHA, Performer, Linformer and Nystromformer within each
backbone using matched subjects, splits, preprocessing and training budgets.
Any backbone training or placement improvement must also be available to its
baselines. Comparable tuning budgets matter when claiming superiority over
tuned alternatives. Report each backbone's absolute accuracy and paired
differences separately; one pooled positive average cannot establish wins in
every backbone. Report the full nine-subject cohort for the eventual BCI claim,
including weak subjects. Previously inspected A03 splits remain development
evidence, rather than fresh confirmation.
