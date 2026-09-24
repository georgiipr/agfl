# AGFL definitions and attention mechanisms

## What the existing implementation actually computes

All original models imported `depricated/agfl_layer_0.py`. The top-level
`agfl_layer.py` was an unused dense variant. The active implementation is now
archived at `tests/references/agfl.py`; its mathematics remains the
default in `agfl/attention/agfl/layer.py`.

For a head with input `X_h` of shape `[B,N,d_h]`:

```
tau = clamp(temperature, 0.1, 5.0)
S = X_h @ X_h.transpose(-1,-2) / (sqrt(d_h) * tau)
s(l,L) = 0.8 + (0.2 - 0.8) * exp(-3*l/L)
k = max(1, int((1 - s(l,L)) * N))
M_ij = [S_ij >= kth_largest(S_i)]
A = softmax(mask(S, M, -infinity), dim=-1)
P_0 = X_h
P_k = A @ P_(k-1)
H = sum(alpha_k * Linear_k(P_k), k=0..K)
output = output_projection(concat_heads(H))
```

`alpha_k` are signed parameters initialized to zero. Their historical name is
`alpha_logits`, although identity activation is the default. `Linear_k` has no
bias; the final output projection has a bias. Each head has its own temperature,
coefficients, and projections. Original state-dict names and random
initialization order are retained so the default layer can load old weights.

The original threshold keeps ties; it can retain more than k edges. This is
preserved as `top_k_ties="threshold"`. `"exact"` is a separately selected ablation
with stable index-based tie breaking. The original polynomial and zero
initialization are not declared incorrect or silently replaced.

The first forward pass at zero coefficients yields only the output projection
bias. Graph/filter/input gradients through this branch are initially zero,
whereas coefficient gradients can become nonzero. This is an observable
initialization property, not evidence that the definition is wrong.
`coefficient_init="uniform"` and `"lower_order"` are explicit alternatives.
`"one_hop"` initializes `[0, 1, 0, ...]`, requires `K >= 1` and identity
activation, and makes the dense polynomial QKV/value-only configuration start
with the MHA computation. See the
[cross-model settings and checks](agfl_cross_model_improvements.md).
Zero initialization with ReLU activation is rejected because its coefficients
cannot learn at zero; frozen identity-zero coefficients are also rejected.
Zero logits remain valid with sigmoid/softmax, whose effective coefficients
are nonzero.

## Reconciliation with the supplied manuscript

Current reference: `Downloads/NEU_art_submission.pdf`, page 3, equations 1–8,
checked against the [final submission review](neurips_submission_reference.md).
The manuscript and active code differ in several ways:

| Item | Active code, preserved default | Manuscript |
|---|---|---|
| Similarity | Split input `X_h X_h^T`, learned clamped temperature | Learned Q/K projections and square-root scaling |
| Values/taps | Input features, separate learned projection per tap | Projected V, scalar tap weights |
| K convention | Maximum hop order; taps 0 through K | Number of taps; taps 0 through K−1 |
| Initial coefficients | All zero | Emphasize lower orders |
| Propagation | Direct polynomial recursion | Also describes feature-norm renormalization |

The manuscript's statement that its `K=1` is one-hop mixing conflicts with its
own sum from 0 through K−1. This repository consistently uses the original code
convention: `K=0` is the identity hop, `K=1` includes one graph hop, and `K=2`
contains three taps. Translate manuscript tap count as `K_paper = K_config + 1`.

`agfl_variant="polynomial"` uses ordinary powers of A. The explicit alternative
`agfl_variant="renormalized"` uses:

```
P_0 = V
raw_k = A @ P_(k-1)
P_k = raw_k * (norm(V, dim=-1) / (norm(raw_k, dim=-1) + 1e-6))
```

The default normalization axis for this alternative is each token's feature
axis, exactly as equation 7 specifies. The additive epsilon means preservation
is approximate at small norms. Zero vectors remain zero. `hop_normalization=
"frobenius"` is a named experimental alternative over both token and feature
axes, not the manuscript equation.

The full renormalized output retains whichever tap projections are configured.
To implement equations 1 and 6–8 without additional per-hop projections, use
`projection="qkv"`, `qkv_bias=false`, `filter_projection="none"`,
`score_scaling="sqrt_dim"`, and `hop_normalization="feature"`. The supplied
`paper-renormalized-eeg` preset makes this distinction explicit; it is not the
historical experiment. Z here denotes a head's weighted sum before the shared
multi-head output projection. No manuscript file was edited.

## Graph stages and controlled ablations

Similarity calculation, scaling, selection, normalization, and propagation are
separate functions. `top_k` accepts `"scheduled"`, a positive integer, a floating
retention ratio in `(0,1]`, or JSON `null` for no sparsification. Counts/ratios are
bounded to the token count; ratios use the historical floor convention.

`top_k_stage` accepts `"raw_scores"`, `"scores"`, or `"softmax"`. Softmax and
positive row-wise scaling preserve score order in exact arithmetic. Before/after
softmax therefore select the same support absent ties and numerical effects.
The additional choice `post_topk_renormalize` controls whether probability mass
removed by post-softmax pruning is restored. With renormalization enabled, this
matches masked softmax mathematically; disabling it changes the operator.

Graph normalization alternatives are `softmax`, `row` (nonnegative ReLU weights
normalized by row degree), `symmetric` (symmetrize nonnegative weights, then
D^−1/2 A D^−1/2), and `none` (masked raw/scaled scores, including signed values).
Symmetrization can increase retained support beyond the directed Top-k mask.
For float16/bfloat16 inputs, row/symmetric normalization and post-softmax
normalization use float32 intermediates before casting back. This keeps the
denominator floor representable and avoids overflow of half-precision degrees.
Unnormalized signed operators can amplify high powers; nonfinite losses or
unscaled gradients stop a run instead of producing publishable metrics.

The historical dense similarity construction and dense propagation still have
quadratic token dependence before/after masking. No sparse-complexity claim is
made. `K=0` is a polynomial-order ablation, not a separately registered
no-attention network. The no-attention model key has been removed as requested.

## Explicit input-dependent hop coefficients

`coefficient_conditioning="static"` retains a signed coefficient vector
`alpha_base` per head, shared across tokens and trials. The explicit
`trial_power` variant uses a zero-initialized linear map of normalized
trial-wide log-power statistics to produce one bounded correction vector
per trial. Its completed EEGNet comparison did not improve mean accuracy;
see the [archived comparison](eegnet_agfl_conditioning.md).

The current attention-only experiment selects `"token_contrast"`. With
`P_0=V`, `P_1` from the configured propagation formula, and a head's token
index `i`, it computes:

```text
d_i = concat(LN_features(V_i), LN_features(P_1[i] - V_i))
r_i = tanh(G d_i)
delta_i = (s / 2) * (r_i - mean_hops(r_i))
alpha_i[k] = alpha_base[k] + delta_i[k]
H_i = sum_{k=0..K} alpha_i[k] * Linear_k(P_k[i])
```

Both feature LayerNorms use epsilon `1e-5` and no affine parameters. The
descriptor/gate arithmetic uses FP32 under autocast, preserving FP64 for
reference calculations. `G` has shape `[K+1,2*d_h]`, no bias, and starts at
zero without consuming random initialization draws. Its initial correction
is zero. The four-head, eight-feature-per-head, `K=2` EEGNet study adds
`4 * 3 * 16 = 192` parameters. Graph construction, Top-k support and hop
propagation are unchanged; the first propagated hop is reused by the gate.

For each token, `sum_k delta_i[k] = 0`, up to roundoff. Since each component
of `r_i` lies in `[-1,1]`, the tight component bound is
`|delta_i[k]| <= s*K/(K+1)`. At `s=0.5` and `K=2`, this is `1/3`.
The base coefficients remain signed and learnable: neither the total
coefficient magnitudes nor the feature norms are bounded by this statement.
The option requires `K>=1`, at least two features per head, identity
coefficient activation, learnable coefficients and positive finite `s`.

For the value-only polynomial preset, the output is
`sum_k diag(alpha_[:,k]) A^k V`. Its propagated features still use powers of
`A`, but token-dependent diagonal coefficients generally do not commute with
`A`. It is therefore not one scalar-coefficient polynomial in `A`; a global
spectral-filter interpretation or a theorem requiring shared coefficients
does not transfer automatically. The manuscript's page-3 allowance for
per-node/input-dependent coefficients motivates this explicit extension,
but does not specify this exact controller or prove an accuracy improvement.

Related work on adaptive propagation in
[DAGNN](https://arxiv.org/abs/2007.09296) and node-wise channel mixing in
[ACM](https://arxiv.org/abs/2210.07606) motivates adapting neighborhood use.
These are conceptual references, not EEG accuracy evidence or claims that
this implementation reproduces either method. The
[fixed EEGNet experiment](eegnet_agfl_token_routing.md) retains the previous
backbone, preprocessing and training recipe and measures static/token-routed
AGFL against MHA and Linformer rank 4.

## Comparable attention baselines

All five attention mechanisms are injected into the same chosen backbone for
a controlled comparison. Models own separate `eeg.py` and `ecg.py` variants;
mechanisms own mathematical settings under `attention_options`. EEG defaults to
`[B,electrode,feature]` tokens; ECG mixes ordered time tokens, whose downsampling
depends on the backbone. Only Signal Transformer uses strided time patches,
padding a final partial patch. Model-specific residuals, normalization, dropout
and classifier stay fixed when swapping attention. Input examples are fixed-length
`[B,C,T]`; arbitrary padded variable-length sequences are not silently accepted.
See [architecture details](model_attention_structure.md).

The common network's initialization stream is isolated from each mixer's
parameter initialization. This keeps common weights identical across mechanisms
for matching seeds/settings. Parameter and trainable-parameter counts are
recorded; equal capacity is not assumed. Attention probability dropout is not
added to individual mechanisms; the chosen backbone retains its own surrounding
dropout policy for all attention choices.

- **Vanilla multi-head attention:** learned Q/K/V, scaled dot products, row
  softmax, value aggregation, and output projection. Its numerical test uses
  PyTorch's multi-head attention as the reference.
- **Performer:** positive orthogonal random features (FAVOR+), fixed as checkpoint
  buffers. Query stability shifts are per query; key shifts are shared across
  all keys so normalization does not change their relative weights. Contraction
  avoids constructing the token-by-token matrix. This follows the mechanism in
  [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794).
- **Linformer:** separate learned sequence projections E and F per head, followed
  by attention to projected keys/values. Rank defaults to 8, capped at token
  count. The fixed-length requirement is explicit. See
  [Linformer](https://arxiv.org/abs/2006.04768).
- **Nyströmformer:** mean segment landmarks and the three-factor approximation
  `softmax(Q K_land^T) pinv(softmax(Q_land K_land^T)) softmax(Q_land K^T) V`.
  Unequal final segments are averaged over actual tokens. A small landmark
  Moore–Penrose inverse uses at least float32 with configurable `pinv_rtol`;
  kernel contractions also stay outside float16 autocast. The full-landmark case evaluates exact
  attention directly. No extra baseline-only convolution is introduced. This
  implements the mixing mechanism, not the complete original BERT model. See
  [Nyströmformer](https://arxiv.org/abs/2102.03902).

EEG Nyström segment landmarks depend on the declared electrode ordering; they
are not a geometric neighborhood model. Longformer is omitted: a local index
window is not an appropriate general electrode neighborhood. ECG temporal
length varies by model (Conformer retains 256 samples by default). A future temporal sparse baseline should be
added as its own explicit mechanism, with the same experimental controls.
