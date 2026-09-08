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

## Reconciliation with the supplied manuscript

Inspected document: `NEU_article_submission.pdf`, Section 2.1, page 3, equations
1–8. The manuscript and active code differ in several ways:

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
Unnormalized signed operators can amplify high powers; nonfinite losses or
unscaled gradients stop a run instead of producing publishable metrics.

The historical dense similarity construction and dense propagation still have
quadratic token dependence before/after masking. No sparse-complexity claim is
made. `K=0` is a polynomial-order ablation, not a separately registered
no-attention network. The no-attention model key has been removed as requested.

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
