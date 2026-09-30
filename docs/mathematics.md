# AGFL definitions and attention mechanisms

## Required node semantics for current experiments

Since 28 September 2026, EEG graph nodes are electrodes in every backbone:
`X` has shape `[B,C,D]`, and `A` has shape `[B,H,C,C]`. The graph is constructed
before spatial aggregation in its attention branch. EEGNet uses a shared per-electrode temporal encoder
and a compact projection to D=32. Its temporal bins are features of each node,
not nodes. AGFL's propagation and coefficient equations below are unchanged;
token-conditioned coefficients now index electrodes. The retained K=2 recipe
mixes V, AV and A²V per electrode before its optional output gate.

EEGNet's `spatial_fusion` backbone additionally computes conventional depthwise
spatial filters on the shared temporal stem, before ELU and pooling. Its
classifier concatenates those `f2 * steps` features with the `electrode_dim`
attention readout. This backbone change preserves AGFL's equations and graph
axis, and is applied equally to every chosen attention mechanism.

ECG nodes remain time/time patches. Temporal convolution within an EEG electrode
is feature extraction and does not violate this separation. Temporal-distance
score bias is rejected for EEG because channel-list indices are not distances.
Old EEG temporal checkpoints retain their saved axes for reconstruction only.
See [architecture, launches and plots](eeg_interchannel.md). Routing graphs are
sensor-level model associations and do not establish anatomical connectivity.

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

The completed token-routing experiment selects `"token_contrast"`. With
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

### Token routing with local energy inputs

Historical definition: this extension was removed from the active code after
its unsuccessful study. Its [reproduction patch](archive/eegnet_energy_reproduction.patch)
preserves the implementation; retained token routing above is unchanged.

The opt-in `coefficient_conditioning="token_energy"` supplements the token
descriptor `d_i` above with two scalar magnitude inputs per trial/head:

```text
E_i = mean_features(V_i^2)
F_i = mean_features(P_1[i]^2)
C_i = mean_features((P_1[i] - V_i)^2) / (2*(E_i + F_i) + epsilon)
q_i = [tanh(log(E_i+epsilon) - mean_tokens(log(E+epsilon))),
       C_i - mean_tokens(C)]
r_i = tanh(G d_i + B q_i)
delta_i = (s/2) * (r_i - mean_hops(r_i))
```

Here `epsilon=1e-6`; all token means are within a trial/head. The inequality
`||x-y||^2 <= 2*(||x||^2+||y||^2)` bounds `C` in `[0,1]`, so the centered
contrast input lies in `[-1,1]`. The energy input is bounded by tanh but is
not necessarily zero-mean after tanh. Both describe latent value magnitudes,
not physiological band-power estimates. Calculations promote FP16/BF16 to
FP32 before subtraction/squaring and preserve FP64.

`B` has shape `[K+1,2]`, no bias, and zero initialization without random draws.
At `B=0` the original token controller is recovered for any `G`. This adds
`4*3*2=24` weights in the fixed EEGNet study. The propagated features, graph,
base coefficients, zero-sum hop correction and `s*K/(K+1)` per-hop bound are
unchanged. No cross-trial statistics or labels enter the controller.

This remains a token-dependent filter `sum_k diag(alpha_[:,k]) P_k`, with
the same limits on scalar-polynomial spectral claims stated above. It is an
explicit controller extension, not a formula or accuracy guarantee supplied
by the manuscript. The [implementation and comparison guide](eegnet_agfl_token_energy.md)
describes its ordinary-AdamW comparison against retained token AGFL, MHA and
Linformer. The completed A03 study measured 84.81% for this extension versus
85.19% for retained token routing; see the [results review](eegnet_a03_energy_review.md).

### Feature-wise token routing

The opt-in `coefficient_conditioning="feature_contrast"` retains the same
descriptor, graph, propagated hops, scale and base coefficients, but gives
each token and head-local output feature a separate hop mixture:

```text
r_i = reshape(tanh(G d_i), [d_h, K+1])
delta_i[f,k] = (s/2) * (r_i[f,k] - mean_hops(r_i[f,:]))
alpha_i[f,k] = alpha_base[k] + delta_i[f,k]
H_i[f] = sum_{k=0..K} alpha_i[f,k] * Linear_k(P_k[i])[f]
```

`G` is bias-free with shape `[d_h*(K+1), 2*d_h]`. Its flattened output uses
feature-major, then hop-major order. It starts at zero without random draws.
The original token router is contained in this family: repeat its gate rows
once per feature. That is a representational property, not a guarantee that
optimization finds an equal or better solution. The actual preset starts
from zero, with fresh training for every arm, rather than importing a selected
checkpoint.

The zero-sum constraint and `s*K/(K+1)` correction bound now hold separately
for every token and feature. With four heads, eight features/head and `K=2`,
the router has 1,536 weights versus the token router's 192. These are learned
feature coordinates, not EEG channels or class-specific routes. No new hops,
graphs, affine normalization or training-data access are introduced.

For value-only polynomial hops the output is `sum_k C_k elementwise_mul (A^k V)`,
where `C_k` has token and feature axes. This is an explicit extension beyond
the manuscript's scalar coefficients. Shared-coefficient spectral statements
cannot be reused without new assumptions/proofs. Related
[AdaGNN](https://arxiv.org/abs/2104.12840) motivates allowing different feature
channels to receive different filtering; this implementation is distinct.
The historical temporal-graph A03 comparison measured 84.07% for feature routing versus
85.19% for token routing, so added feature flexibility did not improve that
experiment. See the [results review](eegnet_a03_feature_review.md) and
[five-arm comparison and commands](eegnet_agfl_feature_routing.md).

## Comparable attention baselines

### Matched MHA output-gate control

The current inter-channel EEG study uses the same `TokenOutputGate` class in
MHA and augmented AGFL. Each gate acts on an electrode message; `B_h=0` and
no temporal-bias parameters are constructed for EEG. The optional
`RelativeTemporalBias` below applies only to ECG time tokens and historical
EEG time-token checkpoint replay. For each head, with temporal relative offset
`r_ij = (j-i)/max(N-1,1)` where applicable:

```text
B_h[i,j] = 0                                             # EEG
B_h[i,j] = u_h * abs(r_ij) + v_h * r_ij + w_h * 1[i=j]    # optional temporal bias
A_h = softmax_j(Q_h K_h^T / sqrt(d_h) + B_h)
P_1 = A_h V_h
d_i = concat(LN_features(V_h[i]), LN_features(P_1[i] - V_h[i]))
g_h[i] = 2 * sigmoid(W_gate,h d_i + b_gate,h)
Z = concat_h(g_h * P_1) W_out + b_out
```

Layer normalization is affine-free with epsilon `1e-5`. Each head has its
own scalar token gate. Both added modules initialize to zero weights/biases,
giving `B=0` and `g=1`, without consuming RNG draws. Gate multiplication
precedes the output projection and leaves the backbone's outer residual
unchanged. With four heads and head dimension eight, either mechanism gains
the same 68 gate weights/biases in EEG. Enabling temporal bias on time tokens
adds another 12 weights, giving 80 for the combined temporal variant.
Plain MHA retains its previous parameter names and formula when both flags
are false. Temporal bias requires ordered time tokens.

AGFL retains its sparse graph, token-dependent polynomial coefficients and
multi-hop mixture; its output gate multiplies that mixture, while MHA's gate
multiplies the single-hop message. MHA receives neither AGFL's sparsification
nor its hop router. These differences are explicit parts of the comparison.
For the temporal variant, the signed lag bias is softmax-equivalent to a source-position ramp, as in
the AGFL version; it does not create an independently expressive directional
interaction. No causal mask is introduced in either mechanism.

### Established mechanisms

All five attention mechanisms are injected into the same chosen backbone for
a controlled comparison. Models own separate `eeg.py` and `ecg.py` variants;
mechanisms own mathematical settings under `attention_options`. EEG requires
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
