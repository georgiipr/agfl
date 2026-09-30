> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# After SAM: restored baseline and next AGFL investigation

Historical investigation record. The proposed energy candidate was later
completed and rolled back on 25 September; the active version is again the
[retained token router](eegnet_token_restore.md).

24 September 2026. Scope: original EEGNet, four-class BCI IV 2a A03T,
individual-subject training, five seeds. No project code or tests ran
locally. Research used source inspection, existing saved validation
diagnostics and primary papers. The candidate proposed below has now been
implemented as opt-in `token_energy`. It was unmeasured at the time of this
investigation; the subsequent study reached 84.81% versus token routing's
85.19%. See the [results review](eegnet_a03_energy_review.md) and
[implementation and launch guide](eegnet_agfl_token_energy.md).

## Rollback completed

The retained recipe is original seven-token EEGNet with token-routing AGFL,
ordinary AdamW 0.005, focal gamma 3, 250 epochs, no augmentation/EMA and the
original first-best validation-accuracy rule. Its latest control reproduced
**85.19%** mean test and **91.48%** validation accuracy.

The rollback restores the original training/configuration and result-plot
files, removes SAM-only additions from analysis while retaining feature-router
analysis that preceded SAM, and removes `agfl/sam.py`, the SAM preset and
three SAM test files. `training.sam_rho` is no longer an accepted option.
The token router, original backbone and diagnostic JSON/report recovery fixes
are untouched. No saved results, checkpoints, datasets or splits were deleted.

The retired SAM changes are preserved as
[`archive/eegnet_sam_reproduction.patch`](archive/eegnet_sam_reproduction.patch),
an ordinary text patch outside package, preset and test discovery. It applies
to the rollback tree before energy routing; static `git apply --check`
verified applicability at rollback time. It does not target the later
energy-routing edits.
It preserves the failed experiment for reproduction in a separate copy,
not as an active dependency. Downloaded SAM reports already contain the
correct labels, paired comparisons and figures. Rebuilding SAM-specific
reports requires its archived reporting code; the restored generic analyzer
does not retain the SAM-specific labels or training-ablation exception.

## What the earlier failures actually tell us

| Experiment | Observed outcome | Consequence for the next change |
|---|---|---|
| Learned temperature/local-value initialization | Validation-selected procedure: 85.56% test; no new fixed candidate beat sparse control's mean validation | Not evidence that another temperature setting will solve the problem |
| 15 tokens plus per-hop feature projections | Selected procedure: 84.44%; every new fixed candidate had lower mean validation than the seven-token control | Preserve backbone and token count; the combined study does not isolate either change |
| Trial-wide power controller | 84.44%, tied with static AGFL; feet/tongue tradeoff | A single trial-wide correction did not resolve local decisions |
| Power branch plus EMA | AGFL 77.04% | Preserve the original feature extractor; do not attribute the entire failure to EMA or power features individually |
| Token routing | 85.19% vs static 84.44%; validation 91.48% vs 92.22% | Best retained fixed AGFL test mean, but its small gain is not a confirmed generalization result |
| Feature-wise token routing | 84.07%, below token routing on validation too | Do not repeat a large increase in router parameters without a distinct information source |
| SAM rho 0.05 | 81.11%; validation 89.63%; feet recall 83.08%→66.15% | Remove this global training change; it did not solve the training/validation gap |

Sources: [refinement](eegnet_agfl_refinement.md),
[capacity review](eegnet_a03_capacity_review.html),
[trial-wide conditioning](eegnet_agfl_conditioning.md),
[power/EMA record](eegnet_power_ema.md),
[token review](eegnet_a03_token_review.md),
[feature review](eegnet_a03_feature_review.md), and
[SAM review](eegnet_a03_sam_review.md).

These studies mix fixed recipes and validation-selected procedures; their
headline scores are not all estimates of the same method. In particular,
85.56% from candidate selection is not the score of one demonstrated better
fixed AGFL configuration. Repeated test inspection also makes A03 a
development subject, not independent confirmation for the article.

## Concrete limitation in the retained token router

The current router sees, for each token i and attention head:

```text
d_i = concat(LayerNorm(V_i), LayerNorm((A V)_i - V_i))
delta_i = (scale/2) * center_hops(tanh(G d_i))
```

It sees normalized feature patterns and normalized neighborhood contrast.
LayerNorm subtracts the feature mean and divides by feature standard
deviation. Consequently, above the epsilon floor, positive rescaling of
either descriptor leaves it approximately unchanged. The controller does
not explicitly receive the token's magnitude or the strength of its
contrast. This follows from the code and the normalization definition;
it is not a diagnosed explanation for the classification errors.
See the [Layer Normalization paper](https://arxiv.org/abs/1607.06450).

The information is not erased from the whole network: V and the propagated
values remain unnormalized in the polynomial output. This is specifically
a limitation of the information available for choosing hop coefficients.

## New measurements from saved validation diagnostics

I inspected only the ordinary token-AGFL validation diagnostic arrays in
`Downloads/report/diagnostics/eeg/subject_A03/eegnet-agfl-bfb47eb1907c65099022/`.
No test labels or fitted correction were used for this investigation.

The saved arrays contain coefficients alpha, norms of each weighted hop
contribution and pairwise hop cosines. Because this recipe has no per-hop
feature projection, `norm(P_k) = norm(alpha_k P_k) / abs(alpha_k)` recovers
hop magnitudes when the coefficient is safely nonzero. I excluded cases
with either required coefficient at or below 1e-5 or invalid hop norms;
none of the 7,560 token/head observations required exclusion.

Across 270 repeated validation trials × four heads = 1,080 trial/head groups:

| Measurement | 10th percentile | Median | 90th percentile |
|---|---:|---:|---:|
| Largest/smallest token squared norm within a trial/head | 3.68× | **11.67×** | 42.56× |
| Contrast strength `norm(P1−V)^2 / (2*(norm(V)^2+norm(P1)^2)+epsilon)` | 0.0271 | **0.1679** | 0.5470 |
| Cosine between first and second propagated hops | 0.8022 | **0.9920** | 0.99996 |

The contrast numerator was reconstructed using the saved norms and cosine,
not by running a checkpoint. These are learned latent-feature magnitudes,
not physiological EEG frequency-band power. The observations overlap across
seeds and are descriptive. They establish variation in a potential input,
not that the input predicts labels or will improve generalization.

High P1/P2 similarity makes simply adding more polynomial hops a lower
priority. It does not prove harmful oversmoothing: signed coefficients and
the zero-hop term already permit contrastive filters. Rewriting the same
degree-two polynomial in another basis alone does not add expressive power.

## Primary-source research and priorities

- [FBCNet](https://arxiv.org/abs/2104.01233) uses spatial filtering and a
  variance layer for motor-imagery decoding. This supports investigating
  magnitude-related information in MI representations. It does not validate
  the proposed AGFL descriptor or make its reported cross-subject aggregate
  comparable with our single-subject pilot. Replacing EEGNet with FBCNet is
  outside this proposed change.
- [GPR-GNN](https://arxiv.org/abs/2006.07988) learns propagation weights to
  adapt graph filtering. That principle is already present in AGFL's signed
  trainable taps. It is not evidence that more taps or a basis relabeling
  would fix our errors; its node-classification setting is different.
- [Gated attention](https://arxiv.org/abs/2505.06708) reports improvements
  from input-dependent output gates in large language models. That offers
  a separate later hypothesis, but the scale and task differ markedly. I
  would not combine an output gate with the next controller change or claim
  an EEG benefit from those language-model results.

## Recommended next candidate: give the token router local magnitude

Keep the existing descriptor and add two bounded scalar inputs per token/head:

```text
E_i = mean_features(V_i^2)
F_i = mean_features(P1_i^2)
C_i = mean_features((P1_i - V_i)^2) / (2*(E_i + F_i) + epsilon)
r_i = [tanh(log(E_i+epsilon) - mean_tokens(log(E+epsilon))),
       C_i - mean_tokens(C)]

u_i = G d_i + B r_i
delta_i = (scale/2) * (tanh(u_i) - mean_hops(tanh(u_i)))
H_i = sum_k (alpha_base[k] + delta_i[k]) * P_k[i]
```

Use the same value-only polynomial, graph, scale 0.5, zero-sum correction,
Q/K/V projections and ordinary training. B is a separate bias-free,
zero-initialized map of shape `[3,2]` for each of four heads: **24 additional
weights**, taking token AGFL from 8,244 to **8,268** parameters. Existing G
keeps its initialization and RNG draws. At B=0 the model contains the
current token-router computation exactly; the old default and checkpoint
layout must stay unchanged unless the new mode is explicitly selected.

This is different from the failed trial-power controller: its inputs vary
across tokens within the trial, and it supplements the retained successful
token descriptor. It is also different from feature-wise routing: it adds
two kinds of information rather than separate hop mixtures for every
feature. It adds no filter bank, power branch, normalization across trials,
new optimizer, class-specific thresholds or labels at inference.

This is the most targeted candidate identified here, not a promised +1 pp.
It was **implemented but not measured** when proposed. The later
[completed study](eegnet_a03_energy_review.md) found no validation or test
improvement. The observation that magnitudes vary does not establish that
this extra freedom helps rather than overfits.

## One controlled next experiment

Use four fixed configurations: retained token AGFL, token AGFL with the two
extra magnitude inputs, MHA, and Linformer rank 4. Five seeds each gives
20 fits in one sequential sweep, with automatic diagnostics and plots.
Freeze EEGNet, preprocessing, train/validation/test IDs, loss, optimizer,
learning rate, epoch budget and checkpoint rule. Report the extra 24
parameters. No radius search or multiple simultaneous mechanism changes.

Checkpoint selection remains validation-only. Test all declared arms once
after selection and report every seed, class tradeoff and negative result.
Judge validation and test together; do not count recovering feet by losing
tongue as a general improvement. Independent subjects/sessions are needed
after the design is frozen. The [launch guide](eegnet_agfl_token_energy.md)
contains the four-arm sweep and separate checkpoint/plot commands.

## Upload the rollback to the cluster

The instructions below record the rollback-only update. For the current
energy-routing experiment use the [complete upload list](eegnet_agfl_token_energy.md#upload-ordinary-files),
which includes the rollback files as well as the new attention and diagnostics.
Run on the Mac after the active study has ended:

```bash
scp /Users/egor/Downloads/AGFL/agfl/config.py \
  /Users/egor/Downloads/AGFL/agfl/engine.py \
  /Users/egor/Downloads/AGFL/agfl/analysis.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp /Users/egor/Downloads/AGFL/agfl/visualization/results.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
```

Then on the cluster remove only the retired SAM source/preset/test files:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
rm -f agfl/sam.py agfl/presets/eegnet-a03-token-sam.json \
  tests/test_sam.py tests/test_training_sam.py tests/test_eegnet_sam_comparison.py
```

No dependency changes, checkpoint conversion or training run is required.
Keep the existing SAM session directory intact. The assistant has not
uploaded files or modified the cluster. For optional reproduction of the
retained token comparison, the [README](../README.md#improve-eegnet--agfl-on-bci-iv-2a)
contains separate training, checkpoint-diagnostic and result-plot commands.
