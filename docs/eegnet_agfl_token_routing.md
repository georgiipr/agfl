> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# EEGNet/A03: improve AGFL while restoring the previous model

**Completed:** revised AGFL averaged **85.19%**, original AGFL **84.44%**,
MHA **85.56%**, and Linformer **85.93%**. All 20 fits finished. The subsequent
diagnostic JSON serialization error is fixed; use the
[saved-results review and recovery commands](eegnet_a03_token_review.md)
to regenerate reports without training again. Commands below preserve the
full experiment for reproduction.

The completed preset is **`eegnet-a03-token-agfl`**. The subsequent
[feature-wise experiment](eegnet_a03_feature_review.md) fell to 84.07%;
token routing remains the best observed AGFL test result at 85.19%.
The subsequent [SAM experiment](eegnet_a03_sam_review.md) fell to 81.11%
and has been rolled back. The retained configuration again uses ordinary
AdamW. The later energy-input candidate reached 84.81% and has also been
rolled back. See the [energy results](eegnet_a03_energy_review.md),
[restoration/upload guide](eegnet_token_restore.md) and
[failure-based research rationale](eegnet_after_sam.md).
This completed study restores the original
seven-token EEGNet and its previous training recipe, then tests one change
inside AGFL: **a separate hop mixture for each token, informed by its graph
neighborhood**. The original static AGFL remains a measured control.

One command runs **20 fits: four fixed configurations × seeds 0–4**, each
for 250 epochs on subject A03. Training, checkpoint diagnostics and result
plots run in sequence. The existing Python environment is sufficient.

## Why the failed recipe was withdrawn

The completed power/EMA experiment produced these mean test accuracies:

| Attention | Power/EMA recipe | Previous original-EEGNet recipe |
|---|---:|---:|
| Static AGFL | 77.04% | 84.44% |
| MHA | 76.30% | 85.56% |
| Linformer rank 4 | 76.67% | 85.93% |

Its small AGFL lead did not compensate for the absolute regression. The
power branch and EMA were active, predictions replayed correctly, and all
15 fits and 346 figure pairs completed. The combined experiment cannot
identify which architecture/training change caused the deterioration.
See the [retired power/EMA record](eegnet_power_ema.md).

The restored active model uses the original convolution path, 32-sample
first temporal kernel, pooling factors 8 and 16, 32 features and seven
tokens. There is no power branch in this model. Archived configurations
explicitly requesting `mean_logvar` are routed to isolated reproduction
classes in `agfl/models/eegnet/reproduction.py`, preserving their checkpoint
keys and equations for diagnostics. Saved results are not reinterpreted.

Training returns to AdamW at **0.005**, balanced focal loss with **gamma 3**,
weight decay 0.001, batch 64 and the original warmup/cosine schedule. EMA is
off (`ema_decay=0`), augmentation is off, and checkpoint ties use the original
first-best rule (`checkpoint_tiebreaker=none`). Preprocessing and splits are
unchanged. These controls make the new comparison an attention change within
the prior model and training recipe.

## The AGFL change

Static AGFL uses one signed coefficient for each head and hop, shared across
all tokens. The completed trial-power controller used one correction for
the entire trial; it traded feet and tongue errors without improving mean
accuracy. The new controller uses each token's own feature vector and its
contrast with the first graph hop. It can therefore alter the local versus
neighbor mixture at different time positions within the same trial.

For one head, token `i`, value matrix `V`, and configured propagated features
`P_k`, the new `coefficient_conditioning="token_contrast"` computes:

```text
P_0 = V
P_k = A @ P_(k-1)                      # polynomial preset used here
d_i = concat(LayerNorm(V_i), LayerNorm(P_1[i] - V_i))
r_i = tanh(G @ d_i)
delta_i = (scale / 2) * (r_i - mean_hops(r_i))
alpha_i = alpha_base + delta_i
H_i = sum_{k=0..K} alpha_i[k] * P_k[i]
```

The LayerNorms operate over a head's features with epsilon `1e-5` and have
no learned affine parameters. Descriptor and controller arithmetic use FP32
outside autocast, retaining FP64 for numerical-reference calculations. No
labels, class-specific rules, batch statistics or cached activations enter
the controller. The graph and propagated features are those already used by
the selected AGFL formula; the controller adds no graph construction or hop.
With the optional renormalized formula, its descriptor uses that formula's
actual first hop rather than silently substituting `A @ V`.

Every head has a bias-free matrix `G` with shape `[K+1, 2*d_head]`. It starts
at zero without consuming random draws, so initial outputs and existing
parameter initialization match static AGFL. With four heads, eight features
per head and `K=2`, the controller adds **192 trainable parameters**.

The correction sums to zero across hops, up to floating-point roundoff.
Its component bound is `|delta_i[k]| <= scale*K/(K+1)`, or **1/3** for this
study's scale 0.5 and `K=2`. This redistributes the sum of tap weights; it
does not constrain the learned base coefficients, force them positive or
guarantee preservation of feature norms. A learned advantage remains to be
measured.

With token-dependent coefficients, the value-only output is
`sum_k diag(alpha_[:,k]) A^k V`. The hops still use polynomial propagation,
but their token-specific multipliers do not form one scalar polynomial in
`A`. Static spectral-filter claims require separate justification for this
variant; see the [mathematical definition](mathematics.md#explicit-input-dependent-hop-coefficients).

```json
{
  "attention_options": {
    "coefficient_conditioning": "token_contrast",
    "coefficient_conditioning_scale": 0.5,
    "coefficient_activation": "identity",
    "learnable_coefficients": true,
    "K": 2
  }
}
```

Token routing requires `K >= 1`, at least two features per head, identity
coefficient activation, learnable coefficients and a positive finite scale.
The global default remains `static`. Existing trial-power configurations
retain their original computation.

The final submission `Downloads/NEU_art_submission.pdf`, page 3, permits
per-node and input-dependent coefficient parameterizations. This exact
zero-sum contrast controller is a **new experimental implementation**, not a
claim that the manuscript previously specified it. Related graph work studies
adaptive propagation depth in [DAGNN](https://arxiv.org/abs/2007.09296) and
node-wise aggregation/diversification/identity mixing in
[Adaptive Channel Mixing](https://arxiv.org/abs/2210.07606). These motivate
the direction; this implementation reproduces neither method and those
papers do not establish an EEG accuracy gain for it.

## Fixed comparison and interpretation

| Configuration | Role | Expected parameter count |
|---|---|---:|
| Static sparse Q/K/V AGFL | Previous strongest AGFL recipe | 8,052 |
| Token-routed sparse Q/K/V AGFL | New attention-only change | 8,244 |
| MHA | Base attention | 8,036 |
| Linformer rank 4 | Strongest observed previous baseline | 8,260 |

Counts are derived from the saved reference counts and the new 192-weight
controller; actual counts are recorded by training. Both AGFL arms retain
`K=2`, polynomial propagation, one-hop initialization `[0,1,0]`, Q/K/V
projections, square-root score scaling, value-only hops, softmax graph
normalization and the same scheduled Top-k settings. They differ only in
`coefficient_conditioning`.

All methods use `../ml/A03T.gdf`, four classes, artifact exclusion, 22 EEG
channels, cue-relative 0–4 seconds, 2–30 Hz trial-local filtering and
training-only channel normalization. Each seed retains its matching
162/54/54 train/validation/test split. Validation chooses one checkpoint for
each fixed configuration; every selected checkpoint receives a test result.
There is no candidate-selection search or no-attention arm.

Assess the mean across **all five seeds**, the token-minus-static difference
and both attention baselines. Report feet, tongue and both hand recalls so
one class's gain cannot hide another's loss. A03 and these overlapping splits
have repeatedly informed development, so this remains exploratory evidence.
**90% and an AGFL advantage are goals, not established outcomes.** This study
does not measure robustness, runtime improvements or general superiority.

## Upload ordinary source files

The updated files are already in `/Users/egor/Downloads/AGFL`. Copy the
source package, entry point and documentation directly into the cluster
checkout. This includes the restored `agfl/models/eegnet/backbone.py` and
the reproduction module required by the updated EEGNet factory. From your Mac:

```bash
scp -r /Users/egor/Downloads/AGFL/agfl \
  /Users/egor/Downloads/AGFL/main.py \
  /Users/egor/Downloads/AGFL/README.md \
  /Users/egor/Downloads/AGFL/docs \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

The files are ready to use after copying; no archive extraction is needed.
The selected paths do not include the dataset or results directories.
Do not replace source files while an experiment is running. The assistant
has not transferred files to the cluster.

## Launch once on the cluster

Obtain an interactive GPU shell unless already allocated:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

After allocation succeeds, use the existing environment. Working directory:
`/beegfs/home/georgii.promyslov/AGFL`. Dataset: `../ml/A03T.gdf`.
Fresh output folder: `results/eegnet-A03-token-agfl-v1`.

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg

python main.py sweep --preset eegnet-a03-token-agfl \
  --output-dir results/eegnet-A03-token-agfl-v1 \
  --report
```

This runs the 20 fits sequentially with progress bars, followed by validation
checkpoint diagnostics and result plots. If checkpoint diagnostics raise an
error, automatic reporting still attempts result plots from saved predictions
before propagating the diagnostic error. A user interruption exits immediately.
The four-hour allocation is a
resource request, not a completion-time estimate. No installation, `.sbatch`
file or development test run is required.

After a training interruption, add `--skip-completed` to that command to retain
matching completed fits. Preserve the same source, configuration, data and
environment; reuse checks include provenance. Incomplete fits restart from
epoch 1. Without that flag, matching existing fits are rerun.

If all fits completed and only plotting failed, run the two separate commands
below instead of relaunching the sweep. A plotting-only source update does
not require retraining; training reuse checks intentionally treat source
changes as a new comparison.

Separate checkpoint-diagnostic command, if automatic reporting needs recovery:

```bash
python main.py diagnose-session results/eegnet-A03-token-agfl-v1 \
  --device cuda --partition validation --embedding tsne
```

Separate result-analysis and plot command:

```bash
python main.py analyze results/eegnet-A03-token-agfl-v1 --plots
```

Download **`results/eegnet-A03-token-agfl-v1/report/`** into a new local folder,
such as `Downloads/eegnet-A03-token-agfl-v1-report/`. Keep `artifacts/` and
checkpoints on the cluster for regenerating checkpoint diagnostics.

## Plots and files to inspect

- `analysis/report.md`: accuracy/AUC/F1 summaries and per-class recall.
- `analysis/figures/index.md`: all saved result plots, including class recall,
  confusion matrices, training histories and attention comparisons.
- `analysis/per_class.csv` and `analysis/per_class_summary.csv`: validation
  and test recall, precision, F1 and support, with per-seed coverage.
- `analysis/statistical_comparisons.csv` and `analysis/aggregation.json`:
  paired token-routed-minus-static and AGFL-minus-baseline differences on
  matched seeds/splits. Overlapping seeded splits do not provide independent
  subjects.
- `diagnostics/eeg/subject_A03/<model-attention-id>/seed_*/index.md`:
  checkpoint signal, embedding, graph and coefficient plots.
- Token-routed runs' `coefficient_conditioning.json` (schema 2) and
  `filters/<layer>_token_coefficients.npz`: effective coefficients and deltas
  with axes `[trial, head, token, hop]`, base coefficients `[head, hop]`, gate
  weights `[head, hop, descriptor_feature]`, and explicit sample IDs, labels
  and axis indices. `within_trial_token_std` retains trial/head/hop axes;
  `delta_sum_over_hops` exposes the zero-sum residual for each trial/head/token.
- The same NPZ includes `propagated_hop_cosines` and validity masks with axes
  `[trial, head, token, hop_left, hop_right]`, plus `hop_contribution_norms`
  `[trial, head, token, hop]`. These show whether propagated hops are nearly
  identical and record their weighted contribution magnitudes before the
  output projection/residual; they are not causal importance estimates.
- `filters/<layer>_token_coefficients_head_<head>.png/.pdf`: each hop's
  effective coefficient and correction as trial-by-token heatmaps. Corrections
  divide by their attainable bound (1/3 here); saturation means exceeding
  95% of that bound. Token variation is retained rather than averaged away.
  Controller activity alone is not evidence of an accuracy benefit.
- `runs/.../config.json`, `split.json`, `history.json` and `result.json`:
  exact settings, selection policy, provenance and all completed outcomes.

Static AGFL and both baselines remain visible even if the new variant loses.
The restored EEGNet has no temporal-statistics feature plots because it has
no power branch. There is no search report in this fixed four-arm comparison.

## Verification status

Implementation, initialization/checkpoint compatibility, formulas, diagnostic
axes and comparison settings are reviewed statically. Development tests cover
the new controller and rollback separately from the launch workflow.
**No project code, training, inference or tests were run locally.** Numerical
behavior, GPU execution and accuracy await the cluster results.
