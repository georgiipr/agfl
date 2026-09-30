> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# EEGNet/A03: local energy inputs for token-routing AGFL

**Archived after rollback on 25 September.** The energy layer, preset and
three development-test files are removed from the active code. The
[plain-text reproduction patch](archive/eegnet_energy_reproduction.patch)
restores the exact completed energy source on the rollback tree in a separate
copy; its applicability was statically checked. Commands below describe that
archived experiment and are not active launch instructions. Use the
[retained-token workflow](eegnet_token_restore.md) for the restored code.

Implemented on 24 September 2026; results reviewed on 25 September.
**Completed: energy AGFL reached 84.81%, below original token AGFL's 85.19%;
validation also declined.** All 20 fits and 511 PNG/PDF figure pairs completed.
The [results review](eegnet_a03_energy_review.md) recommends retaining original
token AGFL. Commands below preserve reproduction of this unsuccessful ablation.
The new preset is **`eegnet-a03-token-energy`**. It retains the original
seven-token EEGNet, token router and ordinary AdamW, and adds two magnitude
inputs to the router. No libraries, optimizer changes or checkpoint conversion
are required. The [research record](eegnet_after_sam.md) explains the choice
using the unsuccessful earlier experiments and saved validation diagnostics.

## What changes inside AGFL

The existing router sees normalized token features and normalized graph
contrast. The new explicit mode, `coefficient_conditioning="token_energy"`,
also receives relative token energy and relative contrast strength. For each
trial and head, let `P0=V`, `P1=A@V`, and `epsilon=1e-6`:

```text
d_i = concat(LayerNorm(V_i), LayerNorm(P1_i - V_i))
E_i = mean_features(V_i^2)
F_i = mean_features(P1_i^2)
C_i = mean_features((P1_i - V_i)^2) / (2*(E_i + F_i) + epsilon)
r_i = [tanh(log(E_i+epsilon) - mean_tokens(log(E+epsilon))),
       C_i - mean_tokens(C)]
u_i = tanh(G d_i + B r_i)
delta_i = (scale/2) * (u_i - mean_hops(u_i))
H_i = sum_k (alpha_base[k] + delta_i[k]) * P_k[i]
```

All reductions are within one trial/head. There are no labels, cross-trial
statistics or learned affine normalization in these descriptors. Both new
inputs lie in `[-1,1]`; contrast is centered across tokens, whereas log energy
is centered **before** its tanh and need not have zero mean afterward.
Magnitudes describe learned value features, not physiological EEG band power.
Squaring and subtraction use FP32 under autocast; FP64 is preserved when used.

`B` is a separate bias-free map with shape `[3,2]` per head: **24 additional
trainable weights**, taking the configured model from 8,244 to **8,268 total
parameters**. It starts at zero without consuming random initialization draws.
At `B=0`, the old router's computation is retained, including a nonzero `G`.
Each sweep arm is nevertheless trained from scratch; no selected checkpoint
is transplanted. The graph, Top-k support, projections, two propagated hops,
scale 0.5, zero-sum coefficient correction and signed base taps remain fixed.
The correction bound is still `scale*K/(K+1) = 1/3` per hop for this preset.

The existing `token_contrast` option and its state keys remain supported.
Only `token_energy` creates `energy_gate.weight` entries in checkpoints.
Old runs retain their saved configuration and interpretation. New runs receive
distinct experiment identities and record source provenance.

This supplements the retained token router with a small information source.
It does not repeat the failed larger feature-wise router or trial-wide power
controller. An accuracy increase, including +1 percentage point, is a
hypothesis to measure, not a guaranteed consequence of extra inputs.

## One matched comparison

| Arm | Role | Total parameters |
|---|---|---:|
| Original token AGFL | Retained control | 8,244 |
| Token AGFL with energy inputs | New candidate | 8,268 |
| MHA | Base attention | 8,036 |
| Linformer rank 4 | Strongest previous attention mean | 8,260 |

One sequential sweep runs **20 fits: four arms × seeds 0–4**, 250 epochs each,
on A03T only. The dataset is `../ml/A03T.gdf` relative to the cluster `AGFL`
directory. The preset copies the previous backbone, preprocessing, per-seed
60/20/20 split policy, batch size 64, balanced focal loss with gamma 3, AdamW
0.005, weight decay 0.001 and warmup/cosine schedule. No augmentation, EMA,
SAM or early stopping is active. The first checkpoint attaining the best
validation accuracy is retained. Test data never select checkpoints.

The report pairs the new candidate with original token AGFL and each attention
baseline only on matching subjects, seeds, splits and comparison provenance.
The token ablation also requires equal conditioning scale and all other graph
settings. Report validation/test means, seed differences and class recall for
all declared arms, including unfavorable results. A03 has already been used
repeatedly for development; gains here require independent confirmation after
the method is frozen. Previous scores are context, not substituted controls.

## Upload ordinary files

Run on the Mac after any active study has finished. These files include the
pre-SAM engine/configuration rollback and the existing report-recovery fix, so
the upload also works if the cluster still contains the SAM iteration:

```bash
cd /Users/egor/Downloads/AGFL
scp agfl/config.py agfl/engine.py agfl/analysis.py agfl/cli.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp agfl/attention/agfl/layer.py agfl/attention/agfl/config.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/attention/agfl/
scp agfl/visualization/diagnostics.py agfl/visualization/results.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
scp agfl/presets/eegnet-a03-token-energy.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

On the cluster, remove only the retired SAM code/preset/tests. Keep previous
session outputs intact. The active configuration no longer accepts `sam_rho`.

```bash
cd /beegfs/home/georgii.promyslov/AGFL
rm -f agfl/sam.py agfl/presets/eegnet-a03-token-sam.json \
  tests/test_sam.py tests/test_training_sam.py tests/test_eegnet_sam_comparison.py
```

## Allocate resources and activate the existing environment

If no GPU allocation is active, request one on the login node:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=4:00:00
```

Wait for the allocation, then enter the compute node:

```bash
srun --pty bash -l
```

Inside the allocation (also use this block if already on a GPU node):

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source /beegfs/home/georgii.promyslov/.venv/bin/activate
export MPLBACKEND=Agg
```

## Train and generate the report

```bash
python main.py sweep --preset eegnet-a03-token-energy \
  --output-dir results/eegnet-A03-token-energy-v1 \
  --report
```

`--report` runs validation checkpoint diagnostics with t-SNE, then result
analysis with plots. If diagnostics fail, saved-result analysis is still
attempted before the error is reported. An error in plotting does not erase
completed fits. The separate commands below also regenerate plots without
training. They are optional after a successful automatic report.

To resume this exact study after interruption, add `--skip-completed` to the
training command only while source, configuration and environment remain
unchanged. Completed runs must pass provenance checks; incomplete runs restart
from epoch 1. Without that flag, matching output runs are overwritten and
retrained. Use a fresh output folder for another code iteration.

Checkpoint diagnostics, including the new router plots:

```bash
python main.py diagnose-session results/eegnet-A03-token-energy-v1 \
  --device cuda --partition validation --embedding tsne
```

Result plots and comparison tables:

```bash
python main.py analyze results/eegnet-A03-token-energy-v1 --plots
```

## What to download and inspect

Download only **`results/eegnet-A03-token-energy-v1/report/`** into a fresh local
folder. Keep `artifacts/` on the cluster; it contains checkpoints needed to
regenerate diagnostics.

- `analysis/report.md`: accuracy, F1 and AUC summaries for all four arms.
- `analysis/statistical_comparisons.csv`: matched new-versus-original token
  ablation and comparisons with MHA/Linformer.
- `analysis/per_class.csv` and `analysis/per_class_summary.csv`: validation/test
  class metrics, including the feet/tongue tradeoff seen in earlier failures.
- `analysis/figures/index.md`: result-plot index.
- Per-run `diagnostics/.../index.md`: checkpoint-plot index. Existing graph,
  embedding, filter and token-coefficient diagnostics remain available.
- For the new mode, `coefficient_conditioning.json` uses schema 4 and records
  energy feature definitions, gate norms and per-hop coefficient effects.
  The corresponding `filters/*_token_coefficients.npz` adds `energy_features`
  `[trial,head,token,2]`, `energy_gate_weights` `[head,hop,2]`, and
  `energy_gate_logits`, `coefficients_without_energy`,
  `energy_coefficient_effect` `[trial,head,token,hop]`. Sample IDs retain the
  diagnostic order. Original token-mode outputs keep schema 2.
- `filters/*_energy_inputs_head_*` and `filters/*_energy_effect_head_*` add two
  PNG/PDF figure pairs per head: eight additional pairs per new-candidate run.
  They show both new inputs, `B*r` logits and the resulting change in hop
  coefficients at the same saved graph/values. This is descriptive, not a
  retrained ablation, prediction difference or causal-importance score.

## Verification boundary

Local verification is limited to source review, Python AST/JSON parsing,
preset comparisons, shell syntax and whitespace checks. All 129 Python files
parsed; all preset JSON files parsed; 61 documentation shell blocks passed
`bash -n`; local documentation links and the nine upload paths were checked.
The training engine, top-level configuration, result-plot implementation,
original token preset and EEGNet backbone remain byte-identical to pre-SAM
HEAD. `git diff --check` passed. These are static checks only. New development
checks cover the formula, zero initialization/RNG compatibility, gradients,
within-trial behavior, AMP, diagnostic serialization/plots, experiment pairing
and checkpoint replay. They have **not** been executed locally and are not a
training prerequisite. No project code, tests, inference or training ran here.
The subsequent cluster study completed; its saved metrics, prediction replay
checks and diagnostics are examined in the [results review](eegnet_a03_energy_review.md).
