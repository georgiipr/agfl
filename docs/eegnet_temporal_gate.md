> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# EEGNet/A03: temporal bias, output gating, and their combination

26 September 2026. This is an unmeasured, opt-in experiment. The retained
token-routing AGFL remains the control (85.19% in the previous completed
study). No local training, inference or project tests were run for this update.

## One launch, four matched configurations

Preset: `eegnet-a03-temporal-gate`. It runs the following configurations
sequentially, each with seeds **0–4**, **250 epochs**, and A03 alone:

| Arm | temporal_bias | output_gate | Expected total parameters |
|---|---|---|---:|
| Retained token AGFL | false | false | 8,244 |
| Temporal bias | true | false | 8,256 |
| Output gate | false | true | 8,312 |
| Both | true | true | 8,324 |

Twenty fits total. Counts include the same four inactive temperature scalars
as the retained model; actual counts are recorded in each result. Every arm
starts from scratch. Both additions use zero initialization without random
draws: the shared parameters, initial function, data order and dropout RNG
start identically for a given seed. Subsequent learning can differ.

The original seven-token EEGNet, flattening classifier, outer residual,
Q/K/V projections, token router and ordinary AdamW training stay unchanged.
This is not a frozen-backbone experiment. Data remain `../ml/A03T.gdf`,
22 channels, four classes, cue-relative 0–4 seconds, trial-local 2–30 Hz
filtering, artifact exclusion and training-only channel normalization.
Each seed retains its 162/54/54 train/validation/test split. Training remains
AdamW 0.005, weight decay 0.001, batch 64, balanced focal gamma 3, warmup 10,
cosine scheduling, no augmentation/EMA/SAM/AMP, first-best validation accuracy.

## Equations and isolation

For one head, retain `P0=V`, `P1=AV`, `P2=A²V` and
`H_i = sum_k alpha_i[k] Pk_i`, with the existing token-contrast router.

The temporal-bias arm uses destination index i, source index j, N tokens:

```text
lag = (j - i) / max(N - 1, 1)
b_h(i,j) = u_h * abs(lag) + v_h * lag + w_h * 1[i == j]
S_h = Q_h K_h^T / sqrt(d_head) + b_h
A_h = masked_softmax(S_h, top5(S_h))
```

All three parameters per head start at zero. The bias is added before both
mask selection and softmax. Existing threshold tie handling is retained.
The linear signed-lag term is softmax-equivalent to a source-position ramp;
it is not an independently expressive past/future relation. Temporal bias
requires time tokens, score-stage top-k and softmax graph normalization.

The output-gate arm uses:

```text
d_i = concat(LayerNorm(V_i), LayerNorm(P1_i - V_i))
g_i = 2 * sigmoid(c_h^T d_i + b_h)
H'_i = g_i * H_i
Z = concat_heads(H') W_O + bias_O
EEGNet attention output = X + Z
```

The descriptor matches the existing token router (feature-axis LayerNorm,
epsilon 1e-5). Each head adds 16 weights and one bias, all initially zero:
g=1. It scales H, before the output projection, and does not gate X or the
outer residual. The projection bias remains when H is suppressed. Descriptor
arithmetic uses FP32 outside autocast, preserving FP64 for reference checks.
This experiment supports the gate with `coefficient_conditioning=token_contrast`.

The implementations live in `agfl/attention/agfl/temporal_gate.py`. The two
flags default to false. With both disabled no new parameters/state keys are
created, and the retained forward arithmetic is used. Existing presets and
saved checkpoints continue to build with the flags omitted. New flags are
included in saved configuration and experiment identity; source and data
provenance checks continue to apply.

## Upload normal files from the Mac

Use the completed energy study / restored token checkout as the starting
point. Its original EEGNet and ordinary-AdamW engine are already correct.
These uploads include the prior energy rollback files. Stop active runs
before replacing source. No new dependencies, archive extraction or pytest
installation is required.

```bash
cd /Users/egor/Downloads/AGFL
scp agfl/attention/agfl/layer.py agfl/attention/agfl/config.py \
  agfl/attention/agfl/temporal_gate.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/attention/agfl/
scp agfl/analysis.py agfl/temporal_gate_study.py agfl/cli.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp agfl/visualization/diagnostics.py agfl/visualization/results.py \
  agfl/visualization/temporal_gate.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
scp agfl/presets/eegnet-a03-temporal-gate.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

No transfer has been performed by the assistant. The optional test files and
documentation are not needed to run this preset. If the cluster still lists
the retired energy preset, do not launch it with the restored source; its
archived reproduction workflow is separate.

## Allocate and launch

On the cluster login node, unless a GPU allocation is already active:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=4:00:00
```

After allocation succeeds:

```bash
srun --pty bash -l
```

Inside that GPU shell:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source /beegfs/home/georgii.promyslov/.venv/bin/activate
export MPLBACKEND=Agg
```

Training, then automatic validation diagnostics and all result plots:

```bash
python main.py sweep --preset eegnet-a03-temporal-gate \
  --output-dir results/eegnet-A03-temporal-gate-v1 --report
```

The requested allocation duration is not an estimate of completion time.
Fits run sequentially on one GPU with existing progress bars. Matching runs
are overwritten by default. To resume after interruption, add
`--skip-completed` only with unchanged source/configuration/data/environment;
incomplete fits restart at epoch 1. Use a fresh output folder for a changed
implementation. Do not replace the control with old downloaded results.

Separate checkpoint diagnostics, including new biases/gates and validation
removal checks (recovery without retraining):

```bash
python main.py diagnose-session results/eegnet-A03-temporal-gate-v1 \
  --device cuda --partition validation --embedding tsne
```

Separate result analysis and plots:

```bash
python main.py analyze results/eegnet-A03-temporal-gate-v1 --plots
```

## What to download and inspect

Download **`results/eegnet-A03-temporal-gate-v1/report/`** only, into a new
Downloads folder. Keep `artifacts/` (including checkpoints) on the cluster.

- `analysis/temporal_gate_study.md`: four-way accuracy table, validation-only
  ranking, combined-minus-control difference, and combination interaction.
- `analysis/temporal_gate_study.json`: seed coverage, exact experiment IDs,
  all matched validation/test results, paired fixed/lost trial IDs and counts.
- `analysis/temporal_gate_per_seed.csv`: individual contrasts and interaction.
- `analysis/figures/temporal_gate/`: four-arm accuracy and interaction plots
  for validation and test, as PNG and PDF. The usual learning curves,
  confusion matrices, ROC curves, per-class recall, paired differences and
  aggregate plots are still generated; `analysis/figures/index.md` lists them.
- Each `diagnostics/.../seed_*/temporal_gate.json`: learned bias weights,
  gate variation and AGFL-output/input norm ratio. Its `temporal_gate/*.npz`
  preserves every selected sample's graph, token/head gate and aligned IDs.
  Heatmaps show all selected trials, not just their average.
- For enabled additions, `temporal_gate/validation_ablation.json`, `.npz`,
  `.png` and `.pdf`: temporarily disable bias, gate, or both at the trained
  checkpoint, then restore all parameters. This is validation sensitivity,
  not retrained performance or a replacement for saved evaluation metrics.
  Test-partition diagnostics do not run these removal checks.

Existing coefficient/hop diagnostics describe the mixture before the new
output gate. Gate effects are recorded separately rather than silently
changing what older diagnostic arrays mean.

## How to interpret the comparison

Five directed contrasts are reported: each of the three candidates minus
control, plus both minus bias and both minus gate. Source, protocol, data,
seed and split must match. Existing paired tests remain exploratory and
Holm-corrected; overlapping splits are not independent observations.

The descriptive interaction is `accuracy_both - accuracy_bias - accuracy_gate
+ accuracy_control`. Positive interaction means the observed combined gain
exceeds the sum of the two individual gains; it does not prove significance.

Rank one architecture across all five seeds by mean validation accuracy,
then mean validation loss at the already selected checkpoints, then parameter
count. No selection is emitted until all four arms have the same five matched
seeds 0–4 and complete validation metrics. Short/partial runs remain visible
but cannot be labeled a completed study. Test results never decide the ranking.
All four test outcomes are reported because this is a declared ablation study.

No improvement is guaranteed. A03 is already a development subject. If a
generic addition helps, a later MHA-with-the-same-addition control and untouched
session/subject evaluation are needed for an AGFL-specific article claim.

## Disable or remove the changes

The old `eegnet-a03-token-agfl` preset still uses neither addition. For any
new training configuration, setting both attention flags to false selects
the retained path. Diagnose trained candidate checkpoints with their saved
flags; do not reinterpret them as controls or overwrite their saved configs.

For a complete source rollback, the ordinary text patch
[`archive/eegnet_temporal_gate_revert.patch`](archive/eegnet_temporal_gate_revert.patch)
reverses only this update, preserving pre-existing uncommitted changes. From
the unchanged updated checkout, `git apply --check` it before applying it.
It is an optional maintenance artifact, not part of upload/training. Keep
this implementation available separately if candidate checkpoints will need
future diagnostics; generic saved-prediction analysis remains available.

The new development checks use standard-library unittest and existing project
dependencies. They are not a training prerequisite and were **not executed
locally**. Static validation covers syntax, preset invariants, upload paths,
shell syntax and applicability of the rollback patch. GPU determinism,
runtime behavior and any accuracy improvement remain unverified until the
cluster run.
