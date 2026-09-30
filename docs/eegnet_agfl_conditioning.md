> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# Fixed EEGNet comparison: original AGFL, conditioned AGFL and Linformer

**Completed:** static and conditioned AGFL both reached 84.44% mean test
accuracy; Linformer rank 4 reached 85.93%. Conditioning's feet improvement
was offset by worse tongue recognition. The active launch retains the original
EEGNet and tests [token-wise AGFL hop routing](eegnet_agfl_token_routing.md).
The subsequent power/EMA recipe also regressed and has been retired. Commands below
preserve this completed comparison; they are not the current launch.

Keep the seven-token sparse Q/K/V AGFL control. It had the highest mean
validation accuracy (**92.22%**) among the fixed recipes in the recent
matched Q/K/V, refinement and capacity searches. The latest capacity-selected
procedure reached **84.44%** mean test accuracy; its larger variants did not
improve mean validation accuracy. See the
[capacity results review](eegnet_a03_capacity_review.html).

This completed comparison changed how AGFL computes its filter coefficients.
It did not improve mean accuracy. The original AGFL recipe remains available
and was measured alongside the new one. This was a fixed comparison, not a
candidate-selection search.

## The implemented change

Static AGFL learns one signed coefficient per head and hop, shared across
trials. The opt-in `trial_power` mode adds a bounded correction derived from
the value features of the current trial. For a head with value matrix
`V_b` of shape `[tokens, features]`:

```text
d_b   = LayerNorm_features(log(mean_tokens(V_b ** 2) + 1e-6), eps=1e-5)
alpha_b = alpha_base + 0.5 * tanh(G @ d_b)
Z_b   = sum_{k=0..K} alpha_b[k] * A_b^k V_b
```

Each head owns a bias-free `G` of shape `[K+1, features]`, initialized to
zero without consuming random draws. At initialization the correction is
zero and the existing static parameters retain their initialization. There
is no learned normalization in the descriptor. Its reductions are within a
single trial, and it uses neither labels nor statistics from other trials.
It describes learned value-feature power, not a direct EEG frequency-band
measurement. Positive and negative coefficients remain possible; the bound
applies to the correction, not to the total learned coefficient.

For this EEGNet setting, four heads × three hops × eight features adds
**96 trainable parameters**. The controller adds operations too; a speed or
efficiency advantage is not assumed. It may still overfit this small dataset.
The experiment tests whether adapting the local/neighbor/multi-hop mixture
per trial improves the fixed static recipe.

Configuration keys:

```json
{
  "attention_options": {
    "coefficient_conditioning": "trial_power",
    "coefficient_conditioning_scale": 0.5,
    "coefficient_activation": "identity",
    "learnable_coefficients": true
  }
}
```

`static` remains the global default. Old saved configurations without these
keys follow the static path and keep the original checkpoint parameter keys.
`trial_power` requires identity activation, learnable coefficients, at least
two features per head and a positive finite scale. The shared implementation
is available to all backbones, with either AGFL propagation formula and
existing hop-projection modes. This preset retains the polynomial formula
and value-only hops. New configurations and source provenance identify the
change explicitly; old result files are not reinterpreted.

## One launch, three fixed configurations

| Configuration | Coefficients | Attention baseline |
|---|---|---|
| Original sparse AGFL | Static, learnable | Retained control |
| Conditioned sparse AGFL | Trial power, scale 0.5 | New variant |
| Linformer | Not applicable | Projection rank 4 |

Linformer rank 4 had the highest observed mean test accuracy in the completed
A03 attention comparison: **85.93%**, versus MHA **85.56%** and static sparse
AGFL **84.44%**. It is the strongest observed development baseline for this
EEGNet setup. The difference from MHA is small; it does not establish that
Linformer is generally superior. The comparison settings and original results
are recorded in the [all-attention workflow](eegnet_a03_all_attentions.md).

The preset is `eegnet-a03-conditioned-agfl`. It performs **15 sequential
fits: three fixed configurations × seeds 0–4, 250 epochs each, A03 only**.
All three configurations use:

- BCI IV 2a training-session file `../ml/A03T.gdf`, four classes, subject A03
  trained individually, artifact trials excluded.
- The same 60/20/20 stratified train/validation/test IDs for each seed,
  1,000-sample trial windows, 2–30 Hz trial filtering and training-only
  channel normalization. There is no preprocessing change in this iteration.
- EEGNet spatial convolution first, then attention over seven time tokens:
  pooling factors 8 and 16, 32 features, four heads, residual attention.
- Batch 64, AdamW at 0.005, weight decay 0.001, balanced focal loss with
  gamma 3, the existing warmup/cosine schedule, no augmentation and AMP off.
- Validation-accuracy checkpoint selection. Each fixed configuration's
  selected checkpoint evaluates test; no test result selects a checkpoint
  or chooses between the three methods.

The two AGFL configurations differ only in `coefficient_conditioning`.
Both retain Q/K/V projections, square-root score scaling, `K=2`, one-hop
initialization `[0, 1, 0]`, softmax graph normalization and scheduled Top-k.
The baseline is rerun with the same source snapshot and environment so that
the comparison uses matched provenance, rather than combining old and new
results. Fifteen fits plus diagnostics may exceed an allocation; no duration
guarantee is implied by a single launch command.

A03 and these splits have already informed development, including the choice
of baseline. These are exploratory results, even though within-run selection
remains isolated from test data. Five overlapping seeded splits are not five
independent subjects. General article claims need a frozen method evaluated
on broader subjects and confirmation data not used for further tuning.

## Source for reproduction

The [current workflow](eegnet_agfl_token_routing.md#upload-the-prepared-update)
provides a coherent source archive, replacing the old six-file partial upload.
No new dependencies or environment installation are required. Development
tests are not a prerequisite for training or plotting. For exact historical
training reproduction, retain the original checkout and environment recorded
in the saved provenance. A newer compatible checkout records its own source.

## Allocate a GPU and activate the environment

From the cluster login shell, unless an allocation is already active:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

After allocation succeeds, use the existing Python 3.12+ CUDA environment:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

Dataset: **`../ml/A03T.gdf`**. Output session:
**`results/eegnet-A03-agfl-conditioned-v1`**. This is the completed study's
folder. Use a different output directory when deliberately rerunning from
a new checkout, and substitute it in both plotting commands below. Retain
the original results and configurations.

## Train and generate the report

```bash
python main.py sweep --preset eegnet-a03-conditioned-agfl \
  --output-dir results/eegnet-A03-agfl-conditioned-v1 \
  --skip-completed --report
```

One command trains all three configurations with the existing progress bars,
then generates validation checkpoint diagnostics and result plots.
`--skip-completed` retains matching completed fits; after an interruption,
relaunch the same command with unchanged source, settings and environment.
Incomplete fits restart at epoch 1. There is no need to delete seed folders.
Omitting `--skip-completed` also overwrites matching completed fits.

## Separate plot commands

These commands are optional after successful automatic reporting. If plotting
is interrupted, regenerate it from the saved results and checkpoints without
retraining. Run from the same cluster directory and environment.

Checkpoint diagnostics, including trial-coefficient plots:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-conditioned-v1 \
  --device cuda --partition validation --embedding tsne
```

Result plots and comparison tables:

```bash
python main.py analyze results/eegnet-A03-agfl-conditioned-v1 --plots
```

## What to download and inspect

Download only **`results/eegnet-A03-agfl-conditioned-v1/report/`**. Keep the
checkpoints and working files in `artifacts/` on the cluster.

| File within `report/` | Purpose |
|---|---|
| `analysis/report.md` | Mean accuracy, macro F1 and ROC-AUC with seed SD for each configuration |
| `analysis/attention_comparison.csv` | Per-configuration summaries across seeds with saved configuration identifiers |
| `analysis/statistical_comparisons.csv` | Both AGFL-minus-Linformer comparisons and conditioned-minus-static AGFL |
| `analysis/aggregation.json` | Full summaries, pairing diagnostics and settings |
| `runs/eeg/subject_A03/<model-attention-id>/seed_*/result.json` | Individual seed metrics and complete saved settings |
| `analysis/figures/index.md` | Learning, confusion, ROC, paired differences and diagnostic comparison plots |
| `diagnostics/eeg/subject_A03/<model-attention-id>/seed_*/index.md` | Per-checkpoint signal, embedding, attention and filter plots |
| Each diagnostic run's `filter_statistics.json` | Learned base coefficients and graph/filter settings |
| Conditioned runs' `coefficient_conditioning.json` | Per-head/hop coefficient variation, bounded adjustments and saturation |
| Conditioned runs' `filters/*_trial_coefficients.npz` | Actual per-trial coefficients, base values, deltas, gate weights and sample IDs |
| Conditioned runs' `filters/*_trial_coefficients.png` and `.pdf` | Effective-coefficient and normalized-adjustment heatmaps |

The report distinguishes **AGFL (static)**, **AGFL (trial-conditioned,
scale 0.5)** and **Linformer (rank 4)**. Experiment IDs remain present to
distinguish other settings. The additional conditioned-minus-static comparison
is emitted only when all other attention settings and the training/data
protocol match; seed/split/data matching still applies. Multiple comparisons
are included in the existing Holm adjustment, and seed-level inference is
explicitly exploratory.

Coefficient diagnostics capture every selected validation trial across every
diagnostic batch, with explicit trial/head/hop axes and checked sample-ID
order. The descriptor and filter use the actual per-head value inputs.
Base-coefficient plots for conditioned runs are labeled as base parameters,
not as the complete effective filter. Nonzero gate weights alone do not show
useful adaptation: inspect coefficient variation, saturation and the paired
accuracy difference together. These plots describe the model; they are not
causal explanations of EEG physiology. Optional diagnostic skips are recorded
in each manifest.

There is no `search_report.md` or candidate-selection winner for this fixed
comparison. All three configurations' results remain visible. A report-only
download supports saved-result plotting; regenerating checkpoint diagnostics
requires the retained cluster checkpoints and original dataset.

## Current scope

Current work stays with the original EEGNet and changes AGFL only. Follow the
[token-routing comparison](eegnet_agfl_token_routing.md) for the active launch.
Linformer rank 4 remains the strongest observed previous A03 baseline; neither
this completed conditioning comparison nor that ranking establishes broader
model or subject superiority. All unfavorable results remain part of the record.

## Verification status

Source and configuration were reviewed independently. Static Python syntax,
JSON consistency, documentation links, shell syntax and whitespace checks
were performed locally. Development tests were added for initialization/RNG
preservation, gradient paths, old static checkpoints, conditioning bounds and
sample independence, preset control retention, report pairing and diagnostic
capture. **No project code, tests, training or inference was run locally.**
The later cluster report completed all 15 runs and 313 figure pairs. Its
conditioning was active, but both AGFL variants averaged 84.44% accuracy.
