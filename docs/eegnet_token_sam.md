> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# EEGNet/A03: token-routing AGFL with SAM

**Archived workflow — SAM code and its preset have been removed.** The
commands below document the completed experiment and require restoring the
[plain-text reproduction patch](archive/eegnet_sam_reproduction.patch) in a
separate copy of the rollback tree before the energy-routing change. It is
not a patch for the current energy-routing tree. These are not active launch instructions.
Existing downloaded reports remain intact. Use their generated tables and
plots; regenerating SAM-specific analysis also requires that archived code.
The patch was checked for applicability, not executed. See the
[rollback instructions](eegnet_after_sam.md).

**Completed: SAM AGFL reached 81.11%, down 4.07 pp from ordinary token
AGFL's 85.19%.** Validation also fell. All 20 runs and 469 PNG/PDF figure
pairs completed. Retain ordinary token AGFL (the active code has no `sam_rho` option);
see the [full results review](eegnet_a03_sam_review.md). Commands below
preserve reproduction of this unsuccessful ablation.

The new preset is **`eegnet-a03-token-sam`**. It keeps the retained
token-routing AGFL and original seven-token EEGNet, and changes training
to Sharpness-Aware Minimization (SAM) around the existing AdamW update.
SAM is also applied to MHA and Linformer for the attention comparison.
One launch runs **20 fits: four configurations × seeds 0–4**.
No dependencies are added. This tested recipe did not improve accuracy.

## Why this experiment

The [completed feature-routing study](eegnet_a03_feature_review.md) gave
token AGFL **85.19%** mean test accuracy, feature AGFL **84.07%**, MHA
**85.56%**, and Linformer rank 4 **85.93%**. Feature routing did not improve
validation either, so the next experiment retains token routing.

The token runs show a training/validation gap and local validation
instability. For example, seed 2's validation accuracy in epochs 158–162
was **77.78%, 83.33%, 88.89%, 87.04%, 81.48%**. This motivates testing a
training objective that penalizes loss in a neighborhood of the weights;
it does not establish that sharpness caused these errors. The
[original SAM paper](https://arxiv.org/abs/2010.01412) defines that objective.
Its reported results are not evidence of an accuracy gain on this dataset.

## Implementation

`training.sam_rho` defaults to **0.0**, preserving ordinary training. The
new preset uses one fixed value, **0.05**, with AMP disabled. For each batch:

```text
g       = gradient_w loss(w, batch)
epsilon = rho * g / (global_L2_norm(g) + 1e-12)
g_sam   = gradient_w loss(w + epsilon, same batch)
restore w exactly
AdamW.step(gradient = g_sam)
```

All trainable parameters participate, including the token router; frozen
parameters do not. The two passes replay the same Torch, Python and NumPy
random states, including CUDA dropout masks. Both use training-mode batch
statistics; only the first pass's running buffers and BatchNorm counters
are retained. The weights are restored from copies before AdamW applies
its moments and weight decay once. Optional gradient clipping applies to
the second gradient; the existing max-norm projection and optional EMA
update happen after the real optimizer step. This preset keeps EMA off.

If either probe fails, the helper restores weights, buffers and random
state, clears gradients, and re-raises before any optimizer step. Only
CPU/CUDA float32/float64 training without autocast is supported. Validation,
checkpoint selection and inference use ordinary, unperturbed weights.
The architecture, parameter count and inference computation are unchanged.
Each SAM batch requires two forward/backward passes, so allow more training
time. In the completed session, AGFL's recorded training/evaluation time
was about 1.61× the ordinary control's, excluding later diagnostics.

History records `sam.rho`, cumulative `sam.updates`, and
`sam.perturbed_train_loss`. The selected checkpoint records the SAM policy
and update count. Training figures include the perturbed loss as a separate
dashed curve. Existing checkpoints retain their saved configurations.

## Fixed comparison and interpretation

| Configuration | Training | Expected parameters |
|---|---|---:|
| Token AGFL control | Original AdamW | 8,244 |
| Token AGFL candidate | SAM + AdamW, rho 0.05 | 8,244 |
| MHA | SAM + AdamW, rho 0.05 | 8,036 |
| Linformer rank 4 | SAM + AdamW, rho 0.05 | 8,260 |

Every fit uses **250 epochs** on **`../ml/A03T.gdf`**, four classes and
22 EEG channels. Retained preprocessing: artifact exclusion, cue-relative
0–4 seconds, trial-local 2–30 Hz filtering, and training-only channel
normalization. Each seed has 162/54/54 train/validation/test trials. The
backbone retains kernel 32, pool 8×16, 32 features, seven time tokens and
dropout 0.5. Training retains AdamW learning rate 0.005, weight decay 0.001,
balanced focal loss with gamma 3, batch size 64, warmup/cosine scheduling,
no augmentation, and first-best validation-accuracy checkpoint selection.

The report answers two distinct questions:

1. **Does SAM help token AGFL?** The named `training_sam` comparison pairs
   SAM AGFL with ordinary AGFL only if everything except `sam_rho` matches,
   including attention options and source/data/package provenance. Their
   normal protocol IDs remain different and are both recorded.
2. **Does AGFL beat other attentions under the new recipe?** SAM AGFL is
   compared with SAM MHA and SAM Linformer under identical training. An
   ordinary-AGFL/SAM-baseline comparison is not treated as an attention effect.

The fresh ordinary control provides a reproduction check against 85.19%.
All arms are retrained with the same source version; old results are not
silently substituted. No test score selects a radius, recipe or checkpoint.
Read validation, per-class recall and all five seed differences alongside
the aggregate test mean. The desired +1 percentage point would require
three net extra correct test occurrences across these five 54-trial test
partitions (+1.11 pp). These are overlapping partitions, not 270 independent
trials. A03 is a repeatedly inspected development subject; confirmation on
untouched subjects or sessions is needed for a broader article claim.
SAM is a general training method, not an AGFL-specific architectural result.

## Upload only this iteration's runtime files

These commands assume the previous token/feature routing and diagnostic
JSON fixes are already on the cluster, as in the latest completed report.
Run from the Mac after any active study has finished. These are ordinary
files; no archive, new library, or development test invocation is needed.
The assistant has not transferred files or started cluster jobs.

```bash
scp /Users/egor/Downloads/AGFL/agfl/sam.py \
  /Users/egor/Downloads/AGFL/agfl/engine.py \
  /Users/egor/Downloads/AGFL/agfl/config.py \
  /Users/egor/Downloads/AGFL/agfl/analysis.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp /Users/egor/Downloads/AGFL/agfl/visualization/results.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
scp /Users/egor/Downloads/AGFL/agfl/presets/eegnet-a03-token-sam.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

## Launch

On the cluster, obtain an interactive GPU shell if not already allocated:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

The four-hour request is not a measured runtime estimate. In that shell,
use the existing environment and repository directory:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

Train all four configurations and automatically build their reports:

```bash
python main.py sweep --preset eegnet-a03-token-sam \
  --output-dir results/eegnet-A03-token-sam-v1 \
  --report
```

Use this fresh output directory. If the allocation expires, obtain a new
allocation, activate the same environment, and rerun that command with
`--skip-completed` added. Reuse requires matching settings, source and
environment provenance; incomplete fits restart at epoch 1. Do not update
the source during the study. Without `--skip-completed`, relaunching
restarts the fits rather than substituting prior completed results.

## Checkpoint diagnostics and result plots

`--report` runs these automatically. If reporting is interrupted, use
these separate recovery commands from the same cluster directory and
environment, without retraining. Keep the session's `artifacts/` available.

Checkpoint plots, graph/filter diagnostics and validation t-SNE:

```bash
python main.py diagnose-session results/eegnet-A03-token-sam-v1 \
  --device cuda --partition validation --embedding tsne
```

Result plots, paired differences, per-class tables and training histories:

```bash
python main.py analyze results/eegnet-A03-token-sam-v1 --plots
```

Result analysis can run even if checkpoint diagnostics failed. Download
only **`results/eegnet-A03-token-sam-v1/report/`**, into a new local folder.
Keep `artifacts/` on the cluster for regenerating diagnostics.

- `analysis/report.md`: readable accuracy/F1/AUC summary, with separate SAM labels.
- `analysis/statistical_comparisons.csv`: explicit SAM ablation and matched
  attention contrasts, effect sizes and exploratory paired statistics.
- `analysis/figures/index.md`: result figures, including paired differences,
  clean/perturbed training loss and per-class recall.
- `analysis/per_class.csv` and `analysis/per_class_summary.csv`: validation
  and test class metrics, including feet and tongue recall.
- `diagnostics/`: each run's `index.md` links checkpoint replay, graph,
  coefficient, embedding and classification figures.

Static checks can verify syntax, preset consistency and command structure.
They cannot establish optimizer numerics, CUDA runtime behavior or accuracy.
Development regression checks cover optimizer restoration, random-mask/BN
handling, checkpoint replay and comparison pairing; they are not a launch
prerequisite and have not been run locally.
