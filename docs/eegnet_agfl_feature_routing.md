> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# EEGNet/A03: feature-wise AGFL hop routing

**Completed: feature AGFL 84.07%, token AGFL 85.19%, static AGFL 84.44%,
MHA 85.56%, Linformer 85.93%.** The extension did not improve validation or
test accuracy. All 25 fits and 588 PNG/PDF figure pairs completed; see the
[results review](eegnet_a03_feature_review.md). Retain token AGFL as the
best observed AGFL test result. Commands below preserve reproduction of
the completed ablation.

The preset `eegnet-a03-feature-agfl` changes AGFL while retaining the original
seven-token EEGNet, preprocessing, training recipe and splits. No new
dependencies are needed. No project code or tests have been run locally.

## Why this change

The [completed token-routing study](eegnet_a03_token_review.md) improved
mean test accuracy from **84.44% to 85.19%**, still below MHA **85.56%** and
Linformer rank 4 **85.93%**. Feet recall rose, while tongue and left-hand
recall fell. Validation accuracy also fell, from 92.22% to 91.48%.

The existing token router applies one three-hop mixture to all eight
features in a head. The new router gives each feature its own mixture, so
local detail and neighborhood information can be retained differently
within a token. This removes a real parameter-sharing constraint; it does
not prove that the constraint caused the observed classification errors.
Feature-dependent graph filtering also has precedent in
[AdaGNN](https://arxiv.org/abs/2104.12840), but that work is not evidence of
an EEG accuracy gain for this implementation.

Two simpler alternatives were checked before choosing this experiment:

- Equal averaging of static/token AGFL probabilities gave validation correct
  counts `[52,48,47,48,49]`, or **90.37%**, below either individual method.
  This check used saved validation probabilities only; no test ensemble was
  evaluated or selected.
- The available seed-0 validation diagnostic showed an active router with
  only **0.29%** of corrections above 95% of their attainable bound.
  First/second propagated-hop cosine means were **0.90–0.94** across heads.
  This does not favor simply increasing the correction bound or adding more
  propagation depth. These diagnostic observations cover one seed only.

## Definition and compatibility

For head-local feature dimension `d`, maximum hop order `K`, token `i`, and
feature `f`:

```text
P_0 = V; P_k = A @ P_(k-1)             # polynomial preset
d_i = concat(LN(V_i), LN(P_1[i] - V_i))
r_i = reshape(tanh(G @ d_i), [d, K+1])
delta_i[f,k] = (scale/2) * (r_i[f,k] - mean_hops(r_i[f,:]))
alpha_i[f,k] = alpha_base[k] + delta_i[f,k]
H_i[f] = sum_k alpha_i[f,k] * P_k[i,f]
```

The opt-in option is `coefficient_conditioning="feature_contrast"` with
scale 0.5. Each bias-free gate has `d*(K+1)` outputs instead of `K+1`.
The gate is zero-initialized without random draws; shared parameters retain
their original initialization, and the initial function is static AGFL.
The new family can also represent the old token router by repeating its
weights across features. Training in this preset starts from scratch for
all arms; old checkpoints are not converted or substituted.

Each feature's corrections sum to zero across hops, with absolute component
bound `scale*K/(K+1)` (1/3 here). Base coefficients remain signed and
learnable. Descriptor arithmetic promotes half precision before subtraction
and uses FP32 outside autocast, preserving FP64 inputs. LayerNorm has no
affine parameters. Routing uses no labels or batch statistics.

The global default stays `static`; existing static, trial-power and token
configurations keep their computations and state-dict layouts. This new
feature axis is an experimental extension beyond the manuscript's scalar
coefficients; see [mathematics](mathematics.md#feature-wise-token-routing).

## One fixed comparison

| Arm | Role | Expected parameters |
|---|---|---:|
| Static AGFL | Original control | 8,052 |
| Token-routed AGFL | Current best observed AGFL test mean | 8,244 |
| Feature-routed AGFL | New extension | 9,588 |
| MHA | Base attention | 8,036 |
| Linformer rank 4 | Strongest observed baseline | 8,260 |

One sweep runs **25 fits: five arms × seeds 0–4**, each for 250 epochs.
All previous arms and their recipes are preserved. The new arm differs from
token AGFL only in its conditioning mode: its router has 1,536 weights,
1,344 more than token AGFL. This is not a parameter-matched comparison;
report the capacity increase alongside any accuracy change.

Dataset: `../ml/A03T.gdf`, four classes, 22 EEG channels, artifact exclusion,
cue-relative 0–4 seconds, trial-local 2–30 Hz filtering, training-only channel
normalization. EEGNet retains kernel 32, pool 8×16, 32 features and seven
tokens. Training retains AdamW 0.005, balanced focal gamma 3, batch 64,
weight decay 0.001, no augmentation/EMA and the original first-best
validation-accuracy checkpoint rule. All arms share 162/54/54 train/validation/
test trials for each seed. No candidate search or class-specific loss change
is included.

The analysis pairs feature AGFL with token AGFL, static AGFL, MHA and
Linformer only when seeds, split/data/source provenance and shared settings
match. Feature-versus-token additionally requires the same conditioning
scale. Different AGFL variants keep distinct labels and results.

The prespecified target was at least **+1 percentage point over 85.19%**,
accompanied by validation improvement and no repeated feet/tongue regression.
It was not achieved: the measured change is −1.11 pp. Three net correct test
occurrences across five 54-trial partitions correspond to +1.11 pp; these
are overlapping observations, not 270 independent trials. A03 remains a
development subject. A gain here needs confirmation on untouched subjects
or sessions before a broader paper claim. No gain, 90% score, or AGFL
advantage is guaranteed.

## Upload changed ordinary files

From the Mac, upload these runtime files after any active experiment ends.
These commands include the previous diagnostic/automatic-report fixes.

```bash
scp /Users/egor/Downloads/AGFL/agfl/attention/agfl/layer.py \
  /Users/egor/Downloads/AGFL/agfl/attention/agfl/config.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/attention/agfl/
scp /Users/egor/Downloads/AGFL/agfl/analysis.py \
  /Users/egor/Downloads/AGFL/agfl/cli.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
scp /Users/egor/Downloads/AGFL/agfl/visualization/diagnostics.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
scp /Users/egor/Downloads/AGFL/agfl/presets/eegnet-a03-feature-agfl.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

No archive extraction, dependency installation or development test invocation
is required. Files have not been transferred by the assistant.

## Launch and plots

Obtain an interactive GPU shell if not already allocated:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

The four-hour request is not a measured runtime estimate. In the allocated
shell use the existing environment, repository directory and fresh output:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg

python main.py sweep --preset eegnet-a03-feature-agfl \
  --output-dir results/eegnet-A03-feature-agfl-v1 \
  --report
```

This runs training followed by validation checkpoint diagnostics and result
plots. If only reporting fails, recover with the following separate commands
without retraining. For an interrupted training sweep, `--skip-completed`
can retain matching completed fits only when source/configuration/data and
environment still match; incomplete fits restart. Without it, matching fits
are rerun. Use this fresh folder for the new comparison.

Checkpoint diagnostics:

```bash
python main.py diagnose-session results/eegnet-A03-feature-agfl-v1 \
  --device cuda --partition validation --embedding tsne
```

Result plots and tables:

```bash
python main.py analyze results/eegnet-A03-feature-agfl-v1 --plots
```

Download **`results/eegnet-A03-feature-agfl-v1/report/`**. Keep `artifacts/`
and checkpoints on the cluster. The report contains:

- `analysis/report.md`, class-metric CSV files, statistical comparisons and
  `analysis/figures/index.md` with accuracy, confusion, class-recall and
  training-history plots.
- Per-run checkpoint figures and indices below `diagnostics/eeg/subject_A03/`.
- Feature AGFL's `coefficient_conditioning.json` schema 3 and
  `filters/<layer>_feature_coefficients.npz`: coefficients/deltas with axes
  `[trial,head,token,feature,hop]`, base `[head,hop]`, gates
  `[head,feature,hop,descriptor_feature]`, explicit IDs/indices, token and
  feature variation, zero-sum residuals, hop cosines and contribution norms.
- Per-head feature-coefficient heatmaps: every trial remains visible;
  columns group features within tokens. Feature coordinates are not
  electrodes or classes. Old token/trial diagnostic shapes stay unchanged.

Static syntax, preset consistency and command checks are performed locally
without importing the project. Regression tests cover initialization,
formulas, gradients, checkpoint compatibility, diagnostic axes and matched
comparisons, but were **not executed locally**. The completed cluster
experiment now verifies the saved results and diagnostic workflow; its
accuracy outcome is unfavorable. This is not a claim that the separate
development regression suite ran on the cluster.
