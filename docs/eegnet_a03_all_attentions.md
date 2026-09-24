# Compare all five attentions inside EEGNet on A03

This study is complete. Fixed sparse AGFL averaged 84.44% test accuracy,
MHA 85.56%, Performer 84.44%, Linformer rank 4/7 85.93%/79.63%, and
Nyström landmarks 4/7 51.48%/85.19%. All 35 runs and 691 figure pairs
completed. The current follow-up is the
[EEGNet power/EMA comparison](eegnet_power_ema.md). Commands below preserve
the completed comparison; a reproduction with changed source is a new study.

This experiment compares MHA, AGFL, Performer, Linformer and Nystromformer
inside the same EEGNet backbone. It uses A03 only, seeds 0–4 and 250 epochs
for each run: **seven configurations × five seeds = 35 fits**. The dataset,
trial partitions, preprocessing, training recipe and checkpoint-selection
policy are shared. Each configuration's checkpoint is
selected using validation accuracy and then evaluated on test. Every
attention remains in the comparison; there is no per-seed choice of a
winning attention and no `tune-eegnet` candidate search.

AGFL uses the existing sparse Q/K/V polynomial recipe: learned projections,
one-hop coefficient initialization, `K=2`, square-root head-dimension scaling
and scheduled Top-k, normally five of seven tokens. The three-neighbor
experiment did not improve mean validation accuracy, so this comparison
retains the established five-neighbor recipe. The threshold tie policy can
retain extra neighbors when scores tie.

All methods use four heads and seven temporal tokens. Shared settings include
training-channel normalization, a 2–30 Hz trial filter, a 1,000-sample window,
artifact exclusion, 60/20/20 stratified train/validation/test partitions,
batch size 64, AdamW with learning rate 0.005, balanced focal loss with
gamma 3, a warmup/cosine schedule, and no data augmentation. Subjects are
not pooled.

## Seven configurations, with explicit approximation controls

| Attention | Configuration |
|---|---|
| MHA | Standard full attention |
| AGFL | Existing sparse Q/K/V polynomial recipe |
| Performer | `random_features=64` |
| Linformer | `projection_rank=4` |
| Linformer | `projection_rank=7` |
| Nystromformer | `landmarks=4`, `pinv_rtol=1e-5` |
| Nystromformer | `landmarks=7`, `pinv_rtol=1e-5` |

EEGNet supplies only seven tokens here. Four landmarks exercise the Nyström
approximation; seven landmarks provide a full-attention formula control.
Floating-point differences from MHA can remain. Linformer with rank seven
still applies learned projections and is a separate configuration from MHA.
The rank-four and landmark-four runs measure compressed settings, while
the corresponding seven-token settings show the effect of removing that
bottleneck.

Report both Linformer settings and both Nyström settings separately. Do not
pool their seeds together or retain only the setting with better test scores.
Every configuration has the same five-seed training and evaluation budget.

## Upload the new preset

The new JSON preset is the only project file required for this launch,
assuming the cluster checkout already ran the preceding top-three study.
It adds experiment settings and uses the existing implementation and Python
environment. No additional library or development test run is required.

Run this command **on the Mac**, before connecting to the cluster:

```bash
scp /Users/egor/Downloads/AGFL/agfl/presets/eegnet-a03-all-attentions.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

## Cluster launch

From the cluster login shell, request one GPU, four CPUs and 10 GB of memory
for up to four hours. Wait until allocation succeeds and the compute-node
shell opens:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

Activate the existing environment and work from `AGFL`. EEG data is expected
in `../ml`, including `A03T.gdf`:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source ~/.venv/bin/activate
export MPLBACKEND=Agg
```

Train all 35 runs in the comparison's own new session folder:

```bash
python main.py sweep --preset eegnet-a03-all-attentions \
  --output-dir results/eegnet-A03-all-attentions-v1 \
  --skip-completed
```

An identical relaunch preserves matching completed runs and restarts an
incomplete run from epoch 1. Keep the checkout, environment and settings
stable during the comparison. The new folder preserves the previous Q/K/V
and top-three studies.

Generate checkpoint plots for every completed configuration and seed:

```bash
python main.py diagnose-session results/eegnet-A03-all-attentions-v1 \
  --device cuda --partition validation --embedding tsne
```

Generate training, confusion-matrix, ROC and attention-comparison plots,
along with the result tables:

```bash
python main.py analyze results/eegnet-A03-all-attentions-v1 --plots
```

## Download and interpret the report

Download only **`results/eegnet-A03-all-attentions-v1/report/`**. Keep the
`artifacts/` folder on the cluster for checkpoint diagnostics.

- `analysis/figures/index.md`: result-plot index.
- `diagnostics/`: per-run signal, embedding and attention figures, with
  their own indexes.
- `analysis/attention_comparison.csv`: attention results across seeds.
- `analysis/aggregation.json`: full configurations and pairing metadata;
  use it to map figure/table IDs to Linformer ranks and Nyström landmark
  counts.
- `analysis/statistical_comparisons.csv`: available paired comparisons.
- `analysis/report.md`: analysis summary.
- `runs/`: saved metrics, predictions, histories, configuration and splits.

Assess every configuration on all five matched seeds, including unfavorable
outcomes. Unlike the earlier validation-selected searches, each fixed
attention setting receives test results across all five seeds. The five
random trial splits overlap and are not independent test cohorts. A03 and
these splits have already been used for development; this experiment is a
comparison on that subject, not new confirmation of the paper's full
nine-subject claim. Accuracy superiority and runtime gains must be measured;
neither is assumed.
