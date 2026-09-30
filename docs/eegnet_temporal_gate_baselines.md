> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# Combined AGFL versus MHA and Linformer on A03

This fixed comparison checks the combined temporal-bias/output-gate AGFL
against the previous MHA and rank-4 Linformer baselines using the current
source in one session. The [completed ablation review](eegnet_a03_temporal_gate_review.md)
records AGFL's 87.78% development result. This launch makes no model,
preprocessing, optimizer, epoch-selection or architecture-selection changes.

Preset: `eegnet-a03-temporal-gate-baselines`.

| Attention | Configuration | Seeds | Epochs per seed |
|---|---|---|---:|
| AGFL | Retained token routing + temporal bias + output gate | 0–4 | 250 |
| MHA | Unmodified base attention | 0–4 | 250 |
| Linformer | Projection rank 4 | 0–4 | 250 |

**15 fits in one sequential launch.** Same original seven-token EEGNet,
A03T data, artifact exclusion, training-only normalization and 162/54/54
partitions as the completed experiment. The AGFL architecture is fixed
before this launch; there is no additional candidate search. MHA and
Linformer settings exactly match their earlier comparison presets.

The five seeds and A03 data have already informed development. Repeating
this deterministic setup checks reproduction and provides directly matched
baseline results; it is not independent confirmation or more independent
statistical evidence. Later investigation of AGFL-specific benefits should
also consider MHA with the same generic enhancements and untouched data.

## Upload only two files

The cluster must already contain the temporal/gate implementation used for
the completed 20-fit report. Only the new JSON preset and one small reporting
update need uploading. The update prevents a three-attention comparison from
being mislabeled an incomplete four-AGFL-arm experiment. Existing accuracy,
per-class, paired-statistic and diagnostic reports continue to be produced.

On the Mac:

```bash
scp /Users/egor/Downloads/AGFL/agfl/presets/eegnet-a03-temporal-gate-baselines.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
scp /Users/egor/Downloads/AGFL/agfl/temporal_gate_study.py \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
```

No transfer was performed by the assistant. No dependencies or test-suite
installation is required. Upload before starting the session.

## GPU allocation and environment

If there is no active allocation, run on the cluster login node:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=4:00:00
```

After allocation succeeds:

```bash
srun --pty bash -l
```

Inside the GPU shell, working directory is the cluster AGFL checkout and
dataset is `../ml/A03T.gdf`:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source /beegfs/home/georgii.promyslov/.venv/bin/activate
export MPLBACKEND=Agg
```

## Training and plots

Train all three attentions, then automatically generate checkpoint diagnostics
and result plots in the fresh session directory:

```bash
python main.py sweep --preset eegnet-a03-temporal-gate-baselines \
  --output-dir results/eegnet-A03-temporal-gate-baselines-v1 --report
```

Separate checkpoint diagnostics, for report recovery without retraining:

```bash
python main.py diagnose-session results/eegnet-A03-temporal-gate-baselines-v1 \
  --device cuda --partition validation --embedding tsne
```

Separate saved-result plots:

```bash
python main.py analyze results/eegnet-A03-temporal-gate-baselines-v1 --plots
```

Download only **`results/eegnet-A03-temporal-gate-baselines-v1/report/`**.
Read `analysis/report.md` and `analysis/statistical_comparisons.csv`; open
`analysis/figures/index.md` for all result plots. `diagnostics/` contains
per-seed checkpoint plots, including combined AGFL's biases/gates and
validation removal checks. `temporal_gate_study.md` is intentionally absent:
this session contains one AGFL variant, not the earlier factorial matrix.

The normal command retrains matching runs. Use `--skip-completed` only to
resume this same session with unchanged source/configuration/data/environment.
Keep the earlier ablation folder intact. If only plotting fails, use the
two separate reporting commands above instead of training again.

## Verification and removal

Static JSON/source and shell-syntax checks were performed; no project code
ran locally. Current runtime and outcomes await the cluster run. Both
attention flags remain opt-in, and the old presets remain unchanged.
The temporal/gate rollback patch has been refreshed to include this preset
and reporting adjustment while preserving pre-existing work and result reviews.
