> **Archived temporal EEG experiment.** These launch instructions are retired as of 28 September 2026. Saved results retain their original interpretation. Use [the inter-channel EEG workflow](eeg_interchannel.md) for all new EEG runs.

# EEGNet attention comparison across all nine subjects

Preset: `eegnet-nine-subject-attentions`. Session:
`results/eegnet-nine-subject-attentions-v1`.

The successful A03 augmented AGFL recipe is frozen. Each of the nine subjects
is trained individually with the same original EEGNet, preprocessing, optimizer,
split ratios and checkpoint rule. Results average five seeds within each
subject, then average the nine subject means with equal weight. There is no
pooled-subject model, candidate search or test-based selection.

| Configuration | Attention settings |
|---|---|
| Augmented AGFL | Retained token routing, scale 0.5, K=2, scheduled top-k; temporal bias and output gate enabled |
| Augmented MHA | The same temporal-bias and per-head output-gate modules; otherwise dense MHA |
| Plain MHA | Both additions disabled |
| Performer | 64 random features |
| Linformer | Rank 4 |
| Linformer | Rank 7 |
| Nyströmformer | 4 landmarks, pinv tolerance 1e-5 |
| Nyströmformer | 7 landmarks, pinv tolerance 1e-5 |

**360 fits: 8 configurations × 9 subjects × seeds 0–4, 250 epochs each.**
The two ranks/landmark settings are carried over from the earlier all-attention
comparison. They are not selected after observing this experiment. Seven
landmarks equals the seven EEGNet tokens and is the dense-attention limit in
this implementation; it is a reference control, not an independent mechanism
from MHA. Learned rank-7 Linformer projections do not make it identical to MHA.

The existing protocol is retained: four classes, T session only, artifact
exclusion, trial-local 2–30 Hz filtering, cue-aligned 1,000-sample windows,
training-only channel normalization and stratified 60/20/20 splits. A03's
162/54/54 counts need not apply to other subjects after artifact exclusion.
All attention configurations receive identical partitions for a given
subject/seed. This is within-session evaluation, not T-to-E evaluation.

## Upload the changed runtime files

Run on the Mac. The destination is the same cluster checkout used for the
completed temporal/gate comparison; its existing AGFL modules remain needed.
No archive, new library or test-suite installation is required.

```bash
scp /Users/egor/Downloads/AGFL/agfl/config.py \
    /Users/egor/Downloads/AGFL/agfl/analysis.py \
    /Users/egor/Downloads/AGFL/agfl/multi_subject_study.py \
    georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
```

```bash
scp /Users/egor/Downloads/AGFL/agfl/attention/mha/__init__.py \
    /Users/egor/Downloads/AGFL/agfl/attention/mha/config.py \
    /Users/egor/Downloads/AGFL/agfl/attention/mha/layer.py \
    georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/attention/mha/
```

```bash
scp /Users/egor/Downloads/AGFL/agfl/visualization/maps.py \
    /Users/egor/Downloads/AGFL/agfl/visualization/results.py \
    /Users/egor/Downloads/AGFL/agfl/visualization/temporal_gate.py \
    /Users/egor/Downloads/AGFL/agfl/visualization/subject_study.py \
    georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/visualization/
```

```bash
scp /Users/egor/Downloads/AGFL/agfl/presets/eegnet-nine-subject-attentions.json \
    georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

Upload all eleven files before starting. No upload was performed by the
assistant. Keep the source and environment unchanged during this session.

## Allocation and environment

On the login node, if there is no active allocation:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=4:00:00
```

After allocation is granted, enter the allocated GPU node:

```bash
srun --pty bash -l
```

In that GPU shell:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source /beegfs/home/georgii.promyslov/.venv/bin/activate
export MPLBACKEND=Agg
```

The dataset directory is `/beegfs/home/georgii.promyslov/ml` (`../ml`), with
`A01T.gdf` through `A09T.gdf`. E-session files and labels are not required.
Four hours is an allocation request, not a measured completion-time estimate.
Training and diagnostics run sequentially on one GPU using the existing
Python 3.12 environment and progress bars.

## Training

```bash
python main.py sweep --preset eegnet-nine-subject-attentions \
  --output-dir results/eegnet-nine-subject-attentions-v1 --report
```

This trains all configurations, then generates checkpoint diagnostics and
saved-result plots. Normal relaunch retrains and replaces matching run outputs.
The previous A03 report is retained in its original session directory.

If the allocation expires, re-enter a GPU allocation and resume the same
unchanged study with:

```bash
python main.py sweep --preset eegnet-nine-subject-attentions \
  --output-dir results/eegnet-nine-subject-attentions-v1 --skip-completed --report
```

Completed runs are reused only after checking saved source, packages, data,
split and configuration provenance; unfinished fits restart from epoch 1.
Do not edit code or the study recipe midway through a resumable run. A new
recipe belongs in a fresh session, not mixed with this comparison.

## Checkpoint diagnostic plots, separately

If training finished but report generation stopped, run this without retraining:

```bash
python main.py diagnose-session results/eegnet-nine-subject-attentions-v1 \
  --device cuda --partition validation --embedding tsne
```

Augmented MHA and AGFL both produce temporal-bias and token-gate arrays/heatmaps,
plus validation-only sensitivity checks disabling bias, gate and both. MHA
routing maps include its learned bias and are explicitly shown before gating.
All other existing checkpoint plots remain available. Saved parameters are
restored after sensitivity checks; reported training/test metrics are unchanged.

## Result plots and averaged tables, separately

```bash
python main.py analyze results/eegnet-nine-subject-attentions-v1 --plots
```

This command can also inspect a partial session. The declared matrix in every
saved config makes entirely missing subjects or configurations visible, as well
as missing seeds. Incomplete configurations receive no nine-subject mean;
subject-paired significance calculations wait until the full matrix is present.
Mismatched split/source/protocol settings block the study's pooled summaries.

## Download and read

Download only **`results/eegnet-nine-subject-attentions-v1/report/`**.
Keep `artifacts/` and checkpoints on the cluster.

- `analysis/subject_study.md`: completeness and the main averaged comparison.
- `analysis/subject_study_per_subject.csv`: each subject's seed-averaged scores.
- `analysis/subject_study_averages.csv`: equal-weight nine-subject means and
  between-subject sample SD for accuracy, macro F1 and macro ROC-AUC.
- `analysis/subject_study_comparisons.csv`: augmented AGFL against all seven
  comparators, plus augmented MHA against plain MHA. Tests use nine paired
  subject means, not 45 subject/seed observations. Holm correction covers all
  eight contrasts × three metrics, separately for t and Wilcoxon tests.
- `analysis/subject_study.json`: full expected matrix and missing-run details.
- `analysis/figures/index.md`: all PNG previews and vector PDF links.
- `analysis/figures/subject_study/eegnet-nine-subject-attentions-v1/`:
  three subject heatmaps, an averaged metric figure, and three paired-subject
  difference figures. Existing per-run and per-subject plots remain.
- `diagnostics/`: checkpoint evidence for each completed subject/seed/configuration.

`analysis/statistical_comparisons.csv` remains the exploratory within-subject
seed-level comparison. Use the new **subject_study** tables for the main
nine-subject analysis. A03 and other previously inspected data remain development
data; broader coverage does not make the entire study independent confirmation.

## Implementation and verification

Plain MHA keeps the same forward computation and checkpoint parameter names.
Its additions are off by default; the augmented arm sets both flags explicitly.
The exact AGFL bias/gate classes are reused, adding 80 parameters to either
attention at this EEGNet width. AGFL, EEGNet and preprocessing code were not
modified for this experiment. The additions are described in
[the mathematics reference](mathematics.md#matched-mha-temporal-biasoutput-gate-control).

Static Python syntax, JSON recipe/matrix checks, shell syntax and diff checks
were performed locally. No project CLI, model, training, inference or tests
were executed locally. Optional standard-library unittest files cover neutral
initialization, learned formulas, diagnostic restoration, complete/partial
matrices, subject pairing and plot outputs; they are not a launch prerequisite.
Numerical execution and experiment outcomes remain unverified until the cluster
run. No accuracy improvement is assumed.
