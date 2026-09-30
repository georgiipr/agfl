# EEG inter-channel experiments

28 September 2026. New EEG runs use **one graph node per electrode** in every
backbone. ECG continues to use temporal or temporal-patch attention. The old
temporal EEG studies are archived and cannot be launched as current presets.
Saved results and checkpoint diagnostics retain their original meaning.

The current follow-up is [EEGNet spatial fusion](eegnet_spatial_fusion.md): A03,
base AGFL versus base MHA, ten fits. It restores EEGNet's early full-channel
spatial convolution and preserves its 224 pooled features alongside the
32-feature electrode attention output. New EEGNet defaults use
`electrode_architecture=spatial_fusion`; the earlier presets in this document
are explicitly pinned to `compact` to preserve the completed experiments.

## Architecture and scope

For BCI IV 2a, each AGFL/MHA graph has 22 electrode nodes. Every EEG attention
insertion is checked against the recording channel count before training.
`attention_axis=time` and `temporal_bias=true` are rejected for new EEG runs,
including custom JSON, overrides, direct model construction and tuning.
Temporal convolutions within each electrode remain necessary to encode its
waveform; this is not temporal attention or a temporal graph.

The preceding `compact` EEGNet path, used by the completed reports, is:

```text
[batch, 22 electrodes, 1000 samples]
  shared EEGNet temporal/depthwise/separable feature extraction per electrode
[batch, 22 electrodes, 32 features × 7 temporal bins]
  shared per-electrode linear projection: 224 → 32
  learned electrode identities (no channel-index distance assumption)
[batch, 22 electrodes, 32 features]
  selected attention across electrodes; 4 heads; AGFL/MHA maps 22 × 22
  residual and learned spatial readout
[batch, 32 features] → four-class classifier
```

This is an electrode-preserving EEGNet adaptation, not a claim of exact parity
with the authors' original spatial convolution or the previous time-token model.
The attention width is 32 instead of the earlier electrode model's 224; graph
mixing happens before the electrodes are aggregated. Temporal bins become
features inside each node, never graph nodes. EEGEncoder and DSTS likewise
encode time within each electrode before channel mixing; Conformer and
Signal Transformer already used electrode nodes. ECG construction is unchanged.

Enhanced AGFL retains token-conditioned signed hop coefficients, Q/K/V,
K=2, scheduled top-k and the optional output gate. Here a token is an electrode.
The scheduled cutoff is 17 of 22 sources in EEGNet's single layer (ties can
retain more), not the old five of seven time positions. Temporal score bias is
disabled: the arbitrary gap between channel indices has no time/distance meaning.
The gate works on electrode messages and is also offered to MHA as a matched
control. No new graph-filter equation or anatomical-distance prior is introduced.

Standard montage coordinates are used only for plots. Learned graph edges are
sensor-level model routing weights, not estimates of anatomical or causal
connectivity. The same electrode order must be preserved in data and checkpoints.

The data protocol is retained: four classes, subjects trained individually,
T-session cue-relative 0–4 s windows, 2–30 Hz trial-local filtering, marked
artifacts excluded, train-only channel normalization, matched stratified
60/20/20 splits, seeds 0–4. This is not official T-to-E evaluation. No test data
select candidates/checkpoints. No 90% accuracy or improvement is assumed.

## Presets

| Preset | Scope | Fits |
|---|---|---:|
| `eegnet-a03-spatial-fusion` | A03: restored spatial path plus electrode attention, base AGFL versus base MHA | 10 |
| `eegnet-a03-base-agfl-attentions` | A03: base AGFL versus base MHA, Performer-64, Linformer-8 and Nyströmformer-8 | 25 |
| `eegnet-a03-a04-interchannel` | A03 and A04 separately: original base AGFL, enhanced AGFL, base MHA | 30 |
| `eegnet-interchannel` | AGFL with electrode routing and output gate, all nine subjects | 45 |
| `eegnet-interchannel-comparison` | A03: gated AGFL, equally gated MHA, plain MHA | 15 |
| `eegnet-interchannel-attentions` | All nine: gated AGFL, AGFL without output gate, gated MHA, plain MHA, Performer-64, Linformer-8, Nyströmformer-8 | 315 |

The ungated variants still contain attention. All arms share the same backbone,
training protocol and per-seed samples. In the seven-arm preset, the two AGFL
arms differ only in output gating. The A03/A04 preset instead compares the
complete original and enhanced AGFL versions; several mechanisms differ.
Eight Nyström landmarks are an approximation for 22
nodes; this is not the old seven-of-seven dense-attention control.

The comparison presets declare their complete subject/seed/arm matrix. If you
change `data.subjects` or `seeds`, change `study.subjects` or `study.seeds` to
match when defining a smaller study. Incomplete declared coverage is reported
as incomplete, not silently averaged into a full-study result.

`eeg`, `eeg-comparison`, `eeg-ablations`, `eeg-session` and
`paper-renormalized-eeg` also resolve to inter-channel EEG. Use these generic
presets with `--model` for other backbones; the EEGNet-specific presets contain
EEGNet-specific settings and must not be repurposed with another model key.

## Upload the normal source files

On the Mac, from `/Users/egor/Downloads/AGFL`:

```bash
scp -r agfl georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
scp main.py pyproject.toml README.md georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

This uploads source only, without datasets/results. There is no archive to
extract and no dependency installation. Stop active jobs before replacing
source. The retired-preset manifest also rejects stale temporal JSON files
left on the remote machine by an earlier `scp`.

## Cluster allocation and environment

Use the existing Python 3.12+ environment. Dataset files are
`/beegfs/home/georgii.promyslov/ml/A01T.gdf` through `A09T.gdf`, hence `../ml`
from `/beegfs/home/georgii.promyslov/AGFL`.

From the login node, unless a GPU is already allocated:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=4:00:00
```

Once granted:

```bash
srun --pty bash -l
```

In the allocated shell:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
source /beegfs/home/georgii.promyslov/.venv/bin/activate
export MPLBACKEND=Agg
```

Four hours is an allocation request, not a measured completion estimate.
No `.sbatch` file, new library or development test suite is required.

## Optional two-epoch check, one subject and seed

Keep this separate from research results. It verifies execution on the cluster;
its accuracy does not estimate the full experiment's performance.

```bash
python main.py run --preset eegnet-interchannel --seeds 0 \
  --set 'data.subjects=[3]' --set training.epochs=2 \
  --output-dir results/eegnet-A03-interchannel-check
```

```bash
python main.py diagnose-session results/eegnet-A03-interchannel-check \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

```bash
python main.py analyze results/eegnet-A03-interchannel-check --plots
```

## Completed design: A03 base AGFL versus other attentions

Use `eegnet-a03-base-agfl-attentions`: A03 only, five seeds (0–4), five
attention mechanisms, 250 epochs each, **25 fits**. All five train afresh in
a new session. The base AGFL and base MHA settings, EEGNet backbone,
preprocessing, training schedule and split protocol are copied from the
preceding A03/A04 experiment. Its AGFL accuracy is a reference, not an assumed
outcome of this run.

| Arm | Attention settings |
|---|---|
| `agfl_base` | Original AGFL defaults: K=2, scheduled top-k, globally learned signed hop coefficients, split-input graph, learned temperature, separate hop projections |
| `mha_base` | Standard four-head MHA |
| `performer64` | Performer with 64 random features |
| `linformer8` | Linformer with projection rank 8 |
| `nystromformer8` | Nyströmformer with 8 landmarks and pseudoinverse tolerance 1e-5 |

All arms have four heads and 22 electrode nodes; no output gates or temporal
bias are enabled. The report declares four comparisons: base AGFL minus each
comparator, for accuracy, ROC-AUC and macro F1. With one subject, subject-paired
significance tests are unavailable; seed-level comparisons remain exploratory.
This experiment does not establish performance across subjects.

Only the new preset needs uploading if the preceding A03/A04 source is already
installed. From the Mac:

```bash
scp /Users/egor/Downloads/AGFL/agfl/presets/eegnet-a03-base-agfl-attentions.json georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
```

Use the existing Python environment in an allocated GPU shell. Working
directory: `/beegfs/home/georgii.promyslov/AGFL`; dataset: `../ml/A03T.gdf`.
Training:

```bash
python main.py sweep --preset eegnet-a03-base-agfl-attentions \
  --output-dir results/eegnet-A03-base-agfl-attentions-v1
```

Checkpoint diagnostics, including electrode graphs and AGFL hop coefficients:

```bash
python main.py diagnose-session results/eegnet-A03-base-agfl-attentions-v1 \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

Result plots and accuracy tables:

```bash
python main.py analyze results/eegnet-A03-base-agfl-attentions-v1 --plots
```

Download `results/eegnet-A03-base-agfl-attentions-v1/report/`. For accuracy
only, run the last command without `--plots` and send
`report/analysis/subject_study.json`; checkpoint diagnostics are not needed
to calculate scores. No `--skip-completed` is included. Configuration and
documentation were checked statically; no project code was executed locally.

## Previous experiment: A03 and A04, base/enhanced AGFL versus base MHA

Use `eegnet-a03-a04-interchannel`. Both subjects are trained separately:
two subjects × three variants × five seeds (0–4) = **30 fits**, 250 epochs each.
All models use the electrode-preserving EEGNet above, the same preprocessing,
training settings and per-seed splits. No temporal bias is enabled in any arm.

| Setting | Base AGFL (`agfl_base`) | Enhanced AGFL (`agfl_enhanced`) | Base MHA (`mha_base`) |
|---|---|---|---|
| Graph inputs | Split the encoder features into heads | Learned Q/K/V | Learned Q/K/V |
| Score scaling | Learned temperature, initialized at 1 | Square-root head dimension | Square-root head dimension |
| Hop coefficients | Globally learned, initialized at zero | Electrode-conditioned; base initialized to one-hop | Single-hop attention |
| Hop feature projections | Separate learned projection per hop | None; value projection before propagation | Value projection before propagation |
| Output gate | Disabled | Enabled | Disabled |
| Graph nodes | 22 electrodes | 22 electrodes | 22 electrodes |

Base AGFL means the original layer defaults, explicitly saved in this preset.
Both AGFL variants use K=2, signed polynomial coefficients and scheduled top-k.
Enhanced AGFL retains the current token-contrast routing with scale 0.5; here
tokens are electrodes. The comparison estimates the effect of the complete
enhanced version, not the isolated effect of its output gate. The backbone is
the new inter-channel adaptation in all three arms, so this is not replay of
the historical temporal EEGNet accuracy.

The study declares three directed `[candidate, baseline]` comparisons in
`study.comparisons`: enhanced AGFL − base AGFL, base AGFL − base MHA, and
enhanced AGFL − base MHA. These appear in the subject-study tables and plots
for accuracy, ROC-AUC and macro F1. A03 and A04 are reported individually, then
equally averaged after averaging seeds within each subject. Two subjects
provide a focused development comparison, not an across-nine-subject claim.

If the previous inter-channel source update is already on the cluster, only
these two additional runtime files need uploading from the Mac:

```bash
cd /Users/egor/Downloads/AGFL
scp agfl/presets/eegnet-a03-a04-interchannel.json georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/presets/
scp agfl/multi_subject_study.py georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/agfl/
```

Otherwise use the full source upload above first. Use the existing allocated
GPU shell/environment, with `AGFL` as working directory and `../ml` containing
`A03T.gdf` and `A04T.gdf`. Training:

```bash
python main.py sweep --preset eegnet-a03-a04-interchannel \
  --output-dir results/eegnet-A03-A04-interchannel-v1
```

Checkpoint diagnostics and electrode plots:

```bash
python main.py diagnose-session results/eegnet-A03-A04-interchannel-v1 \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

Result plots, subject means and all three comparisons:

```bash
python main.py analyze results/eegnet-A03-A04-interchannel-v1 --plots
```

Download `results/eegnet-A03-A04-interchannel-v1/report/`. For accuracy numbers
before checkpoint diagnostics finish, run `analyze` without `--plots` and send
`report/analysis/subject_study.json`. Retraining matching runs overwrites them
by default; no `--skip-completed` is included. The preset and reporting changes
were checked statically only; neither training nor plotting was run locally.

## Earlier A03 matched-gate comparison

Training: 15 fits, three attention arms, five seeds, 250 epochs each.

```bash
python main.py sweep --preset eegnet-interchannel-comparison \
  --output-dir results/eegnet-A03-interchannel-v1
```

Checkpoint diagnostics, including electrode graphs:

```bash
python main.py diagnose-session results/eegnet-A03-interchannel-v1 \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

Result plots and accuracy tables:

```bash
python main.py analyze results/eegnet-A03-interchannel-v1 --plots
```

PCA keeps this first diagnostic pass inexpensive. `--embedding tsne` can replace
it for t-SNE; it does not change classification results. A03 is a development
subject, not independent confirmation of an across-subject claim.

## Full nine-subject attention comparison

Use after the new channel layout executes correctly. This is 315 fits; each
subject is trained separately. The full declared matrix includes both gate
ablations. It does not reuse any temporal EEG accuracy as a spatial result.

```bash
python main.py sweep --preset eegnet-interchannel-attentions \
  --output-dir results/eegnet-nine-subject-interchannel-v1
```

```bash
python main.py diagnose-session results/eegnet-nine-subject-interchannel-v1 \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

```bash
python main.py analyze results/eegnet-nine-subject-interchannel-v1 --plots
```

For an accuracy-only report, run `analyze` without `--plots`; it does not load
checkpoints or recordings. The single file to send is
`results/eegnet-nine-subject-interchannel-v1/report/analysis/subject_study.json`.
Do not wait for checkpoint diagnostics merely to inspect accuracy.

Matching runs overwrite by default, including interrupted seeds. Do not add
`--skip-completed` when intending to retrain changed code. Use fresh session
names for a changed experimental design; keep source fixed during each study.

## Files and plots

Download `results/<session>/report/`; keep `artifacts/` on the cluster.

- `report/analysis/report.md`, CSV and JSON: metrics, parameter counts, graph
  axis, seed coverage and matched comparisons.
- `report/analysis/subject_study.json` and `.md`: declared study completeness,
  subject means and subject-paired comparisons; one A03 subject alone cannot
  support a subject-paired significance test.
- `report/analysis/figures/index.md`: training curves, per-class recall,
  confusion matrices, ROC curves, accuracy/F1/AUC comparisons and subject plots.
- `report/diagnostics/<run>/index.md`: signal plots, spectra, CSP patterns,
  embeddings, labeled 22×22 attention maps, class-specific electrode maps,
  source-weight scalp maps, directed electrode connection figures, AGFL
  coefficients/hop diagnostics and output gates.
- `mixers/*_electrode_edges.csv` and corresponding `.npz`: complete directed
  weights and channel order. The connection drawing displays only the strongest
  off-diagonal edges for readability; it does not discard matrix entries.
- `electrode_gate/`: per-electrode gates and validation-only removal sensitivity.
  There are no EEG temporal-bias plots for new runs.

Every figure is PNG + PDF. For datasets without known electrode positions,
labeled matrices and edge tables remain available; scalp figures explicitly
report missing coordinates rather than inventing a montage.

## Historical compatibility and verification

Original temporal presets are preserved byte-for-byte under
`agfl/presets/archive/eeg_temporal/`. Historical model paths are reachable by
saved-checkpoint reconstruction only. `analyze` reads old metrics unchanged;
`diagnose` reconstructs the saved axis/width and compares probabilities against
saved predictions. It never turns a seven-node graph into an electrode map.
Old experiment launch instructions are archived in
`docs/archive/README-before-interchannel.md`.

The NeurIPS reference PDF itself needs a later manuscript correction: Section
4.2 interprets electrode graphs whereas Appendix C.2 describes temporal
attention after spatial collapse. This change implements the requested
inter-channel experiment; it does not rewrite the PDF or reassign old figures.

Local verification is restricted to source inspection and static syntax/JSON
checks. No project import, CLI, tests, training, inference or checkpoint
diagnostics has run locally. New numerical behavior and accuracy require the
cluster run. Optional development checks remain separate from launch commands.
