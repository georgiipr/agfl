EEGNet spatial-path repair, 28 September 2026.

The A03 electrode study exposed a backbone regression: MHA fell from 85.56% to
66.30% with matching saved data, training settings and split records. Base AGFL
led the new comparison at 69.26%, but reached about 99% training accuracy. See
the [saved-report review](eegnet_a03_base_interchannel_review.md).

Two concrete architecture changes require correction. The preceding compact
adaptation replaced EEGNet's channel-spanning depthwise convolution with a 1×1
operation applied independently to each electrode. It then used ELU and pooled
each electrode before mixing them. Mixing already rectified local features does
not reproduce spatial filtering of signed signals before the nonlinearity.
It also compressed each electrode's 32 features × 7 bins to 32 dimensions, and
made that compressed route the classifier's only input.

EEGNet's authors use a full-channel depthwise spatial filter before ELU/pooling,
followed by separable temporal convolution and a flattened classifier. The repair
restores that order as a shared backbone component. The selected graph attention
remains an electrode mechanism in a parallel path. Reference:
[authors' implementation](https://github.com/vlawhern/arl-eegmodels/blob/master/EEGModels.py).
The project retains its established kernels and pooling settings; this is an
EEGNet adaptation with attention, not an exact reproduction of every published
hyperparameter.

These findings motivate the change; saved results do not isolate how much of
the accuracy loss each architectural change caused. No accuracy improvement has
yet been measured for this repair.

**Implemented computation**

For the A03 preset:

```text
Input: [B,22,1000]
    shared temporal convolution + BatchNorm, evaluated once
    [B,16,22,1000]
      ├─ depthwise spatial convolution across all 22 electrodes
      │  BatchNorm → ELU → pool 8 → dropout
      │  separable temporal convolution → BatchNorm → ELU → pool 16 → dropout
      │  [B,32,1,7] → preserve all 224 features
      │
      └─ electrode-local convolution → ELU/pooling → separable convolution
         [B,22,224] → shared local projection → [B,22,32]
         electrode identity → chosen attention, graph [B,4,22,22]
         residual → learned electrode readout → [B,32]

Concatenate 224 spatial features and 32 electrode-attention features
    [B,256] → max-norm-constrained four-class classifier
```

Attention never receives collapsed spatial features: its input nodes remain
identifiable electrodes. The other path is conventional spatial convolution,
not temporal attention. Each path has its own BatchNorm/dropout after the
shared temporal stem. The restored spatial filters receive the same max-norm
constraint as EEGNet's original spatial filters. Classifier size is computed
algebraically; construction does not run a dummy batch or update BatchNorm.

All attention choices use the same feature paths and classifier. Their
initialization is isolated from the common backbone's random-number stream.
The added spatial path and wider head add 3,264 parameters by layer-shape
calculation under this preset. Training may take longer; there is no measured
runtime or accuracy improvement yet.

This addresses the loss of EEGNet's spatial filtering and its classifier-wide
compression. Overfitting remains an outcome to measure. The new experiment keeps
the existing loss, regularization, 250-epoch budget and validation-selected
checkpoint rule so the backbone change can be assessed without a simultaneous
training-recipe change. No additional AGFL mechanism is introduced.

**Architecture selection and saved results**

New EEGNet EEG configurations default to
`model_options.electrode_architecture=spatial_fusion`. That key is saved with the
run and participates in experiment/comparison identities.

The five preceding named EEGNet electrode presets explicitly use `compact`.
They preserve their old architecture, including the existing `tune-eegnet`
candidate recipes. Saved checkpoints lacking this new selector reconstruct the
previous modules and classifier shape; old flattened-electrode and temporal
checkpoints retain their respective layouts. ECG defaults and computation remain
unchanged. New fusion checkpoints explicitly record their layout and rebuild it.

Returning to the preceding experiment is an explicit configuration choice,
`model_options.electrode_architecture=compact`, with a separate output/study name.
No archive extraction, source reset or checkpoint conversion is required.

**Next experiment**

`eegnet-a03-spatial-fusion`: A03 only, base AGFL versus base MHA, seeds 0–4,
250 epochs, ten fits. The attention settings are copied from the preceding A03
comparison. Base AGFL retains globally learned signed K=2 coefficients,
split-input graph construction, scheduled top-k, learned temperature and separate
hop projections. MHA remains standard four-head MHA. No output gate or temporal
bias is enabled in either arm.

Both use A03T, artifact exclusion, 2–30 Hz trial filtering, 1000-sample windows,
training-only channel normalization, and the same stratified 60/20/20 protocol.
Saved split identities allow checking the exact trial pairing after the run.
The completed A03 study provides the development reference; this is not an
independent confirmation of an across-subject claim.

**Upload changed runtime files from the Mac**

The pre-task local Python source hash matches the latest downloaded training
report. For a cluster checkout still at that version, only these five Python
files and six active EEGNet preset files are needed. `--relative` preserves the
subdirectories. The preset wildcard does not descend into the archive folder.

```bash
cd /Users/egor/Downloads/AGFL
rsync -avR --progress \
  agfl/models/eegnet/backbone.py agfl/models/eegnet/config.py agfl/models/eegnet/eeg.py \
  agfl/models/_shared/experiment.py agfl/models/_shared/modality.py \
  agfl/presets/eegnet-*.json \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

**Cluster setup**

Use the existing environment. Working directory:
`/beegfs/home/georgii.promyslov/AGFL`; dataset: `../ml/A03T.gdf`; output:
`results/eegnet-A03-spatial-fusion-v1`. If a GPU shell is already allocated,
continue there. Otherwise request one:

```bash
salloc -p gpu --gpus=1 --mem=10G -N 1 -n 1 -c 4 --time=04:00:00 \
  srun --pty bash -l
```

After allocation:

```bash
source ~/.venv/bin/activate
cd /beegfs/home/georgii.promyslov/AGFL
```

Training:

```bash
python main.py sweep --preset eegnet-a03-spatial-fusion \
  --output-dir results/eegnet-A03-spatial-fusion-v1
```

Checkpoint diagnostics, including electrode maps and AGFL coefficients:

```bash
python main.py diagnose-session results/eegnet-A03-spatial-fusion-v1 \
  --device cuda --partition validation --max-samples 64 --embedding pca
```

Result plots and accuracy tables:

```bash
python main.py analyze results/eegnet-A03-spatial-fusion-v1 --plots
```

Download the report from a **Mac terminal**, preserving this session's name:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/eegnet-A03-spatial-fusion-v1/report/ \
  /Users/egor/Downloads/eegnet-A03-spatial-fusion-v1/
```

For accuracy before checkpoint diagnostics complete, run `analyze` without
`--plots` and download only the study summary:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/eegnet-A03-spatial-fusion-v1/report/analysis/subject_study.json \
  /Users/egor/Downloads/eegnet-A03-spatial-fusion-v1-subject_study.json
```

No `--skip-completed` is included. Matching runs overwrite on relaunch. Keep
code fixed during a session and use a new session name for another design.

**Verification and interpretation**

Local checks are limited to source review, Python parsing/compilation without
execution, JSON/configuration inspection, and documentation links. No project
imports, CLI, training, inference, diagnostics or tests were executed locally.

Optional development regression checks are supplied in
`tests/test_eegnet_spatial_fusion.py` using standard-library `unittest` and the
already-required PyTorch. They cover gradient paths, actual attention influence,
electrode diagnostic shapes, common initialization, max norms, old/new checkpoint
reconstruction and the ten-fit preset. They are not a launch prerequisite and
have not been executed here.

Evaluate both absolute validation/test scores and AGFL-minus-MHA differences.
Improving both mechanisms would support the backbone repair; it would not by
itself prove that AGFL is superior. More parameters and a new feature path are
explicit architectural changes, and reaching the old 85% or the desired 90%
remains unverified.
