# EEG electrode attention

New EEG experiments use one attention node per electrode. For C channels,
attention consumes `[batch, C, features]` and builds C-by-C channel interactions.
Temporal convolution extracts each electrode's features; time bins are not
EEG attention nodes. ECG retains temporal attention.

Every EEG backbone preserves electrode identity until its attention branch.
Configuration and model checks reject temporal EEG training. EEGNet's
`pre_spatial` layout applies the electrode graph separately at each time step;
its extra time dimension is folded into the attention batch, not the node axis.

## Choose a backbone and mechanism

From the repository root, with the compute environment active and GDF files in
`../ml`, this launches one EEGNet participant/seed with AGFL:

```bash
python main.py run --preset eeg --model eegnet --attention agfl --seeds 0 \
  --set data.data_dir=../ml --set 'data.subjects=[1]' \
  --output-dir results/eeg-electrodes
```

Change `--model` and `--attention` independently. Omit the subjects and seeds
overrides to use all nine participants separately and five seeds. EEGNet's
default is `spatial_fusion`; an explicit
`--set model_options.electrode_architecture=pre_spatial` selects the
[pre-spatial layout](eegnet_pre_spatial.md). These layouts have different
architectures and must be treated as separate configurations.

## Plot the selected runs

Checkpoint diagnostics:

```bash
python main.py diagnose-session results/eeg-electrodes \
  --device cuda --partition validation --embedding tsne
```

Saved-result plots:

```bash
python main.py analyze results/eeg-electrodes --plots
```

Diagnostics label channel matrices with physical channel names when available.
Electrode scalp plots additionally require known coordinates. AGFL hop
coefficients and optional output gates index electrodes in these runs.
For `pre_spatial`, diagnostic graph displays average per-time-step maps into
one map per trial; graph statistics use the documented underlying maps.

These are learned sensor-level routing weights, not anatomical or causal brain
connectivity. Preserving node identity does not make every attention's effective
mixing matrix a probability distribution. See [visualization](visualization.md)
for graph interpretation and [mathematics](mathematics.md) for mechanism details.
