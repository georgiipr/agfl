# EEGNet attention before spatial convolution

Set `model_options.electrode_architecture=pre_spatial` to place the selected
attention between EEGNet's temporal stem and full-channel spatial convolution.
This is an explicit alternative to the default `spatial_fusion` layout.

```text
Input [B,C,T]
  shared temporal convolution + BatchNorm       [B,F1,C,T]
  electrode identity + electrode attention      [B*T,C,F1]
  reshape and optional residual                [B,F1,C,T]
  depthwise spatial convolution over C channels
  BatchNorm -> ELU -> pooling -> dropout
  separable temporal convolution -> pooling
  flattened features -> classifier
```

Each attention node contains one electrode's temporal-filter features at one
time step. Attention connects electrodes, never time steps. All classifier
features pass through the spatial convolution; there is no parallel
per-electrode classifier branch. The layout's attention width is `f1`, which
must be divisible by the chosen head count. `electrode_dim` is unused.

The current EEG defaults include kernel length 125, `f1=16`, depth multiplier
2, `f2=32`, pooling factors 8 and 16, and dropout 0.5. A 1,000-sample trial
therefore produces 224 flattened classifier features. These settings define
an architecture, not an accuracy guarantee. Attention memory use grows with
the number of trial/time-step token sets in a batch.

## General launch

Run from the repository root with the compute environment active. Replace the
relative dataset directory if needed:

```bash
python main.py run --preset eeg --model eegnet --attention agfl \
  --set data.data_dir=../ml \
  --set model_options.electrode_architecture=pre_spatial \
  --output-dir results/eegnet-pre-spatial
```

This uses the EEG preset's nine individual participants and seeds 0–4. Add
`--set 'data.subjects=[1]' --seeds 0` to restrict the run. To compare all five
attentions, use `sweep --preset eeg-comparison` with the same layout override
and a separate output directory. `plan` previews resolved settings.

Checkpoint diagnostics:

```bash
python main.py diagnose-session results/eegnet-pre-spatial \
  --device cuda --partition validation --embedding tsne
```

Result tables and plots:

```bash
python main.py analyze results/eegnet-pre-spatial --plots
```

Diagnostic displays average the per-time-step electrode graphs back to one
map per trial. Sensor-routing weights do not establish anatomical connectivity.

## Retained fixed-study launcher

`eegnet_A03_A04_A09_pre_spatial.sbatch` defines a fixed EEGNet comparison of
AGFL and MHA on A03/A04/A09 with five seeds. It is separate from the general
CLI examples above. Inspect the launcher and its preset before submission;
they specify resource requests, preflight checks, resume behavior and reporting.
Its `AGFL_VENV`, `AGFL_DATA_DIR` and `AGFL_OUTPUT_DIR` variables configure the
environment and paths without embedding an individual's machine layout.

Common environment, transfer and session instructions are in the
[README](../README.md) and [result guide](result_sessions.md).
