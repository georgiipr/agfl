# AGFL research framework

A configurable framework for comparing Adaptive Graph Filter Layer (AGFL)
attention with MHA, Performer, Linformer and Nyströmformer in EEG and ECG
classification. Choose a backbone and an attention mechanism independently;
the selected attention runs inside that backbone.

EEG attention connects electrodes. ECG attention connects time steps or time
patches. BCI Competition IV 2a participants are trained individually.

## Models and attentions

| Selection | Available keys |
|---|---|
| Backbone (`--model`) | `eegnet`, `eegencoder`, `dstseegencoder`, `conformer`, `signal_transformer` |
| Attention (`--attention`) | `agfl`, `mha`, `performer`, `linformer`, `nystromformer` |
| Dataset | `eeg`, `ecg`, `npz_eeg`, `npz_ecg` |
| Backbone settings | `model_options` |
| Attention settings | `attention_options` |

MHA is the base attention. There is no no-attention option. Each backbone has
separate EEG and ECG variants. See [model structure](docs/model_attention_structure.md)
and [attention mathematics](docs/mathematics.md).

## Environment

Use **Python 3.12 or newer**. Reuse an existing compatible environment when
available. For a fresh environment, from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Dependencies are declared in `pyproject.toml`. GPU runs require a CUDA-enabled
PyTorch installation compatible with the compute machine. For CPU execution,
add `--set device=cpu` to a training command.

Plotting uses the optional `plots` extra. If it is not already installed:

```bash
python -m pip install -e '.[plots]'
```

PCA and t-SNE use the core dependencies; UMAP additionally needs the `umap`
extra. Development tests and `pytest` are not prerequisites for launching
experiments or generating reports.

## Dataset locations

Commands below run from the repository root. The default relative layout is:

```text
workspace/
  AGFL/
  ml/
    A01T.gdf ... A09T.gdf
  mit-bih-arrhythmia-database-1.0.0/
    *.hea, *.dat, *.atr
```

Raw datasets are not distributed with the repository. Set `data.data_dir` to
your own location; paths resolve from the working directory. T-to-E EEG
evaluation also needs `A01E.gdf` through `A09E.gdf` and official `AxxE.mat`
files containing `classlabel`. See [dataset protocols](docs/data_audit.md).

## Inspect the configuration

```bash
python main.py list
python main.py presets
python main.py plan --preset eeg --model eegnet --attention agfl \
  --set 'data.subjects=[1]'
```

`plan` prints resolved settings without loading recordings or training.
Explicit preset values override defaults, so inspect the resolved configuration
when switching between presets. `--set KEY=VALUE` accepts JSON values; quote
lists and objects to prevent shell expansion.

## Train one model

EEGNet with AGFL on BCI IV 2a:

```bash
python main.py run --preset eeg --model eegnet --attention agfl \
  --set data.data_dir=../ml \
  --output-dir results/eegnet-eeg
```

This trains all nine subjects **separately**, using seeds 0–4: 45 fits. To
train one subject with one seed instead:

```bash
python main.py run --preset eeg --model eegnet --attention agfl --seeds 0 \
  --set data.data_dir=../ml --set 'data.subjects=[1]' \
  --output-dir results/eegnet-subject
```

Change `--model` or `--attention` independently. EEGNet defaults to its
`spatial_fusion` electrode layout. Add
`--set model_options.electrode_architecture=pre_spatial` to place attention
before its spatial convolution; [layout details](docs/eegnet_pre_spatial.md)
describe the difference.

Conformer with AGFL on the MIT-BIH binary ECG task:

```bash
python main.py run --preset ecg --model conformer --attention agfl \
  --set data.data_dir=../mit-bih-arrhythmia-database-1.0.0 \
  --output-dir results/conformer-ecg
```

ECG splits are grouped by patient. The task maps `N` to class 0 and
`V/A/L/R/F` to class 1; it is not the AAMI five-class benchmark.

## Compare attentions within one model

```bash
python main.py sweep --preset eeg-comparison --model eegnet \
  --set data.data_dir=../ml \
  --output-dir results/eegnet-attentions
```

This compares all five attentions on the selected backbone: nine subjects ×
five seeds × five attentions = 225 fits. Add `--seeds 0` and
`--set 'data.subjects=[1]'` for a smaller comparison. Use `ecg-comparison` for
ECG, or `eeg-ablations` / `ecg-ablations` for the supplied AGFL ablations.

For custom experiments, supply JSON or TOML with `--config experiment.json`.
A single run specifies `model`, `attention`, `dataset`, `data`, `model_options`,
`attention_options`, `training`, `seeds` and `output_dir` as needed. A sweep
contains a `base` object and an `experiments` list of overrides. Specialized
presets and fixed-study launchers are retained for reproducibility; the common
commands above do not depend on a previous experiment's output files.

## EEG session-transfer evaluation

The default `eeg` preset uses a stratified 60/20/20 split of each participant's
T-session trials. For training/validation on T and held-out testing on E:

```bash
python main.py run --preset eeg-session --model eegnet --attention agfl \
  --set data.data_dir=../ml --set data.labels_dir=../official-labels \
  --output-dir results/eegnet-session
```

Omit `data.labels_dir` when the official label files are beside the GDF files.
Missing labels are an error; the loader does not infer evaluation labels.

## Results and plots

Pass a session root to `--output-dir`, for example `results/eegnet-eeg`.
The framework separates downloadable evidence from training artifacts:

```text
results/eegnet-eeg/
  report/
    runs/              settings, splits, histories, predictions, metrics
    analysis/          summary tables and result figures
    diagnostics/       signal, embedding and attention figures
  artifacts/           canonical runs and model checkpoints
```

After training, use the same session path for the following two independent
steps. Replace `results/eegnet-eeg` if you selected another output directory.

**Checkpoint diagnostics:** needs the checkpoints and matching recordings.

```bash
python main.py diagnose-session results/eegnet-eeg \
  --device cuda --partition validation --embedding tsne
```

**Result tables and plots:** reads saved results, histories and predictions.

```bash
python main.py analyze results/eegnet-eeg --plots
```

Open `report/analysis/report.md` for tables and
`report/analysis/figures/index.md` for the PNG/PDF figures. Each completed
checkpoint diagnostic has an `index.md` under `report/diagnostics/`.
Omit `--plots` for tables only. Add `--report` to a training command to run
both reporting steps automatically after training.

Download only the session's `report/` folder. For example, replace the SSH
destination and remote path with those of your compute machine:

```bash
rsync -av --progress \
  user@compute-host:/path/to/AGFL/results/eegnet-eeg/report/ \
  ./downloaded-reports/eegnet-eeg/
```

Full details: [visualization](docs/visualization.md),
[result storage and recovery](docs/result_sessions.md),
[statistical analysis](docs/statistics.md).

## Reruns and reproducibility

`run` and `sweep` overwrite matching seed artifacts by default. Interrupted
fits restart from epoch 1; manual deletion is unnecessary. Add
`--skip-completed` to retain completed fits whose recorded identities match.
This is not an optimizer-state resume. Keep source, dependencies and dataset
content stable during a study, and use a new output directory for a changed
scientific configuration. Regenerate reports after replacing runs.

Saved records include resolved settings, source/package provenance, data
fingerprints, split IDs and checkpoint selection. Validation selects
checkpoints; test scores do not select epochs. Compare attentions within the
same backbone, preprocessing, split and training protocol. Different layouts
or preprocessing settings are separate experiments, not an attention-only
comparison.

## Cluster execution

Run training on the compute machine after activating its Python environment.
For Slurm, an interactive GPU allocation can be requested with:

```bash
salloc --partition=gpu --gpus=1 --mem=16G --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=04:00:00 srun --pty bash -l
```

Partition names and resource limits depend on the cluster. Run the common
training commands after the allocation is granted.

Alternatively, from the repository root with the environment active, queue a
common training command without a separate batch file:

```bash
sbatch --partition=gpu --gpus=1 --mem=16G --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --time=04:00:00 --output=agfl-%j.log \
  --wrap='srun python main.py run --preset eeg --model eegnet --attention agfl --set data.data_dir=../ml --output-dir results/eegnet-eeg'
```

Choose resources and a time limit appropriate for the selected experiment;
the example allocation is not an estimate of its total runtime. Generate
diagnostics and result plots after training with the separate commands above.
Keep experiment-specific `.sbatch` files outside the repository. Submit an
external launcher from the repository root, using
`sbatch /path/to/jobs/AGFL/experiment.sbatch`, so `SLURM_SUBMIT_DIR` points to
the checkout. Inspect its settings and environment overrides before submission.

## Repository structure

```text
agfl/models/          backbones and their EEG/ECG variants
agfl/attention/       independent attention implementations
agfl/datasets/        loaders, sample identities and split handling
agfl/presets/         executable experiment configurations
agfl/engine.py        shared training and evaluation
agfl/analysis.py      saved-result aggregation
agfl/visualization/   result figures and checkpoint diagnostics
tests/               development checks and independent references
docs/                reusable technical documentation
```

See the [documentation index](docs/README.md) for model, data and extension
contracts. Generated results and local experiment notes belong outside the
public documentation tree.
