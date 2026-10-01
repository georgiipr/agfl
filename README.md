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

MHA is the base attention. Each backbone has separate EEG and ECG variants.

## Environment

Use **Python 3.12 or newer**. Run the commands in this guide from the
repository root. Create and activate a virtual environment, then install the
package and plotting dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[plots]'
```

For training without figures, `python -m pip install -e .` is sufficient.
The supplied presets use CUDA and require a compatible GPU and CUDA-enabled
PyTorch installation. For CPU execution, add `--set device=cpu` to a training
command and use `--device cpu` for checkpoint diagnostics.

## Download BCI Competition IV 2a

1. Open the [official competition page](https://www.bbci.de/competition/iv/#download),
   review the dataset attribution terms and agree to them.
2. In the [download area](https://bbci.de/competition/iv/download/), choose
   **Data sets 2a → GDF files zipped** (about 420 MB). The archive is
   [`BCICIV_2a_gdf.zip`](https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip).
3. Put the extracted `.gdf` recordings directly in a folder named `ml`
   **beside the AGFL checkout**. From the repository root, replace the archive
   path below with the location of your downloaded file:

```bash
mkdir -p ../ml
unzip -j /path/to/BCICIV_2a_gdf.zip '*.gdf' -d ../ml
```

The `-j` option removes archive subdirectories so the loader can find
`../ml/A01T.gdf` directly. The default EEG experiments need `A01T.gdf` through
`A09T.gdf`; a single-subject run needs only that subject's recording.

For T-to-E session-transfer evaluation, also keep `A01E.gdf` through
`A09E.gdf`. Download **Data sets 2a → labels** from the official
[evaluation-labels page](https://www.bbci.de/competition/iv/results/#labels)
([`true_labels.zip`](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip))
and extract the evaluation label files into the same `ml` folder:

```bash
unzip -j /path/to/true_labels.zip '*E.mat' -d ../ml
```

These `A01E.mat` through `A09E.mat` files must contain the official `classlabel`
variable. The loader reads signals from GDF recordings and evaluation labels
from these MAT files. T-only experiments do not need the evaluation labels.

### Dataset locations

Commands below run from the repository root. The default relative layout is:

```text
workspace/
  AGFL/
  ml/
    A01T.gdf ... A09T.gdf    required for all-subject T-session experiments
    A01E.gdf ... A09E.gdf    also required for T-to-E evaluation
    A01E.mat ... A09E.mat    official labels for T-to-E evaluation
  mit-bih-arrhythmia-database-1.0.0/
    *.hea, *.dat, *.atr
```

Raw datasets are not distributed with the repository. Set `data.data_dir` to
your own location; paths resolve from the working directory. With this layout,
use `--set data.data_dir=../ml`. If labels are stored separately, also set
`--set data.labels_dir=../official-labels` to their directory.

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

Change `--model` or `--attention` independently. Use `--set KEY=VALUE` for
other settings, for example `--set training.epochs=100`.

For ECG, place the MIT-BIH Arrhythmia Database `.hea`, `.dat` and `.atr` files
in `../mit-bih-arrhythmia-database-1.0.0`, then launch Conformer with AGFL:

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
ECG.

For custom experiments, supply JSON or TOML with `--config experiment.json`.
A single run specifies `model`, `attention`, `dataset`, `data`, `model_options`,
`attention_options`, `training`, `seeds` and `output_dir` as needed. A sweep
contains a `base` object and an `experiments` list of overrides.

## EEG session-transfer evaluation

The default `eeg` preset uses a stratified 60/20/20 split of each participant's
T-session trials. For training/validation on T and held-out testing on E:

```bash
python main.py run --preset eeg-session --model eegnet --attention agfl \
  --set data.data_dir=../ml \
  --output-dir results/eegnet-session
```

This uses the evaluation labels placed beside the GDF files in the download
instructions. If labels are elsewhere, add `--set data.labels_dir=/path/to/labels`.

## Results and plots

Pass a session root to `--output-dir`, for example `results/eegnet-eeg`.
Results are organized as follows:

```text
results/eegnet-eeg/
  report/
    runs/              settings, splits, histories, predictions, metrics
    analysis/          summary tables and result figures
    diagnostics/       signal, embedding and attention figures
  artifacts/           training runs and model checkpoints
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

## Rerunning an experiment

`run` and `sweep` overwrite matching seed artifacts by default. Interrupted
fits restart from epoch 1; manual deletion is unnecessary. Add
`--skip-completed` to retain completed fits whose recorded identities match.
This does not resume unfinished optimizer states. Use a new output directory
to keep results from different configurations. Regenerate reports after
replacing runs.
