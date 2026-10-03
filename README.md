# AGFL_speech

AGFL attention and two baseline attentions, tested on one dataset: the SI_Hom
imagined-speech EEG trials. The code is the AGFL project's framework (config
resolution, training engine, result sessions, statistics, diagnostics) reduced
to what this dataset needs. ECG, BCI Competition IV 2a and the generic array
adapters are removed.

Every attention runs **across the 19 electrodes** (one graph node per
electrode). Nothing here attends over time.

## Layout

The project folder and the data folder sit next to each other, on every machine:

```
<anywhere>/
  AGFL_speech/          this project
  AGFL_speech_data/     si_hom_trials.npz, si_hom_manifest.json, README.md, source/
```

The default data location is `../AGFL_speech_data` **relative to the project
folder**, not to the directory a command is launched from, so nothing has to be
edited when both folders are copied to the cluster. Only `si_hom_trials.npz` and
`si_hom_manifest.json` are needed for training; `source/` holds the authors'
MATLAB files and is only needed to rebuild the archive.

Results go to `results/<name>` under the directory the command is launched from.
Launch from the project folder.

## The dataset

| | |
|---|---|
| Trials | 2640, already cleaned and epoched by the dataset authors |
| Electrodes | 19: P7 P4 Cz Pz P3 P8 O1 O2 T8 F8 C4 F4 Fp2 Fz C3 F3 Fp1 T7 F7 |
| Samples | 500 (the first second of a 2 s epoch, 500 Hz) |
| Classes | 8: four homophone pairs, named `P1a P1b … P4a P4b` |
| Subjects | 7 (`S01`–`S07`: 431, 218, 446, 381, 253, 506, 405 trials), 9 recording blocks |
| Scaling | z-scored per trial and channel by the authors, over the 2 s epoch |

Full provenance, what was verified and what is inferred: [docs/dataset.md](docs/dataset.md).

## Environment

Python 3.12 or newer with torch, numpy, scipy, scikit-learn, mne, tqdm and
matplotlib (for plots): the AGFL project's environment. Nothing is installed;
commands run from the project folder as `python -m agfl_speech …`.

Do this first on the experiment machine. It verifies the archive against its
manifest, builds every split and runs one forward and backward pass of every
backbone/attention pair. Nothing is trained or saved.

```bash
python -m agfl_speech check
```

## Backbones, attentions, presets

| `--model` | What it is |
|---|---|
| `eegnet` | The AGFL project's EEGNet with its electrode-attention layouts (`pre_spatial`, `spatial_fusion`, `compact`). Temporal kernel 250 samples (0.5 s at 500 Hz); max-norm after every step. |
| `eegnet_original` | The dataset authors' EEGNet (temporal kernel 32, max-norm once at construction), hosting the attention between its temporal and spatial filters. |
| `signal_transformer` | One token per electrode, two pre-norm attention blocks, learned electrode readout. |

| `--attention` | What it is |
|---|---|
| `agfl` | AGFL, unchanged from the AGFL project. |
| `mha` | Standard multi-head self-attention (the base attention). |
| `hcann` | The attention module of the dataset authors' HCANN model: multi-head attention with a bias-free Q/K/V projection. In HCANN it also mixes electrode rows. |

A model is a backbone; the chosen attention runs inside it. There is no
no-attention option.

| Preset | Runs |
|---|---|
| `si-hom-attentions` | 3 backbones × 3 attentions × 5 seeds, pooled cohort, 200 epochs |
| `si-hom-eegnet`, `si-hom-eegnet-original`, `si-hom-signal-transformer` | One backbone × 3 attentions × 5 seeds, pooled |
| `si-hom-individual` | `eegnet` × 3 attentions × 7 subjects × 5 seeds, one model per subject, 500 epochs, declared subject study |
| `si-hom-authors-recipe` | `eegnet_original` × 3 attentions with the authors' training recipe and their test trials |
| `si-hom` | One run: `eegnet` + `agfl`, backbone and attention defaults |
| `si-hom-confirmatory` | Pre-declared comparison with the authors' EEGNet settings and attention: 8 arms × 20 new seeds (5–24), 60 epochs; plan in [docs/confirmatory.md](docs/confirmatory.md) |

The sweep presets that include `eegnet` use its `pre_spatial` layout; `si-hom`
uses the backbone defaults (for `eegnet`: `spatial_fusion`).
`python -m agfl_speech presets` lists them; `python -m agfl_speech list` lists models, attentions and datasets.

## Running

```bash
# Resolved settings and the launch command; loads no data
python -m agfl_speech plan --preset si-hom-attentions

# Train; a resubmission keeps completed matching fits
python -m agfl_speech sweep --preset si-hom-attentions --skip-completed
```

```bash
# Checkpoint diagnostics (electrode graphs, scalp maps, embeddings)
python -m agfl_speech diagnose-session results/si-hom-attentions \
  --device cuda --partition validation --embedding tsne
```

```bash
# Result tables, paired statistics and plots from saved results
python -m agfl_speech analyze results/si-hom-attentions --plots
```

`--model` and `--attention` retarget the single-run preset `si-hom`; the sweep
presets state their options explicitly and reject a different backbone or
attention.

Useful overrides (each changes the run identity, so give such a run its own
`--output-dir`):

```bash
--set data.cohort=individual             # one model per subject
--set data.subjects='[1,3,4]'            # a subset of subjects
--set data.classes='[0,1]'               # a two-class task (one homophone pair)
--set split.protocol=chronological       # see below
--set model_options.electrode_architecture=spatial_fusion   # eegnet-only presets: si-hom-eegnet, si-hom-individual, si-hom
--set training.epochs=300
```

`results/<name>/report/` is the part to download; `results/<name>/artifacts/`
holds checkpoints and stays on the cluster.

```bash
rsync -av --progress user@compute-host:/path/to/AGFL_speech/results/<name>/report/ \
  ./downloaded-reports/<name>/
```

## Cohorts and splits

`data.cohort`:

- `pooled` (default): one model on all selected subjects. This is the dataset
  authors' setting.
- `individual`: one experiment per subject (`subject_S01` … in the results).

`split.protocol`, always train/validation/test with checkpoint selection on
validation only:

| Protocol | Meaning |
|---|---|
| `stratified` (default) | Random 60/20/20 within every subject and class. |
| `chronological` | Per subject, the earliest 60% of trials train, the next 20% validate, the latest 20% test. Stricter against slow drifts within a recording; not class-balanced. |
| `group` | Whole subjects are held out (5/1/1 subjects). Pooled cohort only. |
| `authors` | Test = the authors' own random test trials (528); validation is carved from their training trials. |

## What to expect

- Chance is 12.5%. The authors' note reports about 22% test accuracy for a plain
  EEGNet on a pooled random 80/20 split with no validation set, keeping the
  epoch with the best *test* accuracy. Results here select the checkpoint on
  validation data and are not comparable to that number; expect them to be lower.
- With about 528 pooled test trials, one standard error is about 2 points.
  Differences of a few points between attentions need several seeds; per-subject
  test sets are 43–101 trials.
- EEGNet's pooling (8 then 16) keeps 3 time steps of a 500-sample trial; the
  last 116 samples reach the classifier only through convolution overlap. The
  authors' model does the same.

## Rebuilding the archive

Only needed if `AGFL_speech_data/si_hom_trials.npz` is lost. Requires numpy,
h5py and scipy. It never imports the project. On the Mac a rebuild reproduced
the archive byte for byte. Another numpy version may write different bytes;
the dataset fingerprint then changes and result sessions trained on the previous
file can no longer be diagnosed or resumed, so keep the original archive for as
long as its results matter.

```bash
python scripts/prepare_si_hom.py --source ../AGFL_speech_data/source
```

## Tests

Development tests live in `tests/` and need pytest. They are not a prerequisite
for training or plotting; `python -m agfl_speech check` is the pre-flight. They
use a small generated archive, not the recordings, and have not been run yet.

```bash
python -m pytest tests -q
AGFL_SPEECH_REAL_DATA=1 python -m pytest tests/test_data.py tests/test_check.py -q   # also reads the real archive
```

## More

- [docs/dataset.md](docs/dataset.md): data provenance, loader options, splits
- [docs/models.md](docs/models.md): backbones, where the attention sits, the three attentions
- [docs/results.md](docs/results.md): cluster workflow, result sessions, statistics, diagnostics
- [docs/confirmatory.md](docs/confirmatory.md): the pre-declared confirmatory experiment and its analysis
- [docs/mathematics.md](docs/mathematics.md): AGFL equations and the two baseline attentions
