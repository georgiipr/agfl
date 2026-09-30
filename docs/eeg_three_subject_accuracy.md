Queued EEG accuracy comparison for A03, A04 and A09.

The launch file is [eeg_A03_A04_A09_accuracy.sbatch](../eeg_A03_A04_A09_accuracy.sbatch).
Submit it from the updated AGFL checkout. It requests one GPU, four CPU cores,
16 GB host RAM and a 24-hour wall-clock limit. This is a resource limit, not a
measured runtime estimate. Existing environment: `~/.venv`, Python 3.12 or newer.
No package installation is performed.

The fixed matrix contains 150 fits:

| Model | EEG attention nodes | Heads | Base AGFL hops | Fits |
|---|---|---:|---:|---:|
| EEGNet, spatial-fusion repair | 22 electrodes | 4 | 2 | 30 |
| EEGEncoder | 22 electrodes | 4 | 3 | 30 |
| DSTS EEGEncoder | 22 electrodes in every branch/layer | 2 | 1 | 30 |
| Conformer | 22 electrodes in every layer | 4 | 2 | 30 |
| Signal Transformer | 22 electrodes in every layer | 4 | 2 | 30 |

Every model compares base AGFL with base MHA, using seeds 0–4 and 250 epochs.
MHA is the second-ranked method in the latest A03 EEGNet report. It is a fixed
comparator here; no new test scores select a different comparator by model.
Head/hop counts retain the model-specific defaults. AGFL retains its original
signed polynomial formulation, split-input graph, scheduled top-k, static
learnable coefficients, learned temperatures and separate hop projections.
Neither mechanism has an output gate or temporal bias enabled.

Each subject trains independently on its T-session. Data, normalization,
60/20/20 split protocol, loss, learning-rate schedule, and epoch budget are
matched across models/attentions. The table first averages five seeds within
each subject and then averages A03, A04 and A09 equally. A04's different trial
count does not reduce its weight. There is a separate mean for each model;
models are not pooled into a single headline score.

The cluster preflight checks real attention input/output shapes using a small
synthetic batch for every model/mechanism. Every insertion must consume
`[batch,22,features]`. Construction also checks graph metadata; completed runs
must report electrode nodes. Temporal convolutions and per-electrode TCNs encode
waveforms within each sensor and do not create temporal attention graphs.

**No plots or checkpoint files**

`training.save_checkpoints=false` keeps a cloned copy of the validation-selected
weights in CPU memory. The selected epoch is restored before test evaluation;
the final training epoch is not substituted. No `.pt` file is written, and the
job does not invoke diagnostics, plotting or the development test suite.

Small configurations, split records, histories, predictions, source/package
provenance and failure details remain available for auditing. For review,
download only `report/accuracy_table.csv` (or its readable `.md` counterpart).
The CSV includes each subject's score, both three-subject means, their difference
in percentage points, and completed/expected fit counts.

The table updates after every fit. It uses only results returned during this
invocation, so an interrupted rerun cannot inherit old completion counts.
Incomplete groups have blank CSV means / `pending` Markdown means. A failed fit
is recorded, other fits proceed, and the job ultimately exits unsuccessfully
if the declared matrix is incomplete. New submissions retrain all 150 fits;
no completed scores are silently reused by that fresh-launch file. To continue
an interrupted run, use the separate continuation file below. Source/package
changes during the job fail provenance checks. Keep source fixed until the
entire study, including continuations, finishes.

**Continue after the time limit**

The original 24-hour allocation is a limit, not a benchmark. The observed
42 completed fits at about 13 hours suggest roughly 46 hours for 150 fits at
that average pace. Remaining architectures can be faster or slower.

Use [eeg_A03_A04_A09_continue.sbatch](../eeg_A03_A04_A09_continue.sbatch)
to continue the same study in another 24-hour allocation. This is a single
standalone job file: upload only it while the current job runs. It makes no
changes under `agfl/`, so uploading it does not invalidate the active job's
source hash. Do not re-upload model or engine files during this study.

The continuation reads the original `report/accuracy_plan.json`, checks source
and package versions, and validates saved configurations, data/split identities,
complete run artifacts and spatial metadata before retaining a score. New fits
must use the same data and splits as saved fits. A changed implementation or
configuration is rejected; this is not reuse of old scores after a model update.
The table includes previously completed fits throughout the continuation.
An unfinished fit restarts from epoch 1 because no checkpoints are retained.
The original study lock prevents simultaneous writers.

Upload from the Mac:

```bash
rsync -av --progress \
  /Users/egor/Downloads/AGFL/eeg_A03_A04_A09_continue.sbatch \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

On the cluster, find the currently running study's job ID:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
squeue -u "$USER" -o "%.18i %.24j %.10T %.12M %.12l"
```

Replace `JOB_ID` below with that job's numeric ID and submit this while it is
still running. `afterany` releases the continuation when the original job ends,
including when it times out; it then waits for available resources. See the
[Slurm dependency documentation](https://slurm.schedmd.com/sbatch.html#OPT_dependency).

```bash
sbatch --dependency=afterany:JOB_ID eeg_A03_A04_A09_continue.sbatch
```

If the previous job has already ended and no continuation is queued/running,
submit without a dependency:

```bash
sbatch eeg_A03_A04_A09_continue.sbatch
```

If another 24 hours proves insufficient, submit the same continuation file
again after its job ends. Each submission retains all completed matching fits.
The output remains `results/eeg-A03-A04-A09-spatial-accuracy-v1/report/accuracy_table.csv`.
The continuation uses saved dataset paths (normally `../ml` resolved on the
cluster), the existing `~/.venv`, and the same resources and training settings.
`AGFL_OUTPUT_DIR`, `AGFL_VENV` and `AGFL_PYTHON_MODULE` overrides are supported.
There are still no plots, checkpoint files, package installations or test runs.

**Upload from the Mac**

This includes the preceding EEGNet repair if it has not yet been uploaded,
the checkpoint-storage change, the runner, its preset and the one job file.
All transfers are ordinary files with their relative paths preserved.

```bash
cd /Users/egor/Downloads/AGFL
rsync -avR --progress \
  agfl/config.py agfl/engine.py agfl/accuracy_study.py \
  agfl/models/eegnet/backbone.py agfl/models/eegnet/config.py agfl/models/eegnet/eeg.py \
  agfl/models/_shared/experiment.py agfl/models/_shared/modality.py \
  agfl/presets/eegnet-*.json agfl/presets/eeg-three-subject-spatial-accuracy.json \
  eeg_A03_A04_A09_accuracy.sbatch \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

**Submit on the cluster**

Working directory: `AGFL`. Data: `../ml/A03T.gdf`, `../ml/A04T.gdf`,
`../ml/A09T.gdf`. Output: `results/eeg-A03-A04-A09-spatial-accuracy-v1`.

```bash
cd /beegfs/home/georgii.promyslov/AGFL
sbatch eeg_A03_A04_A09_accuracy.sbatch
```

Slurm returns a job ID and starts the batch job when the allocation is granted;
the login session does not need to stay open. See the
[Slurm sbatch documentation](https://slurm.schedmd.com/sbatch.html).
The log is `agfl-spatial-accuracy-<jobid>.log` in the submission directory.
Optional environment overrides are `AGFL_VENV`, `AGFL_DATA_DIR`,
`AGFL_OUTPUT_DIR` and `AGFL_PYTHON_MODULE`; none are required for the established
directory/environment layout.

**Download the final table from the Mac**

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/eeg-A03-A04-A09-spatial-accuracy-v1/report/accuracy_table.csv \
  /Users/egor/Downloads/AGFL_spatial_accuracy.csv
```

If a job fails, `report/accuracy_status.json` identifies failed cells and
`report/spatial_checks.json` records the successful model preflight checks.
Do not interpret a partial table as the requested complete comparison.

Local verification is static only: Python syntax/compilation without execution,
JSON matrix checks, shell syntax, source review and documentation links. Cluster
preflight, training and optional regression checks have not run locally.
No accuracy or runtime improvement is assumed. This is a development comparison
on three named subjects, not an independent across-nine-subject result.
