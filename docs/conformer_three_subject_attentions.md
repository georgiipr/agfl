Conformer comparison on A03, A04 and A09
======================================

Upload the single [conformer_A03_A04_A09_attentions.sbatch](../conformer_A03_A04_A09_attentions.sbatch)
file to the existing updated AGFL checkout. It contains the complete experiment
recipe and runner. No other source update or package installation is required
for the checkout already running the five-model spatial accuracy study.
The launcher lives outside `agfl/`, so uploading it does not change the source
hash pinned by that running study. It creates a separate session.

The fixed comparison contains **45 fits**:

- Model: the repository's Conformer-style electrode adaptation, dimension 64,
  three blocks, four attention heads, dropout 0.3.
- Attentions: **base AGFL**, **base MHA**, **Performer with 64 random features**.
  AGFL uses K=2, static signed coefficients, the original scheduled top-k,
  split-input projections and separate hop projections. No output gate or
  temporal bias is enabled.
- Subjects: **A03, A04, A09**, each trained independently, seeds **0–4**.
- Training: **250 epochs**, batch 64, AdamW learning rate 0.005, weight decay
  0.001, class-balanced focal loss with gamma 3, warmup cosine schedule,
  deterministic training, AMP disabled. Validation accuracy selects weights.
- Data: each subject's BCI IV2a **T-session**, artifact exclusion, trial-level
  2–30 Hz filtering, 1000 samples per trial and training-only channel
  normalization. Each seed uses the same 60/20/20 stratified split across all
  three attention arms. Test scores do not select epochs or settings.

Linformer and Nyströmformer are omitted based on the prior **EEGNet/A03**
five-seed report: respectively **63.70%** and **54.07%**, versus Performer
65.19%, MHA 66.30% and AGFL 69.26%. This is a declared selection from previous
development results, not a measured Conformer ranking. The selection basis is
saved in `report/accuracy_plan.json`. The requested three-subject experiment
does not constitute independent confirmation across all nine subjects.

All three Conformer attention blocks must operate on **22 electrodes**.
The cluster preflight checks actual `[batch,22,64]` inputs and outputs for every
block, finite four-class logits, and a deterministic backward pass for every
attention type. Actual completed runs are checked again for spatial metadata.
This is a custom electrode adaptation; describe it explicitly in the article.

Upload from the Mac:

```bash
rsync -av --progress \
  /Users/egor/Downloads/AGFL/conformer_A03_A04_A09_attentions.sbatch \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

Submit on the cluster:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
sbatch conformer_A03_A04_A09_attentions.sbatch
```

The job requests one GPU, four CPU cores, 16 GB host RAM and a **24-hour
allocation**. This is a limit, not a measured runtime. It uses the existing
`~/.venv` with Python 3.12 or newer. The recordings are `../ml/A03T.gdf`,
`../ml/A04T.gdf`, `../ml/A09T.gdf`. The log is `agfl-conformer-<jobid>.log`.
Optional environment overrides are `AGFL_VENV`, `AGFL_PYTHON_MODULE`,
`AGFL_DATA_DIR` and `AGFL_OUTPUT_DIR`.

Output: `results/conformer-A03-A04-A09-attentions-v1`.
**No plots or checkpoint files** are produced. Selected weights are kept in
memory until test evaluation. Small configurations, histories, split records,
predictions and source/package provenance remain on the cluster for auditing.

The requested result is **`report/accuracy_table.csv`**: three subject rows and
an `AVERAGE` row, with accuracy for each attention, AGFL's differences against
MHA/Performer in percentage points, and completed/expected fit counts. Each
subject score averages five seeds; the overall score weights the three
subjects equally. Incomplete seed groups have blank means, never zero scores.
There is also a readable `accuracy_table.md` and a small `accuracy_by_seed.csv`
with all individual scores. Tables update after each fit; the readable table
is printed in the log when the job finishes.

Download the accuracy table from the Mac, including while the job is running:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/conformer-A03-A04-A09-attentions-v1/report/accuracy_table.csv \
  /Users/egor/Downloads/Conformer_spatial_accuracy.csv
```

Download the individual seed scores when needed for further analysis:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/conformer-A03-A04-A09-attentions-v1/report/accuracy_by_seed.csv \
  /Users/egor/Downloads/Conformer_spatial_accuracy_by_seed.csv
```

Continuation after interruption
-----------------------------

After the preceding job ends, submit the **same file and command** again.
Completed matching fits are retained automatically; an unfinished fit restarts
at epoch 1 because this experiment does not save checkpoints. A lock prevents
two jobs from writing the same session concurrently.

Reusing scores requires the saved plan, launcher hash, source hash, numerical
package versions, configurations, complete run records, data identities and
splits to match. New fits must match the original dataset fingerprint for each
subject and the original split for each subject/seed. Changing the model or
recipe cannot silently reuse its old scores: the launcher refuses and asks for
a new output session or restoration of the original source/settings.

For an explicitly new session, set a new output folder before submitting:

```bash
AGFL_OUTPUT_DIR=results/conformer-A03-A04-A09-attentions-v2 \
  sbatch conformer_A03_A04_A09_attentions.sbatch
```

The launcher does not scan or import scores from the earlier five-model study.
Failures are recorded in `report/accuracy_status.json`; later cells continue,
and an incomplete matrix causes a nonzero job exit. A timeout may leave the
last saved status as `running`; completed artifacts remain resumable. Keep
project source and the environment fixed for the duration of this study.

Local verification is limited to static Python/shell/configuration checks.
The model preflight, training, accuracy and wall-clock runtime require the
cluster; they have not been executed locally.
