Signal Transformer comparison on A03, A04 and A09
================================================

Upload the single
[signal_transformer_A03_A04_A09_attentions.sbatch](../signal_transformer_A03_A04_A09_attentions.sbatch)
file to the existing updated cluster checkout. The file includes the experiment
settings and runner; it uses the installed project and existing environment.
It lives outside `agfl/`, so uploading it preserves the source hash of the
earlier studies. The experiment has its own output session.

This is the repository's **generic Signal Transformer**, model key
`signal_transformer`. A shared temporal convolution encodes each electrode
independently, followed by two attention/feed-forward blocks across electrodes
and a learned spatial readout. It is a custom backbone; no published EEG-model
accuracy is being attributed to this implementation.

The fixed matrix is **45 fits**:

- **Base AGFL, base MHA, Performer with 64 random features**.
- **A03, A04, A09**, trained individually with seeds **0–4**.
- Feature dimension 64, **two blocks**, four heads, dropout 0.1, MLP ratio 4.
- **22 electrode tokens** in every attention layer; no temporal attention.
- AGFL K=2 with the original static signed polynomial coefficients, scheduled
  top-k, split-input projection and separate hop projections. No output gate
  or temporal bias is enabled.
- **250 epochs**, batch 64, AdamW learning rate 0.005, weight decay 0.001,
  class-balanced focal loss with gamma 3, warmup cosine schedule, deterministic
  training and AMP disabled. Validation accuracy selects the tested weights.
- Same data/preprocessing recipe as the Conformer experiment: subject T-session,
  artifact exclusion, trial-level 2–30 Hz filtering, 1000 samples per trial,
  training-only channel normalization, matched 60/20/20 stratified splits.

The attention set is retained from the preceding experiment. Linformer and
Nyströmformer were excluded based on the earlier EEGNet/A03 ranking, which is
recorded in the saved plan; this is not a Signal Transformer ranking. Training
settings are held fixed for this comparison, not claimed to be optimal for the
new backbone. There is no guarantee of an accuracy increase.

Upload from the Mac:

```bash
rsync -av --progress \
  /Users/egor/Downloads/AGFL/signal_transformer_A03_A04_A09_attentions.sbatch \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/
```

Submit on the cluster:

```bash
cd /beegfs/home/georgii.promyslov/AGFL
sbatch signal_transformer_A03_A04_A09_attentions.sbatch
```

Resources: one GPU, four CPU cores, 16 GB host RAM, **24-hour allocation**.
The limit is not a runtime estimate. Environment: existing `~/.venv`, Python
3.12 or newer. Data: `../ml/A03T.gdf`, `../ml/A04T.gdf`, `../ml/A09T.gdf`.
Overrides: `AGFL_DATA_DIR`, `AGFL_OUTPUT_DIR`, `AGFL_VENV`,
`AGFL_PYTHON_MODULE`.

Monitor the latest log from the cluster AGFL directory:

```bash
tail -n 30 "$(ls -t agfl-signal-transformer-*.log | head -n 1)"
```

Output: `results/signal_transformer-A03-A04-A09-attentions-v1`.
The cluster preflight verifies each attention block's `[batch,22,64]` inputs
and outputs, finite four-class logits, and a deterministic backward pass.
Completed runs must also satisfy the two-block electrode contract.

The main result is **`report/accuracy_table.csv`**, updated after each fit:
three subject rows and an `AVERAGE` row, three attention accuracies, AGFL's
differences from MHA/Performer, and completion counts. Each subject averages
five seeds; the overall mean weights the three subjects equally. Incomplete
groups have blank means. Individual seed scores remain available in
`accuracy_by_seed.csv`, and `accuracy_table.md` is printed at completion.

Download the main table from the Mac:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/signal_transformer-A03-A04-A09-attentions-v1/report/accuracy_table.csv \
  /Users/egor/Downloads/Signal_transformer_spatial_accuracy.csv
```

Optional individual seed scores:

```bash
rsync -av --progress \
  georgii.promyslov@10.5.1.1:/trinity/home/georgii.promyslov/AGFL/results/signal_transformer-A03-A04-A09-attentions-v1/report/accuracy_by_seed.csv \
  /Users/egor/Downloads/Signal_transformer_spatial_accuracy_by_seed.csv
```

**No plots or checkpoint files** are generated. Validation-selected weights
stay in memory until test evaluation. Small run records retain histories,
predictions, configurations, splits and provenance on the cluster for review.

If interrupted, resubmit the **same file and command after the preceding job
ends**. Completed matching fits are retained; unfinished fits restart at epoch
1. A session lock prevents concurrent writers. Resuming requires the same
saved plan, launcher, framework source, numerical package versions and fit
configurations. Data fingerprints and subject/seed splits must match. Changes
require restoration of the original files/environment or a new output session:

```bash
AGFL_OUTPUT_DIR=results/signal_transformer-A03-A04-A09-attentions-v2 \
  sbatch signal_transformer_A03_A04_A09_attentions.sbatch
```

The runner does not import scores from earlier model experiments. Failed fits
are recorded in `accuracy_status.json`; the final exit is nonzero if the matrix
is incomplete. A timeout can leave status `running`, but completed artifacts
remain resumable. These development-subject results should be reported with
their model/attention selection history, including unfavorable experiments.

Local checks cover shell/Python syntax, static configuration consistency and
unchanged framework source. No project code, training or model preflight has
been executed locally; numerical correctness and accuracy require the cluster.
