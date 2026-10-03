# Running on the cluster and reading the results

Nothing in this project has been executed yet. The first cluster run is the
real check; start with `check`.

## Cluster workflow

Both folders go next to each other on the cluster. Only two data files are
needed there.

Upload (from the Mac, in the folder that contains both folders):

```bash
rsync -av --progress --exclude '__pycache__' --exclude 'results' --exclude 'splits' \
  AGFL_speech user@compute-host:/path/to/
rsync -av --progress --exclude 'source' \
  AGFL_speech_data user@compute-host:/path/to/
```

Everything below runs in `/path/to/AGFL_speech` with the Python 3.12+
environment of the AGFL project active.

Pre-flight (a few minutes; trains nothing, saves nothing):

```bash
python -m agfl_speech check --device cuda
```

It verifies the archive against its manifest and against the released trial and
subject counts, builds every split protocol for the pooled cohort and for each
subject, and runs one forward and one backward pass of every backbone (all three
EEGNet layouts) with every attention on 16 real trials.

Preview (loads no data):

```bash
python -m agfl_speech plan --preset si-hom-attentions
```

Training:

```bash
python -m agfl_speech sweep --preset si-hom-attentions \
  --output-dir results/si-hom-attentions-v1 --skip-completed
```

Checkpoint diagnostics:

```bash
python -m agfl_speech diagnose-session results/si-hom-attentions-v1 \
  --device cuda --partition validation --embedding tsne
```

Result tables and plots:

```bash
python -m agfl_speech analyze results/si-hom-attentions-v1 --plots
```

Download the report (to the Mac):

```bash
rsync -av --progress \
  user@compute-host:/path/to/AGFL_speech/results/si-hom-attentions-v1/report/ \
  ./downloaded-reports/si-hom-attentions-v1/
```

Adding `--report` to `run` or `sweep` runs the diagnostics and the analysis
automatically after training. Batch launchers are kept outside the repository
(see `AGENTS.md`).

## Result sessions

Pass the session root to `--output-dir`, not its `report` or `artifacts`
subfolder.

```text
results/si-hom-attentions-v1/
  report/                       small files; download this
    runs/                       configurations, splits, histories, predictions, metrics
    analysis/                   tables and result figures
    diagnostics/                checkpoint figures
  artifacts/                    checkpoints and working files; stays on the cluster
    si_hom/
      <model>-<attention>-<experiment id>/seed_<n>/          pooled cohort
      subject_S03/<model>-<attention>-<experiment id>/seed_<n>/   individual cohort
```

Each run folder holds `config.json` (fully resolved settings, data fingerprint,
source and package provenance), `split.json`, `history.json`, `checkpoint.pt`,
`predictions.npz` and `result.json`; `failure.json` appears if a run failed.

A session folder may contain only `artifacts/`, `report/` and lock files. Keep
the data folder and the `splits/` folder outside it.

## Identities and resubmission

- `experiment_id`: hash of the resolved configuration (without seeds and output
  locations). It names the run folder.
- `comparison_id`: hash of everything that must be equal for a fair attention
  comparison: data settings, split settings, training settings, backbone and its
  options, head count, device, thread count, data fingerprint, source code hash
  and package versions.
- `split_id`: hash of the data fingerprint, the seed and the split settings.
  It does not depend on the model or attention, so all arms share splits.

With the default relative `data_dir` these are identical on every machine.
A different device spelling (`cuda` versus `cuda:0`), thread count, explicit
data folder or any change to the source files changes them.

`run` and `sweep` replace matching seed runs by default. `--skip-completed` keeps
completed fits whose saved identities match and restarts unfinished ones from
epoch 1; there is no mid-run resume. Use a new `--output-dir` when the source,
data or settings change.

## What `analyze` writes

`report/analysis/report.md` and CSV files:

| File | Content |
|---|---|
| `per_model.csv` | One row per experiment: mean and sample standard deviation over seeds of accuracy, macro ROC-AUC and macro F1, validation and test |
| `attention_comparison.csv` | The same rows for `agfl`, `mha`, `hcann` |
| `statistical_comparisons.csv` | AGFL minus baseline on identical seeds and splits: mean difference, paired t-test, Wilcoxon signed-rank test, effect sizes, Holm-adjusted p-values |
| `per_class.csv`, `per_class_summary.csv` | Recall, precision and F1 per class |
| `cohort_test_by_subject.csv` | Pooled runs only: test metrics of each subject inside the pooled model, averaged over seeds |
| `per_subject.csv`, `across_subjects.csv` | Individual cohort only: per-subject results, and equal-weight means over subjects |
| `subject_study.*` | Individual cohort with a declared study (`si-hom-individual`): complete subject × arm × seed table, subject-paired comparisons |
| `aggregation.json` | Everything above with full settings and the source files used |

How to read the statistics:

- AGFL is paired with `mha` and with `hcann` only inside the same backbone and
  only when `comparison_id`, seed, split and data fingerprint match. Unmatched
  runs are counted and listed, not dropped silently.
- One Holm family contains every comparison and metric in the session. Analyze
  backbones in separate sessions if you want separate families.
- Seeds are not independent subjects. With five seeds a two-sided Wilcoxon test
  cannot go below p = 0.0625. Seed-level tests are exploratory. The subject
  study of the individual cohort uses the seven subjects as units (smallest
  possible Wilcoxon p = 0.0156).
- Undefined ROC-AUC (a class missing from a held-out partition, possible with
  `chronological` and `group`) stays empty; it is never counted as zero.

## What the diagnostics show

`diagnose` and `diagnose-session` reload a checkpoint and the data, check both
against the saved identities, and write figures for a class-balanced subset of
one partition (validation by default, 256 trials at most):

- example trials and Welch spectra of the model inputs;
- alpha and beta band maps on the scalp, CSP patterns fitted on training trials;
- an embedding of the features entering the classifier (PCA, t-SNE or UMAP);
- for every attention layer: the mean 19 × 19 electrode graph per head, per-class
  graphs, sparsity and entropy, a directed edge table (`*_electrode_edges.csv`)
  and the strongest connections drawn on the scalp;
- for AGFL: hop coefficients, temperatures and filter projections;
- a check that the reloaded model reproduces the saved predictions.

Read them with these limits in mind:

- Inputs are z-scored per trial and channel, so band maps show relative band
  share, not voltage.
- In a pooled run the subset mixes subjects; graphs are cohort-level routing,
  not one person's connectivity. Routing weights are model associations between
  sensors, not anatomical connectivity.
- In `pre_spatial` layouts one graph exists per time step; the figures show the
  average over the 500 time steps of each trial.
- If the data folder moved since training, pass `--data-dir`; the content must
  be unchanged.
