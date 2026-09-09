# Experiment figures

Plotting is an offline operation. It does not run during training, choose a new
checkpoint, or overwrite saved experiment metrics. Commands below are for the
intended machine; the plotting implementation and supplied tests have not been
executed locally.

Training reruns now replace matching seed artifacts by default. Rerun analysis
and checkpoint diagnostics afterwards to refresh exported figures; training does
not regenerate them automatically.

The README contains the full workflow, dependency installation, result figures,
every-subject/every-seed diagnostics, output indexes and diagnostic comparisons.
BCI runs now live under `eeg/subject_Axx/`; each subject trains separately. Old
cohort results keep their original labels.

## Figures from completed runs

Install the plotting extra on the analysis/experiment machine if needed:

```bash
python -m pip install -e '.[plots]'
python main.py analyze results/full-eeg --output-dir analysis/full-eeg --plots
```

This command needs saved results, histories, predictions, and split manifests.
It does not load recordings or model checkpoints. The generated figures include:

| Figure | Source and interpretation |
|---|---|
| Learning curves | Training/validation loss, validation Accuracy/AUC/F1, LR, selected checkpoint epoch; no test-based epoch selection. |
| Confusion matrices | Validation and test, counts and row-normalized values, for each seed. Absent true classes are labeled N/A. |
| ROC curves | One-vs-rest curves per seed, with class-specific AUC. Undefined curves are listed in the manifest. |
| Model comparisons | Individual seed points plus mean and sample SD for Accuracy/AUC/F1. Error bars are not confidence intervals. |
| Paired differences | AGFL minus baseline for identical seed, split, data, and comparison identities. |
| AGFL ablations | Separate configurations labeled with experiment IDs; options remain in aggregation.json and CSVs. |
| Capacity comparisons | Parameter count versus scores with across-seed sample SD. |
| Subject scores | Each held-out subject's metrics within its run, where saved. |

Each figure is exported as PNG and vector PDF. Open
`analysis/full-eeg/figures/index.md` for previews. `manifest.json` identifies the
source runs, generated files, and unavailable figures. Runs with different
comparison protocols are drawn separately. Test predictions from repeated,
overlapping subject splits are never pooled into one apparent independent cohort.
Prediction order must match the split manifest and recomputed metrics must match
the saved result; disagreement is an error.

## Signals, embeddings, and internal mixer diagnostics

Choose one seed directory containing `config.json`, `split.json`, and a trusted
`checkpoint.pt` produced by this project. Substitute its actual experiment ID:

```bash
python main.py diagnose results/full-eeg/eeg/eegnet-agfl-EXPERIMENT_ID/seed_0 \
  --output-dir analysis/diagnostics/agfl-seed-0
```

The command restores that checkpoint and reads the original dataset using the
saved preprocessing. Data content and split fingerprints must match. It defaults
to CPU, validation samples, at most 256 class-balanced examples, and seeded
t-SNE. These limits keep it separate from full-cohort evaluation. Options include
`--device cuda`, `--partition test`, `--max-samples 512`, and
`--embedding pca|tsne|umap`. UMAP additionally requires the `umap` extra.
Relocated recordings/labels/NPZ files can be supplied with `--data-dir`,
`--labels-dir`, or `--data-path`; this does not relax fingerprint checks.

To generate diagnostics for every completed EEG seed after the study finishes,
run this loop from `AGFL` on the target machine:

```bash
find results/full-eeg/eeg -type f -name result.json -print0 |
while IFS= read -r -d '' task_result_file; do
  task_run_dir=${task_result_file%/result.json}
  python main.py diagnose "$task_run_dir" \
    --output-dir "analysis/diagnostics/${task_run_dir#results/full-eeg/}" || break
done
```

Each checkpoint loads its dataset again, so this is more expensive than plotting
the saved result tables. A single-seed diagnostic is sufficient for an initial
visual inspection; the loop processes all completed seeds when requested.

The diagnostic outputs cover:

- EEG/ECG example traces in saved preprocessing/model-input units.
- Welch spectra and channel-frequency maps when sampling rate is known.
- EEG alpha/beta scalp-power maps, and C3/C4 class-average STFT maps when those
  electrodes are available.
- EEG CSP spatial patterns fitted only on a selected subset of the training
  partition, never on validation/test labels.
- PCA, t-SNE, or optional UMAP of classifier-input features.
- Per-layer, per-head mixer heatmaps, source-electrode/source-patch weights,
  sparsity, and probability-row entropy where defined.
- ECG traces with the corresponding example's temporal mixing weights.
- AGFL learned coefficients and per-hop/shared feature-projection heatmaps.

Current checkpoint diagnostics support every model/attention combination in
both modalities. Each injected layer declares its actual token axis. Electrode
maps use physical channel labels where available; temporal EEG variants are
labeled as time tokens and do not get electrode graph topomaps. EEG scalp maps
require known channel positions in the standard 10–20 montage. Generic NPZ/synthetic data without physical metadata still support
signal traces, embeddings, and index-based mixer maps; unavailable figures are
listed in the manifest. Dense diagnostic maps are limited to 256 tokens even for
mechanisms whose training path is linear in token count.

An AGFL heatmap is its graph A before polynomial/renormalized filtering.
Linformer and Nyström figures show effective value-mixing weights, which can be
signed. They are not relabeled as probabilities. Entropy is reported only when
every contributing row is nonnegative and sums to one. Sparsity counts weights
with absolute value at most 1e-8. Feature-projection axes are features, not
electrodes; graph weights and embeddings are descriptive, not causal importance
or accuracy estimates. Spectra of normalized inputs are not raw-voltage spectra.

The diagnostic manifest records selected sample IDs, training IDs used for CSP,
checkpoint hash, plotting settings, and training/diagnostic provenance. It also
compares reconstructed probabilities to saved probabilities (`rtol=1e-4`,
`atol=1e-5`) and records mismatches for review. State loading is strict. Source
hash differences are disclosed because plotting code may be added after a run;
loading a checkpoint alone does not prove that an arbitrarily changed model
implementation reproduces its predictions.

To add descriptive sparsity/entropy-versus-accuracy figures after generating
diagnostics for the desired runs:

```bash
python main.py analyze results/full-eeg --output-dir analysis/full-eeg --plots \
  --diagnostics-root analysis/diagnostics
```

Only matching runs with verified reconstructed predictions are included.
Diagnostic sampling settings and partitions are separated. These scatterplots
do not establish that sparsity caused an accuracy change; the declared Top-k
ablation runs remain the controlled experiments.

## Existing cluster studies and verification

Existing saved results can be plotted without retraining. Labels show both
model and attention. Earlier mechanism-as-model runs are reported as Signal
Transformer, and generic v1 checkpoints have a construction adapter. Other v1
original-backbone checkpoints require their training checkout for diagnostics. For a study already in
progress, keep its source and environment stable until all models/seeds finish,
or use a separate analysis checkout. Copying new source into a live multi-model
study changes the source fingerprint used to pair subsequent runs, even when
the change only adds plotting.

On the target machine, `python -m pytest tests/test_visualization.py` checks
prediction/split alignment, matched pairs, non-training diagnostics, map/output
equivalence, missing entropy semantics, and PNG/PDF output. No local rendering
or runtime validation was performed here.

Implementation uses [Matplotlib file exports](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.savefig.html),
[MNE scalp maps](https://mne.tools/stable/generated/mne.viz.plot_topomap.html),
[training-fitted CSP](https://mne.tools/1.6/generated/mne.decoding.CSP.html), and
[scikit-learn t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
