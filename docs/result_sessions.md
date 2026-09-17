# Results to download and artifacts to keep on the cluster

Each training output directory is a session with two folders:

```text
results/<session>/
  report/                 download this folder
    runs/                 configs, splits, histories, predictions and metrics
    analysis/             tables and result plots
    diagnostics/          checkpoint plots, indexes and diagnostic arrays
    search_report.md      overall tuning score, when this is a search
    search_result.json
    selection_report.json
  artifacts/              keep on the cluster
    eeg/                  ordinary EEG runs, including checkpoints
    ecg/                  ordinary ECG runs, including checkpoints
    candidates/           validation-only search fits, including checkpoints
    selected/             selected search runs, including checkpoints
```

Only the directories relevant to that session are created. Under `report/runs`,
run paths mirror their paths inside `artifacts`, so subjects, seeds, candidates
and attentions remain distinguishable. Candidate histories and validation
selection records are included even when that candidate was not selected.
Saved failure records are included for troubleshooting.

The report retains settings, sample IDs, split and source provenance, learning
curves, prediction probabilities, scores, selection evidence and generated
figures. Checkpoint files and raw recording payloads are excluded. Diagnostic
embedding/graph arrays remain available for interpreting the figures.
`artifacts` contains the canonical training state and must remain available for
resuming an interrupted search or regenerating checkpoint diagnostics. Nothing
is discarded simply because it does not belong in the download.

## Existing completed A03 search

Run from the cluster's `AGFL` directory with the Python environment active.
The example session is `results/eegnet-A03-agfl-qkv-v2`; recordings are in
`../ml`. Do not reorganize a directory while training or plotting writes to it.

If this study started with an older checkout, finish its training and generate
any missing checkpoint diagnostics with that same checkout first. This keeps
checkpoint reconstruction tied to its training implementation. Existing plots
inside the session are retained by organization. Then use the updated checkout
for the following command; it does not retrain or reevaluate a model:

```bash
python main.py organize-results results/eegnet-A03-agfl-qkv-v2
```

This moves the old run layout into `artifacts` and prepares the small report
copies. It preserves saved configuration bytes and provenance rather than
rewriting the experiment to match the new locations. Repeating the command
refreshes the report without duplicating checkpoint payloads.

If checkpoint plots are still needed and the checkout can reconstruct the
saved model, generate them on the cluster:

```bash
python main.py diagnose-session results/eegnet-A03-agfl-qkv-v2 \
  --device cuda --partition validation --embedding tsne
```

This covers completed ordinary runs or selected search runs, not every
validation-only candidate. It reads `artifacts` and the original recordings;
its outputs go to `report/diagnostics`. Saved diagnostic manifests disclose
source differences and reconstructed-prediction checks. Retain the training
checkout if current code cannot reproduce an old checkpoint.

Generate or refresh all result plots and tables:

```bash
python main.py analyze results/eegnet-A03-agfl-qkv-v2 --plots
```

The default output is `report/analysis`; available checkpoint diagnostics are
discovered under `report/diagnostics`. The result-plot index is
`report/analysis/figures/index.md`, and each checkpoint diagnostic has its own
`index.md` beneath `report/diagnostics`. Migrated plots may retain their earlier
`report/analysis/diagnostics` location; keep that directory in the download too.

**Download only `results/eegnet-A03-agfl-qkv-v2/report/`.** Keep this folder's
subdirectories together. The mean score of the validation-selected search is
in `report/search_report.md` and `report/search_result.json`; generic analysis
may split the winners into several configuration groups.

## New sessions

Keep passing the session root to `--output-dir`; do not append `artifacts` or
`report` to a training command. New training writes into `artifacts` and
refreshes the small run records in `report`. Training does not render figures;
run `diagnose-session SESSION` and then `analyze SESSION --plots` after it ends.
After retraining or replacing runs, regenerate their plots before downloading.
Without a preset-specific or explicit output directory, training defaults to
`results/session`. Use a descriptive `--output-dir results/<study-name>` when
starting a separate study.

For one checkpoint, pass its canonical artifact directory to `diagnose`:

```bash
python main.py diagnose \
  results/eegnet-A03-agfl-qkv-v2/artifacts/selected/eeg/subject_A03/seed_0 \
  --device cuda --partition validation --embedding tsne
```

The output defaults to the corresponding path under `report/diagnostics`.
An explicit `--output-dir` still overrides that destination. Within an organized
session it must be inside `report/`; an external output directory is also allowed.
Old explicit paths such as `SESSION/analysis` should become
`SESSION/report/analysis`, or simply omit `--output-dir`.

## Cluster checks for the layout change

These checks were prepared for the experiment machine and were not run locally:

```bash
python -m pytest tests/test_session.py tests/test_session_cli.py \
  tests/test_report_consumers.py tests/test_engine.py tests/test_eegnet_bci2a.py \
  tests/test_analysis.py tests/test_visualization.py -q
```

## Using a downloaded report

Open the PNG/PDF files and Markdown indexes without training dependencies.
On a designated analysis machine with the plotting dependencies, the report
alone also supports saved-result analysis and plot regeneration:

```bash
python main.py analyze /path/to/downloaded/report --plots
```

It cannot regenerate signals, embeddings or attention diagnostics from a
checkpoint: those require `artifacts` and matching source recordings on the
cluster. No project commands, project tests or model diagnostics have been
executed locally to prepare this change.
