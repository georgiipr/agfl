# Result sessions and recovery

Pass the session root to `--output-dir`, not its `report` or `artifacts`
subdirectory. A typical session is:

```text
results/study/
  report/
    runs/           configurations, splits, histories, predictions, metrics
    analysis/       summary tables and result figures
    diagnostics/    checkpoint figures and diagnostic arrays
  artifacts/
    eeg/            subject-specific canonical runs and checkpoints
    ecg/            canonical ECG runs and checkpoints
    candidates/     validation-only tuning candidates, when applicable
    selected/       selected tuning runs, when applicable
```

Only relevant directories are created. Report run paths mirror the artifact
hierarchy. Source/configuration provenance, selection evidence and failures are
included for interpretation; model checkpoints and raw recordings remain in
their original locations. Tuning sessions also include search-level summaries.

## Generate tables and figures

After training, generate checkpoint diagnostics on the compute machine:

```bash
python main.py diagnose-session results/study \
  --device cuda --partition validation --embedding tsne
```

Then generate result figures and tables:

```bash
python main.py analyze results/study --plots
```

`report/analysis/report.md` contains the tables;
`report/analysis/figures/index.md` indexes the result figures. Each diagnostic
run has an `index.md` under `report/diagnostics`. Adding `--report` to a training
command requests both reporting steps automatically after training finishes.

## Rerun or resume

Ordinary `run` and `sweep` commands replace matching seed artifacts by default.
Interrupted runs restart from epoch 1. Add `--skip-completed` to keep completed
fits whose saved identities match. These are selected inference checkpoints,
not optimizer-resume files. Use a new output directory to retain previous runs
when changing source, data or scientific settings.

Training locks prevent simultaneous writers to the same run. Keep a study's
source and environment stable while it runs. After replacing a run, regenerate
its analysis and diagnostics so exported figures reflect the new artifacts.

For a completed session with an older directory layout:

```bash
python main.py organize-results results/study
```

Organization moves canonical runs into `artifacts` and prepares report copies.
It does not retrain models, change metrics or rewrite saved configurations.
Run it after all session writers have stopped. Keep the original training
checkout when an older checkpoint requires its original implementation.

## Transfer and inspect reports

Download only `results/study/report/`, preserving its subdirectories. Substitute
your SSH destination and project path:

```bash
rsync -av --progress \
  user@compute-host:/path/to/AGFL/results/study/report/ \
  ./downloaded-reports/study/
```

Existing figures and Markdown tables can be opened directly. A downloaded
report also supports saved-result analysis on a suitable analysis machine:

```bash
python main.py analyze downloaded-reports/study --plots
```

Checkpoint diagnostics additionally require canonical checkpoints and matching
recordings; the report alone cannot regenerate them. See
[visualization](visualization.md) for single-run diagnostics and relocated data.
