# AGFL_speech project instructions

These instructions apply to this repository and all its subdirectories. They
carry over the AGFL project's working rules.

## Subagents

- **Do not use subagents unless the user explicitly requests them for the task.**
  Work directly by default; task complexity does not authorize delegation.
- When explicitly requested, give subagents bounded tasks, respect concurrency
  limits, and apply all project instructions and current user constraints.
  The primary agent remains responsible for integration and verification.

## Execution and verification

- **Do not run project code locally.** This includes the CLI, `check`, project
  tests, training, model inference, and checkpoint diagnostics. Prepare commands
  for the user's cluster instead, unless the user explicitly changes this rule.
- Local file inspection, static syntax/configuration checks, and standalone
  analysis of saved artifacts are allowed. `scripts/prepare_si_hom.py` is a
  standalone converter and never imports the project. Do not import the project
  as part of a static check or saved-artifact analysis.
- Distinguish static checks from tests actually executed on the cluster. Do not
  claim numerical correctness, accuracy gains, or runtime gains without results.
- Preserve existing uncommitted changes and research artifacts. Do not reset,
  delete, or overwrite unrelated work.

## Research and experiment conventions

- The only dataset is SI_Hom imagined speech (`si_hom`). Do not add ECG, BCI
  Competition IV 2a or generic array datasets back.
- Keep model and attention selections separate: a model is a backbone such as
  EEGNet; the chosen attention runs inside it across the 19 electrodes. The base
  attention is MHA. Do not add a no-attention option or temporal attention.
- The default cohort is pooled (the dataset authors' setting); individual
  subjects and subject-independent splits are explicit options. Keep
  preprocessing and candidate/checkpoint selection isolated from held-out test
  data.
- Improving AGFL's absolute accuracy and performance relative to other
  attentions is the objective, not an assumed outcome. Preserve fair comparisons
  and report unfavorable results as well as favorable ones.
- Preserve saved configuration, split, checkpoint, and source provenance. Make
  experimental changes explicit instead of silently reinterpreting old runs.
- The data folder `AGFL_speech_data` sits next to the project folder. Never
  hard-code an absolute data path; a relative `data.data_dir` is resolved
  against the project folder.
- Keep experiment-specific **`.sbatch`** launchers outside this repository, in
  the user's `Downloads/cluster-jobs/AGFL_speech/` folder. Keep only the latest
  launcher there. Submit launchers from the cluster's project folder so
  `SLURM_SUBMIT_DIR` points to the project. Keep public launch examples portable.
- Use the existing Python environment. Do not add libraries or make `pytest`
  or the development test suite a prerequisite for training or plotting.
- When providing training commands, also provide checkpoint-diagnostic and
  result-plot commands in **separate copyable blocks**. State the working
  directory, dataset path, environment, resources, and output folder clearly.
- Use **`rsync`** for cluster-to-Mac download commands. Default to
  `rsync -av --progress` so new and changed files are transferred. Add
  `--ignore-existing` when the user explicitly requests skipping every
  existing destination file; preserve the intended local folder layout.

## Communication

- Proceed autonomously within the user's authorized scope. Ask only for missing
  information or permission that is actually necessary.
- Give concise progress updates during substantial work and a self-contained
  final response stating what changed, what was checked, and what remains
  unverified.
