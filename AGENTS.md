# AGFL project instructions

These instructions apply to this repository and all its subdirectories.

## Subagents

- **Do not use subagents unless the user explicitly requests them for the task.**
  Work directly by default; task complexity does not authorize delegation.
- This replaces the previous authorization for proactive delegation and the
  recommendation to use 2–4 subagents.
- When explicitly requested, give subagents bounded tasks, respect concurrency
  limits, and apply all project instructions and current user constraints.
  The primary agent remains responsible for integration and verification.

## Execution and verification

- **Do not run project code locally.** This includes the AGFL CLI, project tests,
  training, model inference, and checkpoint diagnostics. Prepare commands for
  the user's cluster instead, unless the user explicitly changes this rule.
- Local file inspection, static syntax/configuration checks, and standalone
  analysis of saved artifacts are allowed. Do not import the project as part of
  a static check or saved-artifact analysis.
- Distinguish static checks from tests actually executed on the cluster. Do not
  claim numerical correctness, accuracy gains, or runtime gains without results.
- Preserve existing uncommitted changes and research artifacts. Do not reset,
  delete, or overwrite unrelated work.

## Research and experiment conventions

- Keep model and attention selections separate: a model is a backbone such as
  EEGNet; the chosen attention runs inside it. The base attention is MHA. Do
  not add a no-attention option.
- Train BCI Competition IV 2a subjects individually. Keep preprocessing and
  candidate/checkpoint selection isolated from held-out test data.
- Improving AGFL's absolute accuracy and performance relative to other
  attentions is the objective, not an assumed outcome. Preserve fair comparisons
  and report unfavorable results as well as favorable ones.
- Preserve saved configuration, split, checkpoint, and source provenance. Make
  experimental changes explicit instead of silently reinterpreting old runs.
- Use interactive **`salloc`** launches. Do not create `.sbatch` files unless
  the user explicitly requests that workflow again.
- Use the existing Python environment. Do not add libraries or make `pytest`
  or the development test suite a prerequisite for training or plotting.
  Development tests remain separate from experiment launches.
- When providing training commands, also provide checkpoint-diagnostic and
  result-plot commands in **separate copyable blocks**. State the working
  directory, dataset path, environment, resources, and output folder clearly.

## Communication

- Proceed autonomously within the user's authorized scope. Ask only for missing
  information or permission that is actually necessary.
- Give concise progress updates during substantial work and a self-contained
  final response stating what changed, what was checked, and what remains
  unverified.
