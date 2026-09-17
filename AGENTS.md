# AGFL project instructions

These instructions apply to this repository and all its subdirectories.

## Autonomous delegation

- The user authorizes proactive delegation. For substantial tasks with useful
  independent subtasks, use **2–4 subagents** without asking the user to choose
  agents, divide the work, or approve routine delegation.
- The primary agent chooses subtasks based on the user's request. Useful roles
  include implementation, an independent correctness review, analysis of saved
  results, and verification or documentation.
- Respect the environment's concurrency limits. If fewer subagent slots are
  available, use the available slots and stage additional work only when useful.
- Delegate concrete, bounded tasks with the necessary context and constraints.
  Give editing agents distinct file ownership to prevent conflicting changes.
- Continue useful work in the primary agent while subagents work. Avoid
  duplicating their tasks, and communicate discoveries that affect their work.
- The primary agent owns integration, review, appropriate verification, and the
  final response. Check subagent findings before presenting them as established.
- Do not create busywork to meet an agent count. Handle trivial edits, short
  answers, and tasks that cannot usefully be divided directly. Use fewer agents
  when the task or available capacity warrants it.
- Apply these project instructions and the user's current constraints to every
  subagent. Delegation does not authorize additional actions or broader scope.

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
- When providing training commands, also provide checkpoint-diagnostic and
  result-plot commands in **separate copyable blocks**. State the working
  directory, dataset path, and output folder clearly.

## Communication

- Proceed autonomously within the user's authorized scope. Ask only for missing
  information or permission that is actually necessary.
- Give concise progress updates during substantial work and a self-contained
  final response stating what changed, what was checked, and what remains
  unverified.
