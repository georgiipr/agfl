# Request status

Implemented means source and configuration are present. No project code,
training, evaluation, installation or tests have been run locally. Empirical
validation remains pending on the intended machine.

| Requirement | Source status and evidence |
|---|---|
| Model folders with EEG/ECG variants | Implemented for EEGNet, EEGEncoder, DSTS EEGEncoder, Conformer and the explicit Signal Transformer backbone. Each owns config, backbone, eeg.py and ecg.py. |
| Separate MODELS and ATTENTION selection | Implemented: independent registries, model/attention configuration keys and --model/--attention CLI options. Each backbone receives the selected attention factory. |
| Remove archived model names | Production adapters and their presets removed. Original sources remain only in tests/references for independent checks; they are not installed or selectable. |
| Five attentions inside each chosen model | AGFL, MHA, Performer, Linformer and Nyströmformer are injected at the model's attention locations. Comparison presets hold the backbone fixed. |
| Shared data/train/evaluate policy | ModelSpec entry methods delegate to common loaders and engine. |
| Five seeds, same persisted splits, subject independence | Defaults and manifests implemented; runtime repeatability checks pending. |
| Preserve AGFL mathematics | Active original polynomial defaults preserved. Independent forward/gradient checks supplied; no measured parity claim yet. |
| EEG inter-electrode / ECG temporal analysis | Explicit default axes in every model; required architecture adaptations documented in model_attention_structure.md. |
| Ablations | Attention settings and EEG/ECG matrices implemented. No-attention deliberately absent following the explicit removal instruction. |
| Metrics and statistical comparisons | Saved-result aggregation, mean/sample SD, paired t/Wilcoxon, effects and Holm correction. Pairing requires the same actual backbone and protocol. |
| Complete artifacts and replay | Config, checkpoint, history, predictions, split and result records with source/package/data identities. v2 replay checks supplied. |
| Relaunch after abort without deleting results | Matching seed artifacts are replaced by default and training restarts from epoch 1. --skip-completed preserves matching finished seeds while restarting incomplete ones. Regression checks are supplied but unrun locally. |
| Audit poor accuracy and original pipeline | Source findings documented in pipeline_audit.md and data_audit.md. Their measured effects remain unknown. |
| Tables and plots | Saved-result figures and model-aware checkpoint diagnostics implemented; rendering checks pending. |
| Earlier saved results | Read without changing original hashes/configs. Mechanism-as-model v1 runs are explicitly labeled Signal Transformer. |
| Execute experiments and verify research behavior | Pending on target machine, as required by the prohibition on local execution. |
| README and cleanup | Updated commands, structure, architecture differences and retained test-source provenance. |

The model/attention structure gap is addressed in source. Required remaining
work is target-machine verification and the declared repeated-seed experiments,
not an assertion that tests or accuracy targets have already passed.
