# Documentation

Start with the repository [README](../README.md) for installation, dataset
locations, common training commands, attention comparisons and both plot steps.

| Guide | Contents |
|---|---|
| [Models and attention](model_attention_structure.md) | Backbones, independent mechanism selection and registry contracts |
| [Attention mathematics](mathematics.md) | AGFL equations, conditioning and comparison mechanisms |
| [Datasets](data_audit.md) | EEG/ECG preprocessing, splits, labels, provenance and NPZ adapters |
| [EEG electrode attention](eeg_interchannel.md) | Spatial node semantics and EEG launch examples |
| [EEGNet pre-spatial layout](eegnet_pre_spatial.md) | Attention before spatial convolution |
| [Result sessions](result_sessions.md) | Report/artifact layout, reruns, migration and transfers |
| [Visualization](visualization.md) | Saved-result plots and checkpoint diagnostics |
| [Statistics](statistics.md) | Aggregation, paired comparisons and interpretation |

These guides describe the reusable implementation and workflow. Store generated
reports, run-specific measurements and local research notes outside `docs/`.
