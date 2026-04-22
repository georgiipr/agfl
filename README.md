To run the code:
```
PYTHONPATH=. python3 agfl/main.py --task <task_name> --model <model_name> --agfl <state>
```

task_name:
```
1. eeg
2. ecg
```

model_name:
```
1. conformer
2. eegnet
3. eegencoder
4. dstseegencoder
```

note: conformer currently for ecg, other models for eeg only

state:
```
1. on
2. off
3. none
```

note: state off means standard MultiHeaded attention is being used, None - no attention



-Data_loaders handle datasets

-Models contain various models, with custom options to either include or exclude agfl_layer (custom attention) and standard MH attention

-Trainers contain the training files for both ecg and eeg

-Plots contain various plotting functions

-keys handles the exectuion of various launch setups

-Main handles general main


Note: eeg uses BCI2a IV Competition dataset (.gdf)

Zip can be downloaded from this link: https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit

To ensure it works, put uploaded files into the ml folder in the same directory where this project is located

ecg uses arrithmiya dataset. ask the owner for the folder.

Results (latest: 16 April), were acquired for eeg for two-classes task (left and right hands, classes 0 and 1)
```
EEGEncoder (AGFL): -> Accuracy: 0.3565 | F1 (Macro): 0.2829 | ROC-AUC: 0.6302 | Loss: 0.5460
EEGEncoder (STANDARD) -> Accuracy: 0.3565 | F1 (Macro): 0.2970 | ROC-AUC: 0.6188 | Loss: 0.5537

Eegnet (AGFL) -> Accuracy: 0.5010 | F1 (Macro): 0.4816 | ROC-AUC: 0.7413
EEGNet (Standard) -> Accuracy: 0.4663 | F1 (Macro): 0.4592 | ROC-AUC: 0.7439

Dstseegencoder (STANDARD) -> Accuracy: 0.4933 | F1 (Macro): 0.4782 | ROC-AUC: 0.7616
Dstseegencoder (AGFL) -> Accuracy: 0.3430 | F1 (Macro): 0.3392 | ROC-AUC: 0.5825
Dstseegencoder (NONE) -> Accuracy: 0.3295 | F1 (Macro): 0.3248 | ROC-AUC: 0.5500
```

