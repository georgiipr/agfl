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
```

note: state off means standard MultiHeaded attention is being used



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
Dstseegencoder (STANDARD) Results:
  Accuracy : 0.7577
  F1 (Macro): 0.7574
  ROC-AUC (OVR): 0.8411
Loss=0.0605

Dstseegencoder (AGFL) Results:
  Accuracy : 0.5571
  F1 (Macro): 0.5439
  ROC-AUC (OVR): 0.5749
Loss=0.0770

Eegnet (AGFL) Results:
  Accuracy : 0.7473
  F1 (Macro): 0.7376
  ROC-AUC (OVR): 0.8375
Loss=0.0630

Eegnet (STANDARD) Results:
  Accuracy : 0.7385
  F1 (Macro): 0.7337
  ROC-AUC (OVR): 0.8314
Loss=0.0639

Eegencoder (AGFL) Results:
  Accuracy : 0.5577
  F1 (Macro): 0.3699
  ROC-AUC (OVR): 0.5006
Loss=0.087

Eegencoder (STANDARD) Results:
  Accuracy : 0.5423
  F1 (Macro): 0.5373
  ROC-AUC (OVR): 0.5127
```

