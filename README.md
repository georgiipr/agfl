To run the code:
```
PYTHONPATH=. python3 agfl/main.py --task <task_name> --model <model_name> --agfl <state>
```

task_name:
```
1. eeg
2. nc
```

model_name:
```
1. snn
2. eegnet
```

note: conformer currently for ecg, other models for eeg only

state:
```
1. on
2. off
3. none
```

note: state off means standard MultiHeaded attention is being used, None - no attention



-Data_loader handles the dataset plus processing for the SNN\

-Models contain various models, with custom options to either include or exclude agfl_layer (custom attention) and standard MH attention\

-Trainer contains the training for both nc and eeg\

-Keys handles the exectuion of various launch setups\


Note: eeg uses BCI2a IV Competition dataset (.gdf): Zip can be downloaded from this link: https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit

(To ensure it works, put uploaded files into the ml folder in the same directory where this project is located)\


```
Results:

EEGNet:
(Standard attention)
  Average Accuracy : 0.7299
  Average F1 Score : 0.7204
  Average ROC-AUC  : 0.8650

(AGFL Attention)
  Average Accuracy : 0.7663
  Average F1 Score : 0.7626
  Average ROC-AUC  : 0.8707

```

