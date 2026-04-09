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

ToDo:
1. Advanced data denoising for eeg
2. Add Pipeline for BCI Password
3. Add more models for eeg
4. Accuracy of encoders is horrible

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

Results (latest), were acquired for eeg for two-classes task (left and right hands, classes 0 and 1)
```
Eegnet (AGFL) Results:                                                                         
  Accuracy : 0.7423                                                                            
  F1 (Macro): 0.7351                                                                           
  ROC-AUC (OVR): 0.8374

Eegnet (STANDARD) Results:                                                                     
  Accuracy : 0.7346                                                                            
  F1 (Macro): 0.7337                                                                           
  ROC-AUC (OVR): 0.8315

Eegencoder (AGFL) Results:
  Accuracy : 0.5538
  F1 (Macro): 0.3858
  ROC-AUC (OVR): 0.5122

Eegencoder (STANDARD) Results:
  Accuracy : 0.5500
  F1 (Macro): 0.4256
  ROC-AUC (OVR): 0.5108

Dstseegencoder (AGFL) Results:
  Accuracy : 0.5615
  F1 (Macro): 0.5603
  ROC-AUC (OVR): 0.5236
```

