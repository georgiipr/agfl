To run the code:
```
PYTHONPATH=. python3 agfl/main.py --task <task_name> --model <model_name> --agfl <state>
```
task_name:
1. eeg
2. ecg

model_name:
1. conformer
2. eegnet
3. eegencoder
4. dstseegencoder

note: conformer currently for ecg, other models for eeg only

state:
1. on
2. off

note: state off means standard MultiHeaded attention is being used

ToDo:
1. Advanced data denoising for eeg
2. Add Pipeline for BCI Password
3. Add more models for eeg

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

