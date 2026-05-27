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