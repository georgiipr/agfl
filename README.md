To run the code after installing dependencies via uv:
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
1. conformer - ecg
2. eegnet - eeg
```

state:
```
1. on
2. off
3. none
```

note: state off means standard MultiHeaded attention is being used, None - no attention





Note: eeg uses BCI2a IV Competition dataset (.gdf)

Zip can be downloaded from this link: https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit

To ensure it works, put uploaded files into the ml folder in the same directory where this project is located

ecg uses arrithmiya dataset: https://physionet.org/content/mitdb/1.0.0/