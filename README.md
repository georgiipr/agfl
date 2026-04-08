To run for ecg:
```
PYTHONPATH=. python3 main.py --task ecg
```

To run for eeg:
```
PYTHONPATH=. python3 main.py --task eeg
```

Current ecg model is conformer, curent eeg model is EEGNet with the support of EEGEncoder

ToDo:
1. Make an option to choose the model and agfl on / off as a key (PYTHONPATH=. python3 main.py --task eeg --model eegnet --agfl on)
2. Remove none attention: off agfl from 1 meand that the standard attention is being used
3. Add Pipeline for BCI Password
4. Do plots for eeg: CSP and STFT (C3 & C4 channels) / first do more advanced data denoising
5. dsts model was added - needs pipeline integration
6. More models needed after step 1 is specified


Project structure: attention5.py is full ecg pipeline (depricated), attention6.py is full eeg pipeline (depricated)

Data_loaders handle datasets
Models contain various models, with custom options to either include or exclude agfl_layer (custom attention) and standard MH attention
Trainers contain the training files for both ecg and eeg
Plots contains various plotting functions
Main handles imports and execution

Note: eeg uses BCI2a IV Competition dataset (.gdf)
Zip can be downloaded from this link: https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit
To ensure it works, put uploaded files into the ml folder in the same directory where this project is located

