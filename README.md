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
1. Make an option to choose the model as a key
2. Add Pipeline for BCI Password
3. Do plots for eeg: CSP and STFT (C3 & C4 channels) / first do more advanced data denoising
4. dsts model was added - needs pipeline integration
5. More models needed after step 1 is specified
