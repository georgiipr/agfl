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

-Trainers contain the training files for both ecg and eeg

-Plots contain various plotting functions

-keys handles the exectuion of various launch setups

-Main handles general main


Note: eeg uses BCI2a IV Competition dataset (.gdf)

Zip can be downloaded from this link: https://www.bbci.de/competition/iv/download/index.html?agree=yes&submit=Submit

To ensure it works, put uploaded files into the ml folder in the same directory where this project is located



While traditional Spiking Neural Networks (SNNs) rely on the natural dynamics of Leaky Integrate-and-Fire (LIF) neurons to capture temporal relationships (acting as biological low-pass filters), their memory is inherently short-term. As time passes, the membrane potential leaks, and long-range dependencies in the EEG signal can fade.

Integrating an attention mechanism creates a hybrid architecture. The SNN acts as an efficient, highly sparse feature extractor, mapping the EEG channels into discrete spike trains. The attention layer then acts as a router, looking across the entire temporal sequence of spikes to weigh the most critical "events" (like the onset of a motor imagery task) while ignoring the silent or noisy periods.

Spiking EEGNet: built via snnTorch.
Difference from standard EEGNet: As the LIF neurons can handle the time dimension, we have no need for the 2D temporal convolutions from the standard EEGNet.

Instead, we apply spatial 1D convolutions at each timestep, feed them into the spiking neurons, collect the temporal sequence, and pass that sequence to the attention blocks.


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

