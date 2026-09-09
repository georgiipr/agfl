EEG_DEFAULTS = {
    'eegn_F1': 16, 'eegn_D': 2, 'eegn_kernelSize': 64, 'pool1': 8, 'pool2': 7,
    'tcn_depth': 3, 'tcn_kernelSize': 4, 'tcn_filters': 32, 'dropout': .3,
    'attention_axis': 'electrode', 'spatial_readout': 'learned',
}
ECG_DEFAULTS = {**EEG_DEFAULTS, 'eegn_kernelSize': 32, 'pool1': 4, 'pool2': 4, 'attention_axis': 'time'}
