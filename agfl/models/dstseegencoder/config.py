EEG_DEFAULTS = {'dim': 32, 'temporal_kernel': 64, 'pool1': 7, 'pool2': 7,
                'num_branches': 5, 'depth': 4, 'mlp_ratio': 4., 'dropout': .3, 'attention_axis': 'electrode'}
ECG_DEFAULTS = {**EEG_DEFAULTS, 'temporal_kernel': 15, 'pool1': 4, 'pool2': 4, 'attention_axis': 'time'}
