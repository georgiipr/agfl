"""Small real-backbone configurations for target-machine verification."""


def small_model_options(key):
    return {
        'eegnet': {'f1': 4, 'd': 2, 'f2': 8, 'pk1': 2, 'pk2': 2, 'temp_kernel': 7},
        'eegnet_original': {'f1': 4, 'd': 2, 'f2': 8, 'pk1': 2, 'pk2': 2, 'temp_kernel': 7},
        'signal_transformer': {'dim': 8, 'depth': 1, 'eeg_temporal_bins': 2, 'eeg_kernel_size': 3},
    }[key]
