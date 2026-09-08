"""Small real-backbone configurations for target-machine verification."""


def small_model_options(key, modality):
    return {
        'eegnet': {'f1': 4, 'd': 2, 'f2': 8, 'pk1': 2, 'pk2': 2, 'temp_kernel': 7},
        'eegencoder': {'eegn_F1': 4, 'eegn_D': 2, 'eegn_kernelSize': 7, 'pool1': 2, 'pool2': 2, 'tcn_filters': 8, 'tcn_depth': 2},
        'dstseegencoder': {'dim': 8, 'temporal_kernel': 7, 'pool1': 2, 'pool2': 2, 'num_branches': 2, 'depth': 2},
        'conformer': {'dim': 8, 'depth': 1, **({'temporal_bins': 2} if modality == 'eeg' else {})},
        'signal_transformer': {'dim': 8, 'depth': 1, 'eeg_temporal_bins': 2, 'eeg_kernel_size': 3, 'ecg_patch_size': 8},
    }[key]
