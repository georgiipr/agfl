EEG_DEFAULTS = {
    'temp_kernel': 32, 'f1': 16, 'd': 2, 'f2': 32, 'pk1': 8, 'pk2': 16,
    'dropout_rate': .5, 'max_norm1': 1., 'max_norm2': .25, 'attention_axis': 'electrode',
    'spatial_readout': 'learned', 'attention_residual': True,
    'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5,
    'attention_dropout': 0.0, 'initialization': 'pytorch',
    # Compatibility selector for archived power-study configurations. Normal
    # EEGNet uses its original mean-pooling backbone; no power branch is built.
    'temporal_statistics': 'mean',
}
ECG_DEFAULTS = {**EEG_DEFAULTS, 'pk1': 4, 'pk2': 4, 'attention_axis': 'time'}
# Keep the original spatial convolution features alongside electrode attention.
# Snapshot reconstruction supplies 'compact' for saved runs without this key.
# ECG defaults above deliberately exclude both EEG-specific selectors.
EEG_DEFAULTS.update(electrode_dim=32, electrode_architecture='spatial_fusion')
