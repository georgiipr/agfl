EEG_DEFAULTS = {
    # The first temporal kernel spans half the 250 Hz sampling rate (125
    # samples, 0.5 s), as the EEGNet authors recommend, so 2 Hz and above are
    # resolvable; 32 samples (0.128 s) could not separate mu from beta rhythms.
    'temp_kernel': 125, 'f1': 16, 'd': 2, 'f2': 32, 'pk1': 8, 'pk2': 16,
    'dropout_rate': .5, 'max_norm1': 1., 'max_norm2': .25, 'attention_axis': 'electrode',
    'spatial_readout': 'learned', 'attention_residual': True,
    'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5,
    'attention_dropout': 0.0, 'initialization': 'pytorch',
    # Compatibility selector for archived power-study configurations. Normal
    # EEGNet uses its original mean-pooling backbone; no power branch is built.
    'temporal_statistics': 'mean',
}
# ECG keeps its previous kernel and pooling; the EEG kernel change is EEG-only.
ECG_DEFAULTS = {**EEG_DEFAULTS, 'temp_kernel': 32, 'pk1': 4, 'pk2': 4, 'attention_axis': 'time'}
# Electrode layouts: `compact` (per-electrode encoder only), `spatial_fusion`
# (original spatial path plus a parallel electrode-attention path) and
# `pre_spatial` (attention between electrodes at every time step, feeding the
# original spatial filter; no parallel classifier input). `electrode_dim` is
# the token width of the first two layouts and is unused by `pre_spatial`.
# Snapshot reconstruction supplies 'compact' for saved runs without this key.
# ECG defaults above deliberately exclude both EEG-specific selectors.
EEG_DEFAULTS.update(electrode_dim=32, electrode_architecture='spatial_fusion')
