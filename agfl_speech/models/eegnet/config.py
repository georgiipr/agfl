EEG_DEFAULTS = {
    # The first temporal kernel spans half the sampling rate, as the EEGNet
    # authors recommend, so 2 Hz and above are resolvable: 250 samples (0.5 s)
    # at the 500 Hz of the SI_Hom recordings. The kernel is even, so
    # padding='same' pads 124 samples on the left and 125 on the right.
    'temp_kernel': 250, 'f1': 16, 'd': 2, 'f2': 32, 'pk1': 8, 'pk2': 16,
    'dropout_rate': .5, 'max_norm1': 1., 'max_norm2': .25, 'attention_axis': 'electrode',
    'spatial_readout': 'learned', 'attention_residual': True,
    'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5,
    'attention_dropout': 0.0, 'initialization': 'pytorch',
}
# Electrode layouts: `compact` (per-electrode encoder only), `spatial_fusion`
# (original spatial path plus a parallel electrode-attention path) and
# `pre_spatial` (attention between electrodes at every time step, feeding the
# original spatial filter; no parallel classifier input). `electrode_dim` is
# the token width of the first two layouts and is unused by `pre_spatial`.
EEG_DEFAULTS.update(electrode_dim=32, electrode_architecture='spatial_fusion')
