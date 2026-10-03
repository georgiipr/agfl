"""Defaults shared by comparable mechanisms, independent of the dataset."""

DEFAULTS = {
    "dim": 64,
    "depth": 2,
    "dropout": 0.1,
    "mlp_ratio": 4.0,
    "eeg_temporal_bins": 8,
    # 31 samples = 62 ms at the 500 Hz of the SI_Hom recordings (the kernel
    # must be odd); the same duration was 15 samples at 250 Hz.
    "eeg_kernel_size": 31,
    "spatial_readout": "learned",
}
