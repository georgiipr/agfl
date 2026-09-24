from .._shared import ModelSpec
from .config import EEG_DEFAULTS, ECG_DEFAULTS
from .eeg import EEGModel
from .ecg import ECGModel


def _model_class(options, modality):
    mode = options.get('temporal_statistics', 'mean')
    if mode == 'mean':
        return EEGModel if modality == 'eeg' else ECGModel
    if mode == 'mean_logvar':
        # Imported only for an explicit historical power-study configuration.
        # Keeping this implementation isolated preserves checkpoint replay
        # without changing the EEGNet used by the active AGFL comparisons.
        from .reproduction import EEGPowerReproduction, ECGPowerReproduction
        return EEGPowerReproduction if modality == 'eeg' else ECGPowerReproduction
    raise ValueError('EEGNet temporal_statistics must be mean or mean_logvar')


def build_eeg(options, metadata, attention):
    return _model_class(options, 'eeg')(options, metadata, attention)


def build_ecg(options, metadata, attention):
    return _model_class(options, 'ecg')(options, metadata, attention)


SPEC = ModelSpec('eegnet', EEG_DEFAULTS, {'eeg': build_eeg, 'ecg': build_ecg},
                 {'eeg': EEG_DEFAULTS, 'ecg': ECG_DEFAULTS}, comparison_family='eegnet_v2')
