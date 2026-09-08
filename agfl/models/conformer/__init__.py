from .._shared import ModelSpec
from .config import EEG_DEFAULTS, ECG_DEFAULTS
from .eeg import EEGModel
from .ecg import ECGModel

SPEC = ModelSpec('conformer', EEG_DEFAULTS, {'eeg': EEGModel, 'ecg': ECGModel},
                 {'eeg': EEG_DEFAULTS, 'ecg': ECG_DEFAULTS}, comparison_family='conformer_v2')
