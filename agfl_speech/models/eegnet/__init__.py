from .._shared import ModelSpec
from .config import EEG_DEFAULTS
from .eeg import EEGModel

SPEC = ModelSpec('eegnet', EEG_DEFAULTS, {'eeg': EEGModel}, comparison_family='eegnet_v2')
