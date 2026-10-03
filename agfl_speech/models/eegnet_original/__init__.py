"""EEGNet as configured by the SI_Hom dataset authors, hosting electrode attention."""
from .._shared import ModelSpec
from .config import DEFAULTS
from .model import OriginalEEGNet

SPEC = ModelSpec('eegnet_original', DEFAULTS, {'eeg': OriginalEEGNet},
                 comparison_family='eegnet_original_v1')
