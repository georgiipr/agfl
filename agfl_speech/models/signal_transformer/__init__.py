"""Explicit generic backbone retained alongside the named model families."""

from .._shared import ModelSpec
from .config import DEFAULTS
from .eeg import EEGModel

SPEC = ModelSpec('signal_transformer', DEFAULTS, {'eeg': EEGModel},
                 comparison_family='signal_transformer_v2')

__all__ = ["ModelSpec"]
