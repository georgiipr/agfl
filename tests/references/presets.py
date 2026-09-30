"""Saved-config fixtures for historical reporting tests, never training APIs."""
from copy import deepcopy
import json
from pathlib import Path

from agfl.config import resolve_experiments


def load_archived_preset(name):
    root = Path(__file__).parents[2] / 'agfl/presets/archive/eeg_temporal'
    return json.loads((root / (name + '.json')).read_text())


def archived_experiments(config):
    """Construct mock saved records after normalizing paths/defaults.

    The active resolver must reject these configurations. Restore historical
    axis/flags only in a test fixture used by saved-artifact consumers.
    """
    source = deepcopy(config)
    assert source['dataset'] == 'eeg' and source['model_options']['attention_axis'] == 'time'
    safe = deepcopy(source)
    safe['model_options'].update(attention_axis='electrode', temporal_statistics='mean')
    if 'temporal_bias' in safe.get('attention_options', {}):
        safe['attention_options']['temporal_bias'] = False
    records = resolve_experiments(safe)
    for result in records:
        result['model_options'].update(source['model_options'])
        result['model_options'].pop('electrode_dim', None)
        result['model_options'].pop('electrode_architecture', None)
        result['attention_options'].update(source.get('attention_options', {}))
    return records
