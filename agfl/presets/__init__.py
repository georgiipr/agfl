"""Packaged, editable JSON presets for the intended execution environment."""
from importlib.resources import files
import json


def retired_presets():
    return set(json.loads(files(__package__).joinpath('_retired.json').read_text())['presets'])


def preset_names():
    retired = retired_presets()
    return sorted(path.name[:-5] for path in files(__package__).iterdir()
                  if path.name.endswith('.json') and not path.name.startswith('_')
                  and path.name[:-5] not in retired)


def load_preset(name):
    if name in retired_presets():
        raise ValueError(f'{name!r} is a retired temporal EEG study. Use eegnet-interchannel '
                         'or eegnet-interchannel-comparison. Original configurations and '
                         'saved results are retained for historical analysis only.')
    if name not in preset_names():
        raise ValueError(f'Unknown preset {name!r}; choose from {preset_names()}')
    return json.loads(files(__package__).joinpath(name + '.json').read_text())
