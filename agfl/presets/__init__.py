"""Packaged, editable JSON presets for the intended execution environment."""
from importlib.resources import files
import json


def preset_names():
    return sorted(path.name[:-5] for path in files(__package__).iterdir() if path.name.endswith('.json'))


def load_preset(name):
    if name not in preset_names():
        raise ValueError(f'Unknown preset {name!r}; choose from {preset_names()}')
    return json.loads(files(__package__).joinpath(name + '.json').read_text())
