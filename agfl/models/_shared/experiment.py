"""Each model directly owns this reusable data/train/evaluate policy."""

from dataclasses import dataclass
from copy import deepcopy


@dataclass(frozen=True)
class ModelSpec:
    key: str
    defaults: dict
    variants: dict
    variant_defaults: dict | None = None
    comparison_family: str | None = None
    attention_defaults: dict | None = None

    def defaults_for(self, modality):
        if modality not in self.variants:
            raise ValueError(f"No {self.key} implementation for modality {modality!r}")
        return deepcopy((self.variant_defaults or {}).get(modality, self.defaults))

    def training_defaults_for(self, modality):
        from .policy import TRAINING_DEFAULTS
        return deepcopy(TRAINING_DEFAULTS[modality])

    def attention_defaults_for(self, attention):
        from agfl.attention import get_attention_spec
        defaults = deepcopy(get_attention_spec(attention).defaults)
        overrides = self.attention_defaults or {}
        defaults.update(deepcopy(overrides.get('all', {})))
        defaults.update(deepcopy(overrides.get(attention, {})))
        return defaults

    def build(self, model_options, metadata, attention='agfl', attention_options=None):
        from agfl.attention import AttentionFactory
        from agfl.config import merge
        defaults = self.defaults_for(metadata['modality'])
        unknown = set(model_options) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {self.key} model_options: {sorted(unknown)}")
        options = {**defaults, **model_options}
        settings = merge(self.attention_defaults_for(attention), attention_options or {})
        return self.variants[metadata['modality']](options, metadata, AttentionFactory(attention, settings))

    def prepare_data(self, config, seed):
        from agfl.datasets import prepare_data

        return prepare_data(config, seed)

    def run(self, config, bundle, split, run_dir):
        from agfl.engine import run_training

        return run_training(config, bundle, split, run_dir)

    def evaluate(self, *args, **kwargs):
        from agfl.engine import evaluate

        return evaluate(*args, **kwargs)
