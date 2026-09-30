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
        return self._build(model_options, metadata, attention, attention_options, historical=False)

    def rebuild_saved(self, model_options, metadata, attention='agfl', attention_options=None):
        """Reconstruct checkpoint semantics; never used to start new training."""
        return self._build(model_options, metadata, attention, attention_options, historical=True)

    def _build(self, model_options, metadata, attention, attention_options, *, historical):
        from agfl.attention import AttentionFactory
        from agfl.config import merge
        from .modality import validate_attention_domain, validate_model_graph
        defaults = self.defaults_for(metadata['modality'])
        unknown = set(model_options) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown {self.key} model_options: {sorted(unknown)}")
        options = {**defaults, **model_options}
        if historical and 'electrode_dim' in defaults and 'electrode_dim' not in model_options:
            # Old EEGNet electrode checkpoints used flattened temporal features.
            options['electrode_dim'] = None
        if historical and 'electrode_architecture' in defaults and 'electrode_architecture' not in model_options:
            # Both pre-fusion electrode layouts must retain their exact modules
            # and classifier shapes when loading already trained checkpoints.
            options['electrode_architecture'] = 'compact'
        settings = merge(self.attention_defaults_for(attention), attention_options or {})
        if not historical:
            validate_attention_domain(metadata['modality'], options, settings)
        factory = AttentionFactory(attention, settings, modality=metadata['modality'],
                                   historical=historical, channels=metadata['channels'])
        model = self.variants[metadata['modality']](options, metadata, factory)
        if not historical:
            validate_model_graph(model, metadata)
        return model

    def prepare_data(self, config, seed):
        from agfl.datasets import prepare_data

        return prepare_data(config, seed)

    def run(self, config, bundle, split, run_dir):
        from agfl.engine import run_training

        return run_training(config, bundle, split, run_dir)

    def evaluate(self, *args, **kwargs):
        from agfl.engine import evaluate

        return evaluate(*args, **kwargs)
