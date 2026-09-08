"""Discover attention mechanisms independently of model/backbone selection."""
from copy import deepcopy
from dataclasses import dataclass
from importlib import import_module
from pkgutil import iter_modules


@dataclass(frozen=True)
class AttentionSpec:
    key: str
    defaults: dict
    constructor: object

    def build(self, options, dim, tokens):
        from agfl.config import merge
        unknown = set(options) - set(self.defaults)
        if unknown:
            raise ValueError(f'Unknown {self.key} attention_options: {sorted(unknown)}')
        settings = {**merge(self.defaults, options), 'dim': dim}
        heads = settings['heads']
        if type(heads) is not int or heads < 1 or dim % heads:
            raise ValueError(f'attention_options.heads={heads} must divide the model feature dimension {dim}')
        return self.constructor(settings, tokens)


def attention_registry():
    registry = {}
    for module in sorted(iter_modules(__path__), key=lambda item: item.name):
        if module.ispkg and not module.name.startswith('_'):
            spec = getattr(import_module(f'{__name__}.{module.name}'), 'SPEC', None)
            if spec is not None:
                if spec.key in registry:
                    raise ValueError(f'Duplicate attention key: {spec.key}')
                registry[spec.key] = spec
    return registry


def get_attention_spec(key):
    registry = attention_registry()
    if key not in registry:
        raise ValueError(f'Unknown attention {key!r}; available: {", ".join(registry)}')
    return registry[key]


def available_attentions():
    return sorted(attention_registry())


class AttentionFactory:
    """Inject attention without shifting initialization of the surrounding model."""
    def __init__(self, key, options):
        self.spec, self.options = get_attention_spec(key), deepcopy(options)
        self.count = 0

    def __call__(self, dim, tokens, layer_idx=0, depth=1, token_axis='time'):
        import torch
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator().manual_seed((torch.initial_seed() + 104729 * (self.count + 1)) % (2**63))
            torch.set_rng_state(generator.get_state())
            module = self.spec.build(self.options, dim, tokens)
        self.count += 1
        module.attention_key = self.spec.key
        module.is_attention = True
        module.num_tokens = tokens
        module.token_axis = token_axis
        module.layer_idx, module.depth = layer_idx, depth
        return module
