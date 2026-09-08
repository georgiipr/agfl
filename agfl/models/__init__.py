"""Model-owned experiments discovered from packages exposing ``SPEC``."""

from importlib import import_module
from pkgutil import iter_modules


def model_registry():
    registry = {}
    for module in sorted(iter_modules(__path__), key=lambda item: item.name):
        if not module.ispkg or module.name.startswith("_"):
            continue
        package = import_module(f"{__name__}.{module.name}")
        spec = getattr(package, "SPEC", None)
        if spec is not None:
            if spec.key in registry:
                raise ValueError(f"Duplicate model key: {spec.key}")
            registry[spec.key] = spec
    return registry


def get_model_spec(key):
    registry = model_registry()
    try:
        return registry[key]
    except KeyError:
        from agfl.attention import available_attentions
        if key in available_attentions() or key == 'transformer':
            raise ValueError(f'{key!r} selects an attention mechanism, not a backbone. Choose --model eegnet (or another model) and --attention {"mha" if key == "transformer" else key}.') from None
        raise ValueError(f"Unknown model {key!r}; available: {', '.join(registry)}") from None


def available_models():
    return sorted(model_registry())
