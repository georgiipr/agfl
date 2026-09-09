"""Discover dataset plugins without model-specific selection in the entry point."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import importlib
import pkgutil
from typing import Callable

from .base import SignalDataset


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    modality: str
    defaults: dict
    loader: Callable[[dict], SignalDataset]
    description: str = ""
    experiment_defaults: dict | None = None
    expand_experiments: Callable | None = None


_REGISTRY: dict[str, DatasetSpec] = {}
_DISCOVERED = False


def register_dataset(key: str, modality: str, defaults: dict, description: str = "", *,
                     experiment_defaults=None, expand_experiments=None):
    def decorator(loader):
        if key in _REGISTRY:
            raise ValueError(f"Dataset key {key!r} is already registered")
        _REGISTRY[key] = DatasetSpec(key, modality, deepcopy(defaults), loader, description,
                                     deepcopy(experiment_defaults), expand_experiments)
        return loader
    return decorator


def _discover():
    global _DISCOVERED
    if not _DISCOVERED:
        package = importlib.import_module(__package__)
        for module in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
            if not module.name.startswith("_") and module.name not in {"base", "registry", "splits"}:
                importlib.import_module(f"{__package__}.{module.name}")
        _DISCOVERED = True


def get_dataset_spec(key: str) -> DatasetSpec:
    _discover()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset {key!r}; available: {', '.join(sorted(_REGISTRY))}")
    return _REGISTRY[key]


def list_datasets() -> list[str]:
    _discover()
    return sorted(_REGISTRY)


def load_dataset(key: str, data_cfg: dict | None = None) -> SignalDataset:
    spec = get_dataset_spec(key)
    config = deepcopy(spec.defaults)
    supplied = data_cfg or {}
    unknown = set(supplied) - set(config)
    if unknown:
        raise ValueError(f"Unknown {key} data settings: {', '.join(sorted(unknown))}")
    config.update(supplied)
    dataset = spec.loader(config)
    if dataset.metadata["modality"] != spec.modality:
        raise ValueError(f"Dataset {key} loader returned the wrong modality")
    return dataset
