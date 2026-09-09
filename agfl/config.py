"""Resolved JSON/TOML configuration; no model decisions belong in the CLI."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import tomllib


TRAINING_DEFAULTS = {
    "epochs": 50, "batch_size": 64, "optimizer": "adamw", "learning_rate": 3e-4,
    "weight_decay": 1e-3, "momentum": 0.9, "loss": "focal",
    "class_weights": "balanced", "focal_gamma": 2.0, "scheduler": "warmup_cosine",
    "warmup_epochs": 10, "min_lr_ratio": 0.0, "checkpoint_criterion": "accuracy",
    "gradient_clip": None, "num_workers": 0, "amp": False,
    "augmentation": {"shift": 0, "scale": 0.0, "noise": 0.0},
}
DEFAULTS = {
    "subject_id": None,
    "schema_version": 2, "model": "eegnet", "attention": "agfl", "dataset": "eeg", "model_variant": "auto",
    "seeds": [0, 1, 2, 3, 4], "deterministic": True, "device": "cuda", "threads": 1,
    "output_dir": "results", "split_dir": "splits", "data": {}, "model_options": {}, "attention_options": {},
    "split": {"protocol": "group", "train": 0.6, "validation": 0.2, "test": 0.2},
    "training": TRAINING_DEFAULTS,
}


def merge(base, update):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read_config(path):
    path = Path(path)
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text())
    return json.loads(path.read_text())


def apply_overrides(config, overrides):
    result = copy.deepcopy(config)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value: {item}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def resolve_config(config):
    from .datasets import get_dataset_spec
    from .models import get_model_spec
    from .attention import get_attention_spec

    config = copy.deepcopy(config)
    if config.get('schema_version', 2) != 2:
        raise ValueError('Training now requires schema_version=2 with separate model and attention keys. Start a new configuration; old results remain readable by analyze.')
    if "seed" in config:
        if 'seeds' in config:
            raise ValueError('Use seed or seeds, not both')
        config["seeds"] = [config.pop("seed")]
    allowed = set(DEFAULTS) | {"resolved_metadata", "dataset_fingerprint", "expected_split_id", "provenance", "comparison_family"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    dataset_spec = get_dataset_spec(config.get('dataset', DEFAULTS['dataset']))
    resolved = merge(merge(DEFAULTS, dataset_spec.experiment_defaults or {}), config)
    model_spec = get_model_spec(resolved["model"])
    family = model_spec.comparison_family or model_spec.key
    if config.get('comparison_family') not in (None, family):
        raise ValueError('Saved comparison family differs from the selected model')
    resolved['comparison_family'] = family
    unknown = set(config.get('data', {})) - set(dataset_spec.defaults)
    if unknown:
        raise ValueError(f'Unknown dataset settings: {sorted(unknown)}')
    # Dataset dictionaries such as label_map are replaced whole, not merged:
    # a declared subset must not silently retain historical extra labels.
    resolved["data"] = {**copy.deepcopy(dataset_spec.defaults), **copy.deepcopy(config.get("data", {}))}
    options = config.get("model_options", {})
    model_defaults = model_spec.defaults_for(dataset_spec.modality)
    unknown = set(options) - set(model_defaults)
    if unknown:
        raise ValueError(f"Unknown options for {resolved['model']}: {sorted(unknown)}")
    resolved["model_options"] = merge(model_defaults, options)
    attention_spec = get_attention_spec(resolved['attention'])
    attention_options = config.get('attention_options', {})
    unknown = set(attention_options) - set(attention_spec.defaults)
    if unknown:
        raise ValueError(f"Unknown options for attention {resolved['attention']}: {sorted(unknown)}")
    resolved['attention_options'] = merge(model_spec.attention_defaults_for(resolved['attention']), attention_options)
    if type(resolved['attention_options']['heads']) is not int or resolved['attention_options']['heads'] < 1:
        raise ValueError('attention_options.heads must be a positive integer')
    resolved['training'] = merge(merge(TRAINING_DEFAULTS, model_spec.training_defaults_for(dataset_spec.modality)), config.get('training', {}))
    unknown = set(resolved["training"]) - set(TRAINING_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    modality = dataset_spec.modality
    if resolved["model_variant"] == "auto":
        resolved["model_variant"] = modality
    if resolved["model_variant"] != modality:
        raise ValueError("model_variant must match the dataset modality")
    seeds = resolved["seeds"]
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique integers")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("seeds must be integers in [0, 2**32)")
    t = resolved["training"]
    if resolved['schema_version'] != 2:
        raise ValueError('Unsupported configuration schema_version')
    if type(resolved['threads']) is not int or resolved['threads'] < 1:
        raise ValueError('threads must be a positive integer')
    if type(resolved['deterministic']) is not bool or type(t['amp']) is not bool:
        raise ValueError('deterministic and amp must be booleans')
    for key in ('learning_rate', 'weight_decay', 'focal_gamma', 'momentum', 'min_lr_ratio'):
        if not isinstance(t[key], (int, float)) or not math.isfinite(t[key]):
            raise ValueError(f'training.{key} must be finite')
    for key in ("epochs", "batch_size"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if t["learning_rate"] <= 0 or t["weight_decay"] < 0:
        raise ValueError("learning_rate must be positive and weight_decay nonnegative")
    for key, choices in {
        "optimizer": {"adamw", "adam", "sgd"},
        "loss": {"cross_entropy", "focal"}, "class_weights": {"balanced", "none"},
        "scheduler": {"none", "cosine", "warmup_cosine", "plateau"},
        "checkpoint_criterion": {"loss", "accuracy", "f1", "roc_auc"},
    }.items():
        if t[key] not in choices:
            raise ValueError(f"training.{key} must be one of {sorted(choices)}")
    if t["focal_gamma"] < 0 or (t["gradient_clip"] is not None and
            (not math.isfinite(t['gradient_clip']) or t["gradient_clip"] <= 0)):
        raise ValueError("Invalid focal_gamma or gradient_clip")
    if type(t["num_workers"]) is not int or t["num_workers"] < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if type(t['warmup_epochs']) is not int or t["warmup_epochs"] < 0 or not 0 <= t["min_lr_ratio"] <= 1:
        raise ValueError("Invalid warmup_epochs or min_lr_ratio")
    augmentation = t['augmentation']
    if set(augmentation) != {'shift', 'scale', 'noise'}:
        raise ValueError('augmentation accepts only shift, scale, noise')
    if type(augmentation['shift']) is not int or augmentation['shift'] < 0:
        raise ValueError('augmentation.shift must be a nonnegative integer')
    if not 0 <= augmentation['scale'] < 1 or not math.isfinite(augmentation['noise']) or augmentation['noise'] < 0:
        raise ValueError('Invalid augmentation scale or noise')
    for key in ("output_dir", "split_dir"):
        resolved[key] = str(Path(resolved[key]).expanduser().resolve())
    for key in ("data_dir", "path", "labels_dir"):
        if resolved["data"].get(key):
            resolved["data"][key] = str(Path(resolved["data"][key]).expanduser().resolve())
    return resolved


def resolve_experiments(config):
    """Dataset-owned expansion is shared by CLI previews and Python launches."""
    from .datasets import get_dataset_spec
    resolved = resolve_config(config)
    expand = get_dataset_spec(resolved['dataset']).expand_experiments
    return expand(resolved) if expand is not None else [resolved]


def experiment_identity(config):
    return digest({k: v for k, v in config.items() if k not in {
        "seeds", "output_dir", "split_dir", "expected_split_id", "provenance"
    }})[:20]


def comparison_identity(config):
    if config.get('schema_version', 1) >= 2:
        return digest({
            'dataset': config['dataset'], 'data': config['data'], 'split': config['split'],
            'training': config['training'], 'model_variant': config['model_variant'],
            'model': config['model'], 'backbone': config['comparison_family'],
            'model_options': config['model_options'], 'heads': config['attention_options']['heads'],
            'deterministic': config['deterministic'], 'device': config['device'], 'threads': config['threads'],
            'dataset_fingerprint': config.get('dataset_fingerprint'),
            'source_sha256': config.get('provenance', {}).get('source_sha256'),
            'packages': config.get('provenance', {}).get('packages'),
        })[:20]
    # Preserve v1 identity calculation solely for analysis of already saved runs.
    shared = {k: v for k, v in config["model_options"].items() if k in {
        "dim", "depth", "heads", "dropout", "mlp_ratio", "eeg_temporal_bins",
        "eeg_kernel_size", "ecg_patch_size",
    }}
    return digest({
        "dataset": config["dataset"], "data": config["data"], "split": config["split"],
        "training": config["training"], "model_variant": config["model_variant"],
        "backbone": config['comparison_family'],
        "shared_architecture": shared, "deterministic": config["deterministic"],
        "device": config["device"], "threads": config["threads"],
        "dataset_fingerprint": config.get("dataset_fingerprint"),
        "source_sha256": config.get('provenance', {}).get('source_sha256'),
        "packages": config.get('provenance', {}).get('packages'),
    })[:20]


def experiment_selection(config):
    """Read old result labels without changing stored configs or their identities."""
    if config.get('schema_version', 1) >= 2:
        return config['model'], config['attention']
    key = config['model']
    mechanisms = {'agfl': 'agfl', 'transformer': 'mha', 'performer': 'performer',
                  'linformer': 'linformer', 'nystromformer': 'nystromformer'}
    if key in mechanisms:
        return 'signal_transformer', mechanisms[key]
    options = config['model_options']
    mechanism = options.get('attention_type', options.get('mode', 'agfl'))
    return key.removeprefix('legacy_'), 'mha' if mechanism == 'standard' else mechanism
