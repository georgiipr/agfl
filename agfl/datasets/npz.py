"""A simple extension point for preprocessed arrays with explicit subject IDs."""
from pathlib import Path
import numpy as np

from .base import SignalDataset, normalize_samples, source_fingerprint
from .registry import register_dataset


def _load(config, modality):
    path = Path(config["path"]).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"NPZ dataset not found: {path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {"x", "y", "groups"}
        if not required.issubset(archive.files):
            raise ValueError("NPZ must contain x [N,C,T], y [N], and groups [N] subject IDs")
        x, y, groups = archive["x"], archive["y"], archive["groups"]
        ids = archive["sample_ids"].astype(str).tolist() if "sample_ids" in archive else [f"npz:{i}" for i in range(len(y))]
    if x.ndim != 3:
        raise ValueError("NPZ x must use explicit [sample, channel, time] dimensions")
    metadata = {"dataset": f"npz_{modality}", "modality": modality, "preprocessing": config,
                "sources": [source_fingerprint(path)], "protocol_version": "npz-v1"}
    if config["num_classes"] is not None:
        metadata["num_classes"] = int(config["num_classes"])
    return SignalDataset(normalize_samples(x, config["normalization"]), y, groups, ids, metadata)


@register_dataset("npz_eeg", "eeg", {"path": "", "num_classes": None, "normalization": "per_sample"})
def npz_eeg(config):
    return _load(config, "eeg")


@register_dataset("npz_ecg", "ecg", {"path": "", "num_classes": None, "normalization": "per_sample"})
def npz_ecg(config):
    return _load(config, "ecg")
