"""Deterministic fixtures for pipeline checks, never scientific benchmarks."""
import numpy as np

from .base import SignalDataset, normalize_samples
from .registry import register_dataset


def _synthetic(config, modality):
    rng = np.random.default_rng(int(config["data_seed"]))
    subjects = int(config["subjects"])
    per_subject = int(config["samples_per_subject"])
    channels, samples, classes = (int(config[k]) for k in ("channels", "samples", "num_classes"))
    if min(subjects, per_subject, channels, samples) < 1 or classes < 2:
        raise ValueError("Synthetic dimensions must be positive and num_classes >= 2")
    if per_subject < classes:
        raise ValueError("Synthetic samples_per_subject must cover every class")
    x = rng.normal(0, 0.3, (subjects * per_subject, channels, samples)).astype(np.float32)
    y = np.tile(np.arange(per_subject) % classes, subjects)
    t = np.linspace(0, 1, samples, endpoint=False)
    for index, label in enumerate(y):
        wave = np.sin(2 * np.pi * (label + 2) * t)
        spatial = np.cos(np.arange(channels) + label) if modality == "eeg" else np.ones(channels)
        x[index] += spatial[:, None] * wave[None, :]
    groups = np.repeat([f"subject_{s:03d}" for s in range(subjects)], per_subject)
    ids = [f"{modality}:fixture:{index:06d}" for index in range(len(y))]
    return SignalDataset(normalize_samples(x, config["normalization"]), y, groups, ids,
                         {"dataset": f"synthetic_{modality}", "modality": modality,
                          "num_classes": classes, "synthetic": True, "preprocessing": config,
                          "protocol_version": "synthetic-v1"})


@register_dataset("synthetic_eeg", "eeg", {
    "subjects": 9, "samples_per_subject": 16, "channels": 8, "samples": 128,
    "num_classes": 4, "data_seed": 1234, "normalization": "per_sample",
}, "Small generated fixture; excluded from claims about EEG accuracy")
def synthetic_eeg(config):
    return _synthetic(config, "eeg")


@register_dataset("synthetic_ecg", "ecg", {
    "subjects": 9, "samples_per_subject": 16, "channels": 1, "samples": 128,
    "num_classes": 2, "data_seed": 1234, "normalization": "per_sample",
}, "Small generated fixture; excluded from claims about ECG accuracy")
def synthetic_ecg(config):
    return _synthetic(config, "ecg")
