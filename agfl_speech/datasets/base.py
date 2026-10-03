"""Dataset contract: finite signals [sample, channel, time] and stable identities."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_fingerprint(path: str | Path) -> dict:
    """Hash actual source bytes; paths and modification times are not identities."""
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"name": path.name, "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def normalize_samples(x: np.ndarray, normalization: str) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 3 or min(x.shape) < 1 or not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x):
        raise ValueError('Normalization expects real, nonempty [N,C,T] signals')
    if not np.isfinite(x).all():
        raise ValueError('Normalization input contains non-finite signals')
    if normalization == "per_sample":
        # The original protocol standardizes each channel within its own trial.
        # No validation/test statistics enter a training sample.
        mean = x.mean(axis=-1, keepdims=True, dtype=np.float64)
        std = x.std(axis=-1, keepdims=True, dtype=np.float64)
        return ((x - mean) / (std + 1e-8)).astype(np.float32)
    if normalization in {"none", "train_channel"}:
        # train_channel is fitted by the shared engine using training indices only.
        return x.astype(np.float32, copy=False)
    raise ValueError("normalization must be per_sample, train_channel, or none")


@dataclass
class SignalDataset:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    sample_ids: list[str]
    metadata: dict
    _fingerprint: str = field(init=False, repr=False)

    def __post_init__(self):
        self.x = np.ascontiguousarray(self.x, dtype=np.float32)
        raw_labels = np.asarray(self.y)
        if not np.issubdtype(raw_labels.dtype, np.integer):
            if not np.isfinite(raw_labels).all() or not np.equal(raw_labels, np.floor(raw_labels)).all():
                raise ValueError("Labels must be finite integer class indices")
        self.y = np.ascontiguousarray(raw_labels, dtype=np.int64)
        self.groups = np.asarray(self.groups, dtype=str)
        self.sample_ids = [str(item) for item in self.sample_ids]
        if self.x.ndim != 3 or min(self.x.shape) < 1:
            raise ValueError("Dataset signals must have nonempty shape [samples, channels, time]")
        n, channels, samples = self.x.shape
        if self.y.shape != (n,) or self.groups.shape != (n,) or len(self.sample_ids) != n:
            raise ValueError("Signals, labels, subject groups, and sample IDs must have matching lengths")
        if len(set(self.sample_ids)) != n:
            raise ValueError("Sample IDs must be unique")
        if not np.isfinite(self.x).all():
            raise ValueError("Dataset contains non-finite signals after preprocessing")
        if not all(self.sample_ids) or not np.all(self.groups != ""):
            raise ValueError("Sample IDs and subject groups must be nonempty")
        num_classes = int(self.metadata.get("num_classes", self.y.max() + 1))
        if num_classes < 2 or self.y.min() < 0 or self.y.max() >= num_classes:
            raise ValueError("Labels must lie in [0, num_classes); classification needs at least two classes")
        if self.metadata.get("modality") != "eeg":
            raise ValueError("Dataset metadata must declare modality eeg")
        self.metadata.update(channels=channels, samples=samples, num_classes=num_classes,
                             num_samples=n, class_counts=np.bincount(self.y, minlength=num_classes).tolist())
        self._fingerprint = self._content_fingerprint()

    def _content_fingerprint(self) -> str:
        digest = hashlib.sha256()
        for array in (self.x, self.y):
            digest.update(str(array.shape).encode())
            digest.update(str(array.dtype).encode())
            digest.update(memoryview(array).cast("B"))
        digest.update(canonical_json(self.groups.tolist()).encode())
        digest.update(canonical_json(self.sample_ids).encode())
        # Absolute data paths do not alter identity when the same dataset is relocated.
        identity_metadata = dict(self.metadata)
        settings = dict(identity_metadata.get("preprocessing", {}))
        settings.pop("data_dir", None)
        identity_metadata["preprocessing"] = settings
        identity_metadata.pop("data_dir", None)
        digest.update(canonical_json(identity_metadata).encode())
        return digest.hexdigest()

    @property
    def fingerprint(self) -> str:
        # Recheck bytes if callers have modified arrays or metadata after loading.
        # The shared engine normally works on a separate normalized data view.
        return self._content_fingerprint()

    def __len__(self):
        return len(self.y)
