"""Persist dataset/seed splits once and reuse them for every mechanism."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .base import SignalDataset, canonical_json


SPLIT_VERSION = "agfl-splits-v2"


def _fractions(config):
    aliases = {"train": "train_fraction", "validation": "validation_fraction", "test": "test_fraction"}
    for name, alias in aliases.items():
        if name in config and alias in config and config[name] != config[alias]:
            raise ValueError(f"Conflicting split fractions {name} and {alias}")
    fractions = {name: float(config.get(name, config.get(alias, default)))
                 for (name, alias), default in zip(aliases.items(), (0.6, 0.2, 0.2))}
    if any(not np.isfinite(value) or value <= 0 or value >= 1 for value in fractions.values()):
        raise ValueError("train, validation, and test split fractions must each lie strictly between 0 and 1")
    if not np.isclose(sum(fractions.values()), 1.0):
        raise ValueError("train + validation + test fractions must sum to 1")
    return fractions


def _sizes(count, fractions):
    if count < 3:
        raise ValueError("Three independent partitions require at least three samples/groups per splitting unit")
    validation = max(1, int(round(count * fractions["validation"])))
    test = max(1, int(round(count * fractions["test"])))
    while validation + test >= count:
        if validation >= test and validation > 1:
            validation -= 1
        elif test > 1:
            test -= 1
        else:
            raise ValueError("Cannot create nonempty train, validation, and test partitions")
    return count - validation - test, validation, test


def _group_split(bundle, fractions, rng):
    unique = np.unique(bundle.groups)
    train_size, validation_size, _ = _sizes(len(unique), fractions)
    groups = rng.permutation(unique)
    assignments = {"train": groups[:train_size],
                   "validation": groups[train_size:train_size + validation_size],
                   "test": groups[train_size + validation_size:]}
    return {name: np.flatnonzero(np.isin(bundle.groups, selected)).tolist()
            for name, selected in assignments.items()}


def _stratified_split(bundle, fractions, rng, subset=None):
    indices = np.arange(len(bundle)) if subset is None else np.asarray(subset)
    partitions = {"train": [], "validation": [], "test": []}
    for label in np.unique(bundle.y[indices]):
        members = rng.permutation(indices[bundle.y[indices] == label])
        try:
            train_size, validation_size, _ = _sizes(len(members), fractions)
        except ValueError as error:
            raise ValueError(f"Class {label} needs at least three samples for a stratified three-way split") from error
        partitions["train"].extend(members[:train_size].tolist())
        partitions["validation"].extend(members[train_size:train_size + validation_size].tolist())
        partitions["test"].extend(members[train_size + validation_size:].tolist())
    return {name: sorted(values) for name, values in partitions.items()}


def _session_split(bundle, config, rng):
    sessions = np.asarray(bundle.metadata.get("sample_sessions", []))
    if sessions.shape != bundle.y.shape or set(sessions) != {"T", "E"}:
        raise ValueError("Session protocol requires both labeled EEG T and E sessions")
    fraction = float(config.get("validation_within_train", 0.2))
    if not 0 < fraction < 1:
        raise ValueError("validation_within_train must lie strictly between zero and one")
    partitions = {"train": [], "validation": [], "test": np.flatnonzero(sessions == "E").tolist()}
    training = np.flatnonzero(sessions == "T")
    for label in np.unique(bundle.y[training]):
        members = rng.permutation(training[bundle.y[training] == label])
        if len(members) < 2:
            raise ValueError(f"Class {label} has too few training-session trials to create validation")
        n_validation = min(len(members) - 1, max(1, int(round(len(members) * fraction))))
        partitions["validation"].extend(members[:n_validation].tolist())
        partitions["train"].extend(members[n_validation:].tolist())
    return {name: sorted(values) for name, values in partitions.items()}


def validate_split(bundle: SignalDataset, split: dict) -> None:
    if split.get("fingerprint") != bundle.fingerprint:
        raise ValueError("Persisted split dataset fingerprint differs from the loaded dataset")
    parts = {name: split[name] for name in ("train", "validation", "test")}
    diagnostic = split.get("diagnostic_only", False)
    for name, indices in parts.items():
        if (not indices and not (diagnostic and name == "test")) or any(type(i) is not int for i in indices):
            raise ValueError(f"Split {name} must be a nonempty list of integer indices")
        if len(indices) != len(set(indices)) or any(i < 0 or i >= len(bundle) for i in indices):
            raise ValueError(f"Split {name} contains duplicate or out-of-range indices")
    sets = {name: set(indices) for name, indices in parts.items()}
    if any(sets[a] & sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Train, validation, and test samples overlap")
    if set.union(*sets.values()) != set(range(len(bundle))):
        raise ValueError("Split does not cover the complete dataset")
    observed_classes = set(bundle.y[parts["train"]].tolist())
    required_classes = set(range(bundle.metadata["num_classes"]))
    if observed_classes != required_classes:
        raise ValueError(f"Training split lacks classes {sorted(required_classes - observed_classes)}; "
                         "change the declared cohort or splitting protocol, not the selected model")
    if split["protocol"] == "group":
        group_sets = {name: set(bundle.groups[indices]) for name, indices in parts.items()}
        if any(group_sets[a] & group_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
            raise ValueError("Subject groups overlap despite a subject-independent split protocol")
    if "sample_ids" in split:
        expected_ids = {name: [bundle.sample_ids[i] for i in indices] for name, indices in parts.items()}
        if split["sample_ids"] != expected_ids:
            raise ValueError("Persisted sample IDs do not match split indices")
    if "groups" in split:
        expected_groups = {name: sorted(set(bundle.groups[indices].tolist())) for name, indices in parts.items()}
        if split["groups"] != expected_groups:
            raise ValueError("Persisted subject groups do not match split indices")
    if "partition_digest" in split:
        actual = hashlib.sha256(canonical_json(parts).encode()).hexdigest()
        if split["partition_digest"] != actual:
            raise ValueError("Persisted split partition checksum was modified")


def get_split(bundle: SignalDataset, split_cfg: dict | None, seed: int,
              split_dir: str | Path = "results/splits") -> dict:
    supplied = dict(split_cfg or {})
    allowed = {"protocol", "train", "validation", "test", "train_fraction", "validation_fraction", "test_fraction",
               "validation_within_train", "allow_subject_overlap"}
    if set(supplied) - allowed:
        raise ValueError(f"Unknown split settings: {sorted(set(supplied) - allowed)}")
    protocol = supplied.get("protocol", "group")
    protocol = {"subject": "group", "trial": "stratified", "subject_dependent": "stratified"}.get(protocol, protocol)
    if protocol not in {"group", "stratified", "session", "diagnostic_eeg"}:
        raise ValueError("split.protocol must be group, stratified, session, or diagnostic diagnostic_eeg")
    config = dict(supplied, protocol=protocol)
    if protocol in {"group", "stratified"}:
        config.update(_fractions(supplied))
        for name in ("train_fraction", "validation_fraction", "test_fraction"):
            config.pop(name, None)
    identity = {"version": SPLIT_VERSION, "fingerprint": bundle.fingerprint,
                "seed": int(seed), "config": config}
    split_id = hashlib.sha256(canonical_json(identity).encode()).hexdigest()
    directory = Path(split_dir).expanduser().resolve()
    path = directory / f"{split_id}.json"
    if path.is_file():
        stored = json.loads(path.read_text())
        if stored.get("identity") != identity or stored.get("split_id") != split_id:
            raise ValueError(f"Persisted split identity was modified: {path}")
        validate_split(bundle, stored)
        return stored
    rng = np.random.default_rng(int(seed))
    if protocol == "group":
        partitions = _group_split(bundle, config, rng)
    elif protocol == "stratified":
        if bundle.metadata["modality"] == "ecg" and not config.get("allow_subject_overlap", False):
            raise ValueError("ECG stratified beat splitting can leak overlapping beats and patient identity; "
                             "use group or explicitly set allow_subject_overlap for a labeled diagnostic")
        partitions = _stratified_split(bundle, config, rng)
    elif protocol == "session":
        partitions = _session_split(bundle, config, rng)
    else:
        if bundle.metadata["modality"] != "eeg":
            raise ValueError("diagnostic_eeg is only defined for EEG diagnostic parity checks")
        # Match the historical torch random_split ordering and fixed seed 42.
        # This is deliberately diagnostic, has no test set, and must not train a
        # production experiment. The shared engine rejects diagnostic_only.
        import torch
        order = torch.randperm(len(bundle), generator=torch.Generator().manual_seed(42)).tolist()
        boundary = int(0.8 * len(bundle))
        partitions = {"train": order[:boundary], "validation": order[boundary:], "test": []}
    split = {"split_id": split_id, "fingerprint": bundle.fingerprint,
             "seed": int(seed), "protocol": protocol, "identity": identity,
             "diagnostic_only": protocol == "diagnostic_eeg",
             "subject_independent": protocol == "group", **partitions}
    split["sample_ids"] = {name: [bundle.sample_ids[i] for i in indices] for name, indices in partitions.items()}
    split["groups"] = {name: sorted(set(bundle.groups[indices].tolist())) for name, indices in partitions.items()}
    split["class_counts"] = {name: np.bincount(bundle.y[indices], minlength=bundle.metadata["num_classes"]).tolist()
                             for name, indices in partitions.items()}
    split["partition_digest"] = hashlib.sha256(canonical_json(partitions).encode()).hexdigest()
    validate_split(bundle, split)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False) as stream:
        json.dump(split, stream, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = stream.name
    os.replace(temporary, path)
    return split


def prepare_data(full_config: dict, seed: int) -> tuple[SignalDataset, dict]:
    from .registry import load_dataset
    bundle = load_dataset(full_config["dataset"], full_config.get("data", {}))
    split = get_split(bundle, full_config.get("split", {}), seed,
                      full_config.get("split_dir", "results/splits"))
    return bundle, split
