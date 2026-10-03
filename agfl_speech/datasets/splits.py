"""Persist dataset/seed splits once and reuse them for every mechanism."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from .base import SignalDataset, canonical_json


SPLIT_VERSION = "agfl-speech-splits-v1"
# stratified: random within every subject and class (the default).
# chronological: per subject, earlier trials train and the latest trials test.
# group: whole subjects are held out (subject-independent).
# authors: the dataset authors' pooled random test part is the test partition.
PROTOCOLS = ("stratified", "chronological", "group", "authors")
PROTOCOL_ALIASES = {"subject": "group", "trial": "stratified", "subject_dependent": "stratified"}


def canonical_protocol(name):
    protocol = PROTOCOL_ALIASES.get(name, name)
    if protocol not in PROTOCOLS:
        raise ValueError("split.protocol must be stratified, chronological, group or authors")
    return protocol


SPLIT_KEYS = {"protocol", "train", "validation", "test", "train_fraction", "validation_fraction",
              "test_fraction", "validation_within_train"}


def split_settings(split_cfg):
    """Validated split settings holding only the keys the protocol uses.

    Used when a configuration is resolved (so a mistake fails at plan time) and
    again by ``get_split``; applying it twice changes nothing. Keys of another
    protocol, such as the default fractions for ``authors``, are dropped so they
    cannot enter the saved configuration or the split identity.
    """
    supplied = dict(split_cfg or {})
    if set(supplied) - SPLIT_KEYS:
        raise ValueError(f"Unknown split settings: {sorted(set(supplied) - SPLIT_KEYS)}")
    protocol = canonical_protocol(supplied.get("protocol", "stratified"))
    if protocol == "authors":
        fraction = supplied.get("validation_within_train", 0.2)
        if type(fraction) not in (int, float) or not 0 < fraction < 1:
            raise ValueError("validation_within_train must lie strictly between zero and one")
        return {"protocol": protocol, "validation_within_train": float(fraction)}
    return {"protocol": protocol, **_fractions(supplied)}


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


def _subject_class_cells(bundle, indices):
    """Trials of one subject and one class, in a fixed subject/class order."""
    for group in np.unique(bundle.groups[indices]):
        members = indices[bundle.groups[indices] == group]
        for label in np.unique(bundle.y[members]):
            yield group, label, members[bundle.y[members] == label]


def _stratified_split(bundle, fractions, rng):
    # Every subject contributes each of its classes to every partition in the
    # same proportions, so a pooled model is never tested on a subject or class
    # mixture that differs from its training data.
    partitions = {"train": [], "validation": [], "test": []}
    for group, label, cell in _subject_class_cells(bundle, np.arange(len(bundle))):
        members = rng.permutation(cell)
        try:
            train_size, validation_size, _ = _sizes(len(members), fractions)
        except ValueError as error:
            raise ValueError(f"Subject {group}, class {label} needs at least three trials "
                             "for a stratified three-way split") from error
        partitions["train"].extend(members[:train_size].tolist())
        partitions["validation"].extend(members[train_size:train_size + validation_size].tolist())
        partitions["test"].extend(members[train_size + validation_size:].tolist())
    return {name: sorted(values) for name, values in partitions.items()}


def _sample_order(bundle):
    order = np.asarray(bundle.metadata.get("sample_order", []))
    if order.shape != bundle.y.shape or not np.issubdtype(order.dtype, np.integer):
        raise ValueError("Chronological splitting requires one integer sample_order per trial")
    return order


def _chronological_split(bundle, fractions):
    # No randomness: within each subject the earliest trials train, the next
    # validate and the latest test. Classes are not balanced by construction;
    # validate_split still requires every class in the training partition.
    order = _sample_order(bundle)
    partitions = {"train": [], "validation": [], "test": []}
    for group in np.unique(bundle.groups):
        members = np.flatnonzero(bundle.groups == group)
        members = members[np.argsort(order[members], kind="stable")]
        try:
            train_size, validation_size, _ = _sizes(len(members), fractions)
        except ValueError as error:
            raise ValueError(f"Subject {group} needs at least three trials for a chronological split") from error
        partitions["train"].extend(members[:train_size].tolist())
        partitions["validation"].extend(members[train_size:train_size + validation_size].tolist())
        partitions["test"].extend(members[train_size + validation_size:].tolist())
    return {name: sorted(values) for name, values in partitions.items()}


def _author_test(bundle):
    flags = np.asarray(bundle.metadata.get("author_test", []))
    if flags.shape != bundle.y.shape or flags.dtype != np.bool_ or not flags.any() or flags.all():
        raise ValueError("The authors protocol requires the dataset authors' train/test membership of every trial")
    return flags


def _authors_split(bundle, config, rng):
    flags = _author_test(bundle)
    fraction = config["validation_within_train"]
    partitions = {"train": [], "validation": [], "test": np.flatnonzero(flags).tolist()}
    for group, label, cell in _subject_class_cells(bundle, np.flatnonzero(~flags)):
        members = rng.permutation(cell)
        if len(members) < 2:
            raise ValueError(f"Subject {group}, class {label} has too few author-training trials to create validation")
        n_validation = min(len(members) - 1, max(1, int(round(len(members) * fraction))))
        partitions["validation"].extend(members[:n_validation].tolist())
        partitions["train"].extend(members[n_validation:].tolist())
    return {name: sorted(values) for name, values in partitions.items()}


def validate_split(bundle: SignalDataset, split: dict) -> None:
    if split.get("fingerprint") != bundle.fingerprint:
        raise ValueError("Persisted split dataset fingerprint differs from the loaded dataset")
    parts = {name: split[name] for name in ("train", "validation", "test")}
    for name, indices in parts.items():
        if not indices or any(type(i) is not int for i in indices):
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
    if split["protocol"] == "chronological":
        order = _sample_order(bundle)
        for group in np.unique(bundle.groups):
            spans = [order[[i for i in parts[name] if bundle.groups[i] == group]]
                     for name in ("train", "validation", "test")]
            if any(len(span) == 0 for span in spans) or not (
                    spans[0].max() < spans[1].min() and spans[1].max() < spans[2].min()):
                raise ValueError("Chronological split must train on the earliest and test on the latest "
                                 "trials of every subject")
    if split["protocol"] == "authors":
        if set(np.flatnonzero(_author_test(bundle)).tolist()) != sets["test"]:
            raise ValueError("Authors split must test exactly on the dataset authors' test trials")
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
    config = split_settings(split_cfg)
    protocol = config["protocol"]
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
        partitions = _stratified_split(bundle, config, rng)
    elif protocol == "chronological":
        partitions = _chronological_split(bundle, config)
    else:
        partitions = _authors_split(bundle, config, rng)
    split = {"split_id": split_id, "fingerprint": bundle.fingerprint,
             "seed": int(seed), "protocol": protocol, "identity": identity,
             "subject_independent": protocol == "group", **partitions}
    split["sample_ids"] = {name: [bundle.sample_ids[i] for i in indices] for name, indices in partitions.items()}
    split["groups"] = {name: sorted(set(bundle.groups[indices].tolist())) for name, indices in partitions.items()}
    split["class_counts"] = {name: np.bincount(bundle.y[indices], minlength=bundle.metadata["num_classes"]).tolist()
                             for name, indices in partitions.items()}
    split["group_counts"] = {name: {str(group): int((bundle.groups[indices] == group).sum())
                                    for group in sorted(set(bundle.groups[indices].tolist()))}
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
