"""SI_Hom imagined speech: the dataset authors' preprocessed single trials.

2640 trials, 19 electrodes, 500 samples (1 s at 500 Hz), 8 classes (four
homophone pairs), 7 subjects recorded in 9 blocks. The trials were filtered,
epoched, cleaned and z-scored per trial and channel by the authors; this loader
only selects subjects, classes and a time window. It reads the single archive
written by ``scripts/prepare_si_hom.py`` and never touches the MATLAB sources.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .base import SignalDataset, normalize_samples, source_fingerprint
from .registry import register_dataset


DATASET_KEY = "si_hom"
ARCHIVE_NAME, MANIFEST_NAME = "si_hom_trials.npz", "si_hom_manifest.json"
FORMAT_VERSION = "si-hom-v1"
# The data folder sits next to the project folder on every machine. A relative
# data_dir is resolved against the project folder, never the working directory,
# so saved configurations and their identities do not depend on the location.
DEFAULT_DATA_DIR = "../AGFL_speech_data"
NUM_CHANNELS, NUM_CLASSES = 19, 8
SAMPLING_RATE = 500.0
SUBJECTS = tuple(range(1, 8))
BLOCK_TO_SUBJECT = (1, 2, 3, 4, 5, 5, 6, 6, 7)
CHANNEL_NAMES = ["P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8",
                 "C4", "F4", "Fp2", "Fz", "C3", "F3", "Fp1", "T7", "F7"]
COHORTS = ("pooled", "individual")
REQUIRED_ARRAYS = ("x", "y", "subjects", "blocks", "trial_numbers", "order_in_block",
                   "channel_names", "label_names", "event_codes", "sampling_rate",
                   "format_version", "author_test")
# The released recordings. The loader accepts any archive that is consistent
# with its own manifest (development tests use small ones); `check` compares the
# loaded data with these numbers.
EXPECTED_RELEASE = {
    "trials": 2640, "samples": 500, "author_test_trials": 528,
    "subject_trials": {"S01": 431, "S02": 218, "S03": 446, "S04": 381, "S05": 253, "S06": 506, "S07": 405},
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_data_dir(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else project_root() / path).resolve()


def subject_name(subject: int) -> str:
    return f"S{subject:02d}"


def _unique_integers(values, allowed, name):
    if (not isinstance(values, list) or not values or any(type(v) is not int for v in values)
            or len(set(values)) != len(values) or not set(values) <= set(allowed)):
        raise ValueError(f"{name} must be a nonempty list of unique integers in "
                         f"{min(allowed)}..{max(allowed)}")
    return sorted(values)


def validate_selection(data):
    """Check the data settings without opening any file (used by plan)."""
    subjects = _unique_integers(data["subjects"], SUBJECTS, "SI_Hom subjects")
    classes = _unique_integers(data["classes"], range(NUM_CLASSES), "SI_Hom classes")
    if len(classes) < 2:
        raise ValueError("SI_Hom classes must select at least two classes")
    if data["cohort"] not in COHORTS:
        raise ValueError("SI_Hom cohort must be pooled or individual")
    start, window = data["start"], data["window"]
    if type(start) is not int or start < 0 or (window is not None and (type(window) is not int or window < 1)):
        raise ValueError("SI_Hom start must be a nonnegative sample index and window a positive "
                         "number of samples (or null for the rest of the stored trial)")
    if data["normalization"] not in ("train_channel", "per_sample", "none"):
        raise ValueError("SI_Hom normalization must be train_channel, per_sample or none")
    if not isinstance(data["data_dir"], str) or not data["data_dir"]:
        raise ValueError("SI_Hom data_dir must be a folder path")
    return subjects, classes


def expand_cohort(config):
    """Pooled: one model for all selected subjects. Individual: one per subject."""
    data = config["data"]
    subjects, _ = validate_selection(data)
    protocol = config["split"]["protocol"]
    if data["cohort"] == "pooled":
        if config.get("subject_id") is not None:
            raise ValueError("A pooled SI_Hom cohort has no subject_id")
        if protocol == "group" and len(subjects) < 3:
            raise ValueError("A subject-independent split needs at least three subjects")
        return [config]
    if protocol == "group":
        raise ValueError("Individual SI_Hom subjects cannot use the subject-independent group split; "
                         "use stratified, chronological or authors")
    experiments = []
    for subject in subjects:
        experiment = deepcopy(config)
        subject_id = subject_name(subject)
        if config.get("subject_id") not in (None, subject_id):
            raise ValueError("Saved subject_id does not match data.subjects")
        experiment["data"]["subjects"] = [subject]
        experiment["subject_id"] = subject_id
        experiments.append(experiment)
    return experiments


def read_archive(root: Path):
    """Return the stored arrays after checking them against the shipped manifest."""
    archive_path, manifest_path = root / ARCHIVE_NAME, root / MANIFEST_NAME
    for path in (archive_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required SI_Hom file not found: {path}. Place the data folder next to the project "
                f"folder (default {DEFAULT_DATA_DIR}) or set data.data_dir explicitly.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = source_fingerprint(archive_path)
    recorded = manifest.get("archive", {})
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{manifest_path.name} is not a {FORMAT_VERSION} manifest")
    if recorded.get("sha256") != source["sha256"] or recorded.get("bytes") != source["bytes"]:
        raise ValueError(f"{archive_path.name} does not match its manifest; the transfer is incomplete "
                         "or the archive was rebuilt without its manifest")
    with np.load(archive_path, allow_pickle=False) as archive:
        missing = set(REQUIRED_ARRAYS) - set(archive.files)
        if missing:
            raise ValueError(f"{archive_path.name} lacks arrays: {sorted(missing)}")
        arrays = {name: archive[name] for name in REQUIRED_ARRAYS}
    if str(arrays["format_version"]) != FORMAT_VERSION:
        raise ValueError(f"{archive_path.name} is not a {FORMAT_VERSION} archive")
    x, y = arrays["x"], arrays["y"]
    if x.ndim != 3 or x.shape[1] != NUM_CHANNELS or x.dtype != np.float32 or min(x.shape) < 1:
        raise ValueError(f"SI_Hom signals must be float32 [trials, {NUM_CHANNELS}, samples]")
    trials = x.shape[0]
    for name in ("y", "subjects", "blocks", "trial_numbers", "order_in_block", "author_test"):
        if arrays[name].shape != (trials,):
            raise ValueError(f"SI_Hom array {name} must hold one value per trial")
    for name in ("y", "subjects", "blocks", "trial_numbers", "order_in_block"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            raise ValueError(f"SI_Hom array {name} must be integer")
    if arrays["author_test"].dtype != np.bool_:
        raise ValueError("SI_Hom author_test must be boolean")
    if (manifest.get("trials"), manifest.get("channels"), manifest.get("samples")) != tuple(int(v) for v in x.shape):
        raise ValueError(f"{archive_path.name} signal shape differs from its manifest")
    if arrays["channel_names"].astype(str).tolist() != CHANNEL_NAMES:
        raise ValueError("SI_Hom electrode order differs from the recording montage")
    if float(arrays["sampling_rate"]) != SAMPLING_RATE:
        raise ValueError("SI_Hom sampling rate must be 500 Hz")
    if (y.min() < 0 or y.max() >= NUM_CLASSES or arrays["label_names"].shape != (NUM_CLASSES,)
            or arrays["event_codes"].shape != (NUM_CLASSES,)):
        raise ValueError("SI_Hom labels must lie in 0..7 with eight label names and event codes")
    blocks, subjects = arrays["blocks"], arrays["subjects"]
    if (blocks.min() < 1 or blocks.max() > len(BLOCK_TO_SUBJECT)
            or not np.array_equal(np.asarray(BLOCK_TO_SUBJECT)[blocks - 1], subjects)):
        raise ValueError("SI_Hom recording blocks do not match their subjects")
    if trials > 1 and not (np.diff(arrays["trial_numbers"]) > 0).all():
        raise ValueError("SI_Hom trials must be stored in their source order")
    # Only the archive identifies the data; the manifest also holds notes such
    # as a creation date and must not change the dataset fingerprint.
    return arrays, [source]


@register_dataset(DATASET_KEY, "eeg", {
    "data_dir": DEFAULT_DATA_DIR, "subjects": list(SUBJECTS), "classes": list(range(NUM_CLASSES)),
    # pooled: one model on all selected subjects (the dataset authors' setting);
    # individual: one experiment per subject.
    "cohort": "pooled",
    # First sample and number of samples taken from each stored trial; null
    # keeps the rest of the trial. The stored trial is the first second (500
    # samples) of the authors' 2 s epoch.
    "start": 0, "window": None,
    # Trials are already z-scored per trial and channel over the 2 s epoch;
    # train_channel additionally standardizes each electrode with training
    # statistics only (fitted by the shared engine).
    "normalization": "train_channel",
}, "Eight-class imagined speech (SI_Hom), 19 electrodes, 1 s at 500 Hz, 7 subjects",
    experiment_defaults={"split": {"protocol": "stratified"}}, expand_experiments=expand_cohort)
def load_si_hom(config):
    subjects, classes = validate_selection(config)
    arrays, sources = read_archive(resolve_data_dir(config["data_dir"]))
    selected = np.flatnonzero(np.isin(arrays["subjects"], subjects) & np.isin(arrays["y"], classes))
    if not len(selected):
        raise ValueError("No SI_Hom trials match the selected subjects and classes")
    stored = arrays["x"].shape[-1]
    start = config["start"]
    stop = stored if config["window"] is None else start + config["window"]
    if start >= stop or stop > stored:
        raise ValueError(f"SI_Hom start/window must select samples inside the {stored} stored samples")
    x = normalize_samples(arrays["x"][selected, :, start:stop], config["normalization"])
    # Selected classes become consecutive indices in their original order.
    labels = np.searchsorted(np.asarray(classes), arrays["y"][selected])
    subject_ids = arrays["subjects"][selected]
    block_ids = arrays["blocks"][selected]
    trial_numbers = arrays["trial_numbers"][selected]
    groups = np.asarray([subject_name(int(s)) for s in subject_ids])
    ids = [f"{subject_name(int(s))}:B{int(b):02d}:trial:{int(t):04d}"
           for s, b, t in zip(subject_ids, block_ids, trial_numbers)]
    label_names = arrays["label_names"].astype(str).tolist()
    event_codes = arrays["event_codes"].astype(int).tolist()
    return SignalDataset(x, labels, groups, ids, {
        "dataset": DATASET_KEY, "modality": "eeg", "num_classes": len(classes),
        "sampling_rate": SAMPLING_RATE, "channel_names": list(CHANNEL_NAMES),
        "label_names": [label_names[c] for c in classes],
        "event_codes": [event_codes[c] for c in classes], "class_indices": list(classes),
        # What was selected, in one spelling: the same trials and samples give
        # the same fingerprint whether window was null or 500, or subjects were
        # listed in another order. The cohort only decides how experiments are
        # launched and is left out, so one subject's data, fingerprint and split
        # are identical in both cohorts.
        "preprocessing": {"data_dir": config["data_dir"], "subjects": list(subjects), "classes": list(classes),
                          "start": start, "window": stop - start, "normalization": config["normalization"]},
        "sources": sources,
        "subject_trials": {subject_name(s): int((subject_ids == s).sum()) for s in subjects},
        # Source order is the order of the epoched recordings: trials of one
        # subject follow their recording blocks and positions inside a block.
        "sample_blocks": [f"B{int(b):02d}" for b in block_ids],
        "sample_order": [int(t) for t in trial_numbers],
        # Membership in the authors' pooled random test part (see docs/dataset.md).
        "author_test": [bool(v) for v in arrays["author_test"][selected]],
        "protocol_version": FORMAT_VERSION, "preprocessed_by_authors": True,
    })
