"""Build the SI_Hom imagined-speech trial archive from the authors' MATLAB files.

Standalone by design: it needs numpy, h5py and scipy, and it never imports the
training package. Run it once on
the machine that holds the original ``Data_voices`` material; training needs
only the resulting ``si_hom_trials.npz`` and ``si_hom_manifest.json``.

    python scripts/prepare_si_hom.py --source /path/to/Data_voices

Source files (paths relative to ``--source``):

* ``数据集文件/第四章预处理后数据/oneTrailMat1s/<i>.mat`` for i = 1..2640: MATLAB
  v7.3 files, variable ``singleMatrix`` = one trial, [19 electrodes, 500 samples].
* ``第四章/label_merged_two_splited_subject_and_recorrect_theLabel.mat``:
  ``label_merged`` [2640, 2] = (subject 1..7, class 0..7) where row i belongs to
  ``<i>.mat``, and ``subjectTrails`` = trials in each of the 9 recording blocks.
* ``数据集文件/第四章预处理后数据/dataset.mat``: the authors' pooled random 80/20 split
  of the same trials with their class labels. It cross-checks the labels and
  supplies the ``author_test`` flag that the loader requires.

``label_Cell.mat`` in the same folder is deliberately NOT used: it holds the
same counts in a different (word-sorted) trial order and does not align with the
single-trial files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

TRIALS_DIR = Path("数据集文件/第四章预处理后数据/oneTrailMat1s")
LABELS_FILE = Path("第四章/label_merged_two_splited_subject_and_recorrect_theLabel.mat")
AUTHOR_SPLIT_FILE = Path("数据集文件/第四章预处理后数据/dataset.mat")

NUM_TRIALS, NUM_CHANNELS, NUM_SAMPLES, NUM_CLASSES = 2640, 19, 500, 8
SAMPLING_RATE = 500.0
# Electrode order of the recording montage (NEDF header of the raw files and the
# authors' BrainNetViewer node file agree; EEGLAB only removed the EXT channel).
CHANNEL_NAMES = ["P7", "P4", "Cz", "Pz", "P3", "P8", "O1", "O2", "T8", "F8",
                 "C4", "F4", "Fp2", "Fz", "C3", "F3", "Fp1", "T7", "F7"]
# The authors relabel these stimulus event codes to classes 0..7 (reLabel_1.m).
EVENT_CODES = [21, 22, 31, 32, 41, 42, 51, 52]
# Four homophone pairs; a/b are the two written words that share one sound.
LABEL_NAMES = ["P1a", "P1b", "P2a", "P2b", "P3a", "P3b", "P4a", "P4b"]
# Word order of the authors' semantic-distance script (word2vec/Chinese2vec.py).
# The correspondence to the event codes is inferred from that order only.
WORDS_INFERRED = ["助手", "住手", "报复", "暴富", "制服", "制伏", "初中", "初衷"]
# Recording blocks merged into subjects by the authors (subject_label_merge.m):
# blocks 5+6 are two files of one participant, as are blocks 7+8.
BLOCK_TO_SUBJECT = [1, 2, 3, 4, 5, 5, 6, 6, 7]
ARCHIVE_NAME, MANIFEST_NAME = "si_hom_trials.npz", "si_hom_manifest.json"
FORMAT_VERSION = "si-hom-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matlab_array(dataset) -> np.ndarray:
    """h5py exposes MATLAB v7.3 arrays with reversed axes; restore MATLAB order."""
    return np.array(dataset).T


def read_labels(path: Path):
    import h5py
    with h5py.File(path, "r") as archive:
        merged = matlab_array(archive["label_merged"])
        block_sizes = matlab_array(archive["subjectTrails"]).reshape(-1)
    if merged.shape == (2, NUM_TRIALS):
        # The authors saved this variable as 2 x trials (row 1 subject, row 2 class).
        merged = merged.T
    if merged.shape != (NUM_TRIALS, 2):
        raise ValueError(f"label_merged must be [{NUM_TRIALS}, 2], found {merged.shape}")
    if not np.equal(merged, np.floor(merged)).all() or not np.equal(block_sizes, np.floor(block_sizes)).all():
        raise ValueError("Subject ids, class labels and block sizes must be integers")
    subjects, labels = merged[:, 0].astype(np.int64), merged[:, 1].astype(np.int64)
    block_sizes = block_sizes.astype(np.int64)
    if block_sizes.shape != (len(BLOCK_TO_SUBJECT),) or block_sizes.sum() != NUM_TRIALS:
        raise ValueError(f"subjectTrails must list {len(BLOCK_TO_SUBJECT)} blocks summing to {NUM_TRIALS}")
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError("Class labels must lie in 0..7")
    blocks = np.repeat(np.arange(1, len(block_sizes) + 1), block_sizes)
    expected_subjects = np.asarray(BLOCK_TO_SUBJECT)[blocks - 1]
    if not np.array_equal(subjects, expected_subjects):
        raise ValueError("Subject column does not match the recording blocks merged as 5+6 and 7+8")
    return subjects, labels, blocks, block_sizes


def read_trials(directory: Path):
    import h5py
    signals = np.empty((NUM_TRIALS, NUM_CHANNELS, NUM_SAMPLES), dtype=np.float64)
    digest = hashlib.sha256()
    for index in range(NUM_TRIALS):
        path = directory / f"{index + 1}.mat"
        if not path.is_file():
            raise FileNotFoundError(f"Missing single-trial file: {path}")
        file_hash = sha256_file(path)
        digest.update(f"{index + 1}.mat:{file_hash}\n".encode())
        with h5py.File(path, "r") as archive:
            trial = matlab_array(archive["singleMatrix"])
        if trial.shape != (NUM_CHANNELS, NUM_SAMPLES):
            raise ValueError(f"{path.name}: expected [{NUM_CHANNELS}, {NUM_SAMPLES}], found {trial.shape}")
        signals[index] = trial
    extra = sorted(p.name for p in directory.glob("*.mat")
                   if not (p.stem.isdigit() and 1 <= int(p.stem) <= NUM_TRIALS))
    if extra:
        raise ValueError(f"Unexpected files in {directory}: {extra[:5]}")
    if not np.isfinite(signals).all():
        raise ValueError("Source trials contain non-finite values")
    return signals, digest.hexdigest()


def check_author_split(path: Path, signals: np.ndarray, labels: np.ndarray) -> dict:
    """Every trial must occur exactly once in the authors' split with our label."""
    try:
        from scipy.io import loadmat
    except ImportError:
        return {"checked": False, "reason": "scipy is not installed"}
    if not path.is_file():
        return {"checked": False, "reason": f"{path.name} not found"}
    saved = loadmat(path)
    parts = {name: (np.asarray(saved[f"data_{name}"])[:, 0], np.asarray(saved[f"label_{name}"]).reshape(-1).astype(np.int64))
             for name in ("train", "test")}
    pooled = np.concatenate([parts["train"][0], parts["test"][0]])
    pooled_labels = np.concatenate([parts["train"][1], parts["test"][1]])
    if pooled.shape != signals.shape:
        raise ValueError(f"{path.name} holds {pooled.shape}, expected {signals.shape}")
    rows = {row.tobytes(): position for position, row in enumerate(pooled)}
    if len(rows) != len(pooled):
        raise ValueError(f"{path.name} contains duplicated trials")
    positions = np.asarray([rows.get(trial.tobytes(), -1) for trial in signals])
    if (positions < 0).any() or len(set(positions.tolist())) != NUM_TRIALS:
        raise ValueError(f"{path.name} is not a permutation of the single-trial files")
    if not np.array_equal(pooled_labels[positions], labels):
        raise ValueError(f"{path.name} class labels disagree with label_merged")
    in_test = positions >= len(parts["train"][0])
    return {"checked": True, "file": source_entry(path), "identical_trials": NUM_TRIALS,
            "label_agreement": 1.0, "author_train_trials": int((~in_test).sum()),
            "author_test_trials": int(in_test.sum()),
            "note": "The authors' pooled random 80/20 split has no validation part and is not used for training here."}, in_test


def source_entry(path: Path) -> dict:
    return {"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build(source: Path, output: Path, created: str | None) -> dict:
    subjects, labels, blocks, block_sizes = read_labels(source / LABELS_FILE)
    signals, trials_digest = read_trials(source / TRIALS_DIR)
    checked = check_author_split(source / AUTHOR_SPLIT_FILE, signals, labels)
    if not isinstance(checked, tuple):
        raise RuntimeError("The archive must record the authors' test membership, which needs scipy and "
                           f"{AUTHOR_SPLIT_FILE.name}: {checked['reason']}. Nothing was written.")
    author_split, author_test = checked
    stored = signals.astype(np.float32)
    max_rounding = float(np.abs(stored.astype(np.float64) - signals).max())
    trial_numbers = np.arange(1, NUM_TRIALS + 1, dtype=np.int64)
    # Position of each trial inside its recording block; the single-trial files
    # keep the order of the epoched recordings (see the README caveat).
    order_in_block = np.concatenate([np.arange(size, dtype=np.int64) for size in block_sizes])
    arrays = {
        "x": stored, "y": labels, "subjects": subjects, "blocks": blocks,
        "trial_numbers": trial_numbers, "order_in_block": order_in_block,
        "channel_names": np.asarray(CHANNEL_NAMES), "label_names": np.asarray(LABEL_NAMES),
        "event_codes": np.asarray(EVENT_CODES, dtype=np.int64),
        "sampling_rate": np.asarray(SAMPLING_RATE, dtype=np.float64),
        "format_version": np.asarray(FORMAT_VERSION),
    }
    arrays["author_test"] = author_test
    output.mkdir(parents=True, exist_ok=True)
    archive = output / ARCHIVE_NAME
    np.savez(archive, **arrays)
    manifest = {
        "format_version": FORMAT_VERSION, "created": created,
        "archive": {"name": ARCHIVE_NAME, "sha256": sha256_file(archive), "bytes": archive.stat().st_size,
                    "arrays": {name: {"shape": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
                               for name, value in arrays.items()}},
        "trials": NUM_TRIALS, "channels": NUM_CHANNELS, "samples": NUM_SAMPLES,
        "sampling_rate": SAMPLING_RATE, "num_classes": NUM_CLASSES,
        "channel_names": CHANNEL_NAMES, "label_names": LABEL_NAMES, "event_codes": EVENT_CODES,
        "words_inferred": WORDS_INFERRED,
        "words_note": "Word order taken from the authors' semantic-distance script; its match to the event codes is inferred, not confirmed.",
        "subjects": {f"S{subject:02d}": int((subjects == subject).sum()) for subject in np.unique(subjects)},
        "blocks": {f"B{block:02d}": {"subject": f"S{BLOCK_TO_SUBJECT[block - 1]:02d}", "trials": int(size)}
                   for block, size in enumerate(block_sizes, start=1)},
        "class_counts": np.bincount(labels, minlength=NUM_CLASSES).tolist(),
        "subject_class_counts": {f"S{subject:02d}": np.bincount(labels[subjects == subject], minlength=NUM_CLASSES).tolist()
                                 for subject in np.unique(subjects)},
        "storage": {"dtype": "float32", "source_dtype": "float64", "max_abs_rounding_error": max_rounding},
        "preprocessing_by_authors": [
            "EEGLAB: unused channels removed, 49-51 Hz notch and band-pass filtering, epoching to 2 s, manual rejection of bad epochs",
            "z-score of every channel within every 2 s trial (Normalization.m)",
            "first 500 samples (1 s) kept (split2SingleTrailMat.m)",
        ],
        "sources": {"trials_directory": str(TRIALS_DIR), "trials_digest_sha256": trials_digest,
                    "trials_digest_rule": "sha256 over lines '<i>.mat:<sha256 of file>\\n' for i = 1..2640",
                    "labels": source_entry(source / LABELS_FILE)},
        "author_split_check": author_split,
        "script": {"name": Path(__file__).name, "sha256": sha256_file(Path(__file__))},
    }
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, help="Folder holding the authors' material (Data_voices)")
    parser.add_argument("--output", default=str(project_root.parent / "AGFL_speech_data"),
                        help="Data folder; default: AGFL_speech_data next to the project folder")
    parser.add_argument("--created", help="Optional creation date recorded in the manifest")
    args = parser.parse_args(argv)
    manifest = build(Path(args.source).expanduser().resolve(), Path(args.output).expanduser().resolve(), args.created)
    print(json.dumps({key: manifest[key] for key in ("archive", "subjects", "class_counts", "storage", "author_split_check")},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
