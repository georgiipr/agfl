"""A tiny SI_Hom-style archive for development tests; never a research dataset.

Plain module without a pytest dependency, so unittest files can use it too. The
archive follows the schema written by ``scripts/prepare_si_hom.py``: the same
arrays, electrode order, subject/block layout and manifest fields, with random
signals instead of recordings.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from agfl_speech.datasets import si_hom


def write_si_hom_fixture(directory, *, per_cell=4, samples=64, seed=0):
    """Write ``si_hom_trials.npz`` and its manifest into ``directory``.

    Every one of the 7 subjects has ``per_cell`` trials of each of the 8
    classes (``7 * 8 * per_cell`` trials). Subjects 5 and 6 are split over two
    recording blocks, as in the recordings. Exactly one trial of every
    subject/class cell is marked as an author test trial. Returns the folder as
    a string, ready for ``data.data_dir``.
    """
    if per_cell < 4:
        raise ValueError('per_cell must be at least 4: three-way splits need three trials per cell, '
                         'and the authors protocol removes one more')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings = {'per_cell': per_cell, 'samples': samples, 'seed': seed}
    manifest_path = directory / si_hom.MANIFEST_NAME
    if manifest_path.is_file() and (directory / si_hom.ARCHIVE_NAME).is_file():
        # Never rewrite an archive a test already loaded: its checksum is part
        # of the dataset fingerprint.
        if json.loads(manifest_path.read_text(encoding='utf-8')).get('fixture') == settings:
            return str(directory)
    rng = np.random.default_rng(seed)
    classes, channels = si_hom.NUM_CLASSES, si_hom.NUM_CHANNELS
    patterns = rng.standard_normal((classes, channels, samples)).astype(np.float32)
    signals, labels, subjects, blocks, author_test = [], [], [], [], []
    for subject in si_hom.SUBJECTS:
        subject_blocks = [index + 1 for index, owner in enumerate(si_hom.BLOCK_TO_SUBJECT) if owner == subject]
        # Classes appear in a shuffled (presentation-like) order within a subject.
        order = rng.permutation(np.repeat(np.arange(classes), per_cell))
        test_position = {label: int(rng.choice(np.flatnonzero(order == label))) for label in range(classes)}
        boundaries = np.linspace(0, len(order), len(subject_blocks) + 1).astype(int)
        for position, label in enumerate(order):
            trial = rng.standard_normal((channels, samples)).astype(np.float32) + 0.5 * patterns[label]
            trial = (trial - trial.mean(-1, keepdims=True)) / trial.std(-1, keepdims=True)
            signals.append(trial.astype(np.float32))
            labels.append(int(label))
            subjects.append(subject)
            blocks.append(subject_blocks[int(np.searchsorted(boundaries, position, side='right')) - 1])
            author_test.append(test_position[int(label)] == position)
    blocks = np.asarray(blocks, dtype=np.int64)
    order_in_block = np.concatenate([np.arange(int((blocks == block).sum()), dtype=np.int64)
                                     for block in np.unique(blocks)])
    arrays = {
        'x': np.stack(signals).astype(np.float32), 'y': np.asarray(labels, dtype=np.int64),
        'subjects': np.asarray(subjects, dtype=np.int64), 'blocks': blocks,
        'trial_numbers': np.arange(1, len(labels) + 1, dtype=np.int64), 'order_in_block': order_in_block,
        'channel_names': np.asarray(si_hom.CHANNEL_NAMES),
        'label_names': np.asarray(['P1a', 'P1b', 'P2a', 'P2b', 'P3a', 'P3b', 'P4a', 'P4b']),
        'event_codes': np.asarray([21, 22, 31, 32, 41, 42, 51, 52], dtype=np.int64),
        'sampling_rate': np.asarray(si_hom.SAMPLING_RATE, dtype=np.float64),
        'format_version': np.asarray(si_hom.FORMAT_VERSION),
        'author_test': np.asarray(author_test, dtype=np.bool_),
    }
    assert set(arrays) == set(si_hom.REQUIRED_ARRAYS)
    archive = directory / si_hom.ARCHIVE_NAME
    np.savez(archive, **arrays)
    manifest = {
        'format_version': si_hom.FORMAT_VERSION, 'fixture': settings,
        'archive': {'name': archive.name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
                    'bytes': archive.stat().st_size},
        'trials': int(arrays['x'].shape[0]), 'channels': channels, 'samples': samples,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return str(directory)


def tiny_config(tmp_path, **changes):
    """A two-epoch CPU run on the fixture archive: signal_transformer with MHA."""
    tmp_path = Path(tmp_path)
    config = {
        'dataset': 'si_hom', 'model': 'signal_transformer', 'attention': 'mha',
        'attention_options': {'heads': 2}, 'device': 'cpu', 'seeds': [3],
        'output_dir': str(tmp_path / 'results'), 'split_dir': str(tmp_path / 'splits'),
        'data': {'data_dir': write_si_hom_fixture(tmp_path / 'data')},
        'model_options': {'dim': 8, 'depth': 1, 'eeg_kernel_size': 3, 'eeg_temporal_bins': 2},
        'training': {'epochs': 2, 'batch_size': 16},
    }
    config.update(changes)
    return config
