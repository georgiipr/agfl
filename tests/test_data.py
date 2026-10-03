"""SI_Hom loader, split and cohort invariants on a tiny schema-correct archive.

Development checks for the experiment machine only; they are not a prerequisite
for training. The archive is written by ``tests.si_hom_fixture`` (7 subjects x
8 classes x 4 trials, 19 electrodes, 64 samples), so no recording is needed.
One test reads the released archive and runs only with
``AGFL_SPEECH_REAL_DATA=1``.

Run from the project folder: ``python -m unittest tests.test_data``.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from agfl_speech.config import resolve_experiments
from agfl_speech.datasets import (SignalDataset, get_dataset_spec, get_split, list_datasets,
                                  load_dataset, prepare_data, validate_split)
from agfl_speech.datasets import si_hom
from agfl_speech.datasets.base import normalize_samples
from tests.si_hom_fixture import write_si_hom_fixture


PER_CELL, SAMPLES = 4, 64
SUBJECT_IDS = [f'S{subject:02d}' for subject in range(1, 8)]
CHANNEL_NAMES = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'T8', 'F8',
                 'C4', 'F4', 'Fp2', 'Fz', 'C3', 'F3', 'Fp1', 'T7', 'F7']
LABEL_NAMES = ['P1a', 'P1b', 'P2a', 'P2b', 'P3a', 'P3b', 'P4a', 'P4b']
TRIALS = 7 * 8 * PER_CELL
PARTITIONS = ('train', 'validation', 'test')


class FixtureCase(unittest.TestCase):
    """One read-only archive per class; every test gets its own scratch folder."""

    @classmethod
    def setUpClass(cls):
        cls._archive = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._archive.cleanup)
        cls.data_dir = write_si_hom_fixture(Path(cls._archive.name) / 'data',
                                            per_cell=PER_CELL, samples=SAMPLES)

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.splits = self.root / 'splits'

    def load(self, **settings):
        return load_dataset('si_hom', {'data_dir': self.data_dir, **settings})

    def stored(self):
        with np.load(Path(self.data_dir) / si_hom.ARCHIVE_NAME, allow_pickle=False) as archive:
            return {name: archive[name] for name in archive.files}

    def copy_files(self, name, files=(si_hom.ARCHIVE_NAME, si_hom.MANIFEST_NAME)):
        """Copy the stored files byte for byte; the archive is not regenerated."""
        target = self.root / name
        target.mkdir()
        for file in files:
            shutil.copyfile(Path(self.data_dir) / file, target / file)
        return target

    def rewrite_archive(self, name, change):
        """Store changed arrays in a new folder together with a matching manifest."""
        target = self.root / name
        target.mkdir()
        arrays = self.stored()
        change(arrays)
        archive = target / si_hom.ARCHIVE_NAME
        np.savez(archive, **arrays)
        manifest = json.loads((Path(self.data_dir) / si_hom.MANIFEST_NAME).read_text(encoding='utf-8'))
        manifest['archive'] = {'name': archive.name, 'bytes': archive.stat().st_size,
                               'sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}
        (target / si_hom.MANIFEST_NAME).write_text(json.dumps(manifest), encoding='utf-8')
        return str(target)


class LoaderTests(FixtureCase):
    def test_registry_holds_only_si_hom_with_its_documented_defaults(self):
        self.assertEqual(list_datasets(), ['si_hom'])
        spec = get_dataset_spec('si_hom')
        self.assertEqual(spec.modality, 'eeg')
        self.assertEqual(spec.defaults, {
            'data_dir': '../AGFL_speech_data', 'subjects': [1, 2, 3, 4, 5, 6, 7],
            'classes': [0, 1, 2, 3, 4, 5, 6, 7], 'cohort': 'pooled',
            'start': 0, 'window': None, 'normalization': 'train_channel'})
        self.assertEqual(spec.experiment_defaults, {'split': {'protocol': 'stratified'}})
        with self.assertRaisesRegex(ValueError, 'Unknown dataset'):
            load_dataset('eeg')
        with self.assertRaisesRegex(ValueError, 'Unknown si_hom data settings'):
            load_dataset('si_hom', {'data_dir': self.data_dir, 'typo': 1})

    def test_arrays_and_metadata_follow_the_dataset_contract(self):
        bundle = self.load()
        self.assertEqual(len(bundle), TRIALS)
        self.assertEqual(bundle.x.shape, (TRIALS, 19, SAMPLES))
        self.assertEqual(bundle.x.dtype, np.float32)
        self.assertEqual(bundle.y.shape, (TRIALS,))
        self.assertTrue(np.issubdtype(bundle.y.dtype, np.integer))
        self.assertEqual(sorted(set(bundle.y.tolist())), list(range(8)))
        self.assertEqual(sorted(set(bundle.groups.tolist())), SUBJECT_IDS)
        self.assertEqual(len(bundle.sample_ids), TRIALS)
        self.assertEqual(len(set(bundle.sample_ids)), TRIALS)
        metadata = bundle.metadata
        self.assertEqual(metadata['modality'], 'eeg')
        self.assertEqual(metadata['channels'], 19)
        self.assertEqual(metadata['samples'], SAMPLES)
        self.assertEqual(metadata['num_classes'], 8)
        self.assertEqual(metadata['sampling_rate'], 500.0)
        self.assertEqual(metadata['channel_names'], CHANNEL_NAMES)
        self.assertEqual(metadata['label_names'], LABEL_NAMES)
        self.assertEqual(metadata['class_counts'], [7 * PER_CELL] * 8)
        self.assertEqual(metadata['subject_trials'], {subject: 8 * PER_CELL for subject in SUBJECT_IDS})
        for key in ('sample_blocks', 'sample_order', 'author_test'):
            self.assertEqual(len(metadata[key]), TRIALS, key)
        self.assertEqual(sorted(set(metadata['sample_blocks'])), [f'B{block:02d}' for block in range(1, 10)])
        self.assertTrue(all(type(value) is int for value in metadata['sample_order']))
        self.assertTrue(all(type(value) is bool for value in metadata['author_test']))
        # The fixture marks one author test trial in every subject/class cell.
        self.assertEqual(sum(metadata['author_test']), 7 * 8)
        for index in (0, TRIALS // 2, TRIALS - 1):
            self.assertEqual(bundle.sample_ids[index],
                             f"{bundle.groups[index]}:{metadata['sample_blocks'][index]}"
                             f":trial:{metadata['sample_order'][index]:04d}")
        # What was selected, in one spelling; the cohort is not part of the data.
        self.assertEqual(metadata['preprocessing'], {
            'data_dir': self.data_dir, 'subjects': [1, 2, 3, 4, 5, 6, 7], 'classes': list(range(8)),
            'start': 0, 'window': SAMPLES, 'normalization': 'train_channel'})
        self.assertEqual([source['name'] for source in metadata['sources']], ['si_hom_trials.npz'])
        # Saved configurations embed the metadata, so it must be plain JSON.
        json.dumps(metadata, allow_nan=False)

    def test_default_normalization_returns_the_stored_trials_unchanged(self):
        bundle = self.load()
        stored = self.stored()
        self.assertEqual(bundle.metadata['preprocessing']['normalization'], 'train_channel')
        np.testing.assert_array_equal(bundle.x, stored['x'])
        np.testing.assert_array_equal(bundle.y, stored['y'])
        np.testing.assert_array_equal(np.asarray(bundle.metadata['author_test']), stored['author_test'])
        untouched = self.load(normalization='none')
        np.testing.assert_array_equal(untouched.x, stored['x'])
        self.assertNotEqual(untouched.fingerprint, bundle.fingerprint)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            self.load(normalization='zscore')

    def test_fingerprint_ignores_the_folder_and_the_cohort(self):
        first, second = self.load(), self.load()
        self.assertEqual(first.fingerprint, second.fingerprint)
        relocated = load_dataset('si_hom', {'data_dir': str(self.copy_files('relocated'))})
        self.assertEqual(relocated.fingerprint, first.fingerprint)
        pooled = self.load(subjects=[3], cohort='pooled')
        individual = self.load(subjects=[3], cohort='individual')
        self.assertEqual(pooled.fingerprint, individual.fingerprint)
        np.testing.assert_array_equal(pooled.x, individual.x)
        self.assertEqual(pooled.sample_ids, individual.sample_ids)
        self.assertNotEqual(pooled.fingerprint, first.fingerprint)
        # One subject therefore has one split, whichever cohort launched it.
        self.assertEqual(get_split(pooled, {}, 2, self.splits)['split_id'],
                         get_split(individual, {}, 2, self.splits)['split_id'])

    def test_fingerprint_ignores_the_spelling_of_a_selection_and_manifest_notes(self):
        full = self.load()
        # A null window and the explicit stored length select the same samples.
        self.assertEqual(self.load(window=SAMPLES).fingerprint, full.fingerprint)
        self.assertEqual(self.load(subjects=[7, 6, 5, 4, 3, 2, 1], classes=[7, 6, 5, 4, 3, 2, 1, 0]).fingerprint,
                         full.fingerprint)
        self.assertEqual(self.load(subjects=[6, 2], classes=[5, 1, 6]).fingerprint,
                         self.load(subjects=[2, 6], classes=[1, 5, 6]).fingerprint)
        # Only the archive identifies the data: a note added to the manifest
        # (the checksum of the archive stays valid) changes nothing.
        annotated = self.copy_files('annotated')
        manifest_path = annotated / si_hom.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['created'] = 'another day'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        self.assertEqual(load_dataset('si_hom', {'data_dir': str(annotated)}).fingerprint, full.fingerprint)

    def test_content_changes_the_fingerprint(self):
        bundle = self.load()
        modified = SignalDataset(bundle.x + 0.01, bundle.y, bundle.groups, bundle.sample_ids,
                                 deepcopy(bundle.metadata))
        self.assertNotEqual(bundle.fingerprint, modified.fingerprint)
        split = get_split(bundle, {}, 0, self.splits)
        with self.assertRaisesRegex(ValueError, 'fingerprint'):
            validate_split(modified, split)

    def test_subject_and_class_subsets_are_remapped_to_consecutive_labels(self):
        bundle = self.load(subjects=[6, 2], classes=[5, 1, 6])
        stored = self.stored()
        selected = np.isin(stored['subjects'], [2, 6]) & np.isin(stored['y'], [1, 5, 6])
        self.assertEqual(len(bundle), 2 * 3 * PER_CELL)
        np.testing.assert_array_equal(bundle.x, stored['x'][selected])
        remapped = {1: 0, 5: 1, 6: 2}
        self.assertEqual(bundle.y.tolist(), [remapped[int(label)] for label in stored['y'][selected]])
        self.assertEqual(bundle.groups.tolist(),
                         [f'S{int(subject):02d}' for subject in stored['subjects'][selected]])
        self.assertEqual(bundle.metadata['num_classes'], 3)
        self.assertEqual(bundle.metadata['label_names'], ['P1b', 'P3b', 'P4a'])
        self.assertEqual(bundle.metadata['class_indices'], [1, 5, 6])
        self.assertEqual(bundle.metadata['class_counts'], [2 * PER_CELL] * 3)
        self.assertEqual(bundle.metadata['subject_trials'], {'S02': 3 * PER_CELL, 'S06': 3 * PER_CELL})
        self.assertNotEqual(bundle.fingerprint, self.load().fingerprint)

    def test_invalid_selections_are_rejected(self):
        for settings in ({'subjects': [8]}, {'subjects': []}, {'subjects': [1, 1]}, {'subjects': 3},
                         {'subjects': ['S01']}, {'classes': [3]}, {'classes': [0, 8]},
                         {'classes': [0, 0, 1]}, {'cohort': 'group'}, {'data_dir': ''},
                         {'normalization': 'train_sample'}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.load(**settings)

    def test_start_and_window_crop_the_stored_trial(self):
        full = self.load()
        cropped = self.load(start=8, window=16)
        self.assertEqual(cropped.x.shape, (TRIALS, 19, 16))
        np.testing.assert_array_equal(cropped.x, full.x[:, :, 8:24])
        self.assertEqual(cropped.metadata['samples'], 16)
        self.assertNotEqual(cropped.fingerprint, full.fingerprint)
        tail = self.load(start=40)
        np.testing.assert_array_equal(tail.x, full.x[:, :, 40:])
        whole = self.load(window=SAMPLES)
        np.testing.assert_array_equal(whole.x, full.x)
        self.assertEqual(cropped.metadata['preprocessing']['start'], 8)
        self.assertEqual(cropped.metadata['preprocessing']['window'], 16)
        self.assertEqual(tail.metadata['preprocessing']['window'], SAMPLES - 40)
        for settings in ({'start': 60, 'window': 5}, {'start': SAMPLES}, {'window': SAMPLES + 1},
                         {'start': -1}, {'window': 0}, {'start': 1.5}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    self.load(**settings)

    def test_missing_files_are_reported_with_their_path(self):
        empty = self.root / 'empty'
        empty.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, 'si_hom_trials'):
            load_dataset('si_hom', {'data_dir': str(empty)})
        archive_only = self.copy_files('archive_only', files=(si_hom.ARCHIVE_NAME,))
        with self.assertRaisesRegex(FileNotFoundError, 'si_hom_manifest'):
            load_dataset('si_hom', {'data_dir': str(archive_only)})

    def test_archive_must_match_its_manifest(self):
        changed_checksum = self.copy_files('changed_checksum')
        manifest_path = changed_checksum / si_hom.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['archive']['sha256'] = '0' * 64
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'does not match its manifest'):
            load_dataset('si_hom', {'data_dir': str(changed_checksum)})
        incomplete = self.copy_files('incomplete')
        archive_path = incomplete / si_hom.ARCHIVE_NAME
        archive_path.write_bytes(archive_path.read_bytes()[:-1])
        with self.assertRaisesRegex(ValueError, 'does not match its manifest'):
            load_dataset('si_hom', {'data_dir': str(incomplete)})
        other_format = self.copy_files('other_format')
        manifest_path = other_format / si_hom.MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['format_version'] = 'si-hom-v0'
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'manifest'):
            load_dataset('si_hom', {'data_dir': str(other_format)})

    def test_archive_contents_are_validated(self):
        # Control: an unchanged rewrite loads, so the failures below come from
        # the changed arrays and not from the rewriting helper.
        same = load_dataset('si_hom', {'data_dir': self.rewrite_archive('same', lambda arrays: None)})
        np.testing.assert_array_equal(same.x, self.load().x)

        def reverse_electrodes(arrays):
            arrays['channel_names'] = arrays['channel_names'][::-1].copy()
        with self.assertRaisesRegex(ValueError, 'electrode order'):
            load_dataset('si_hom', {'data_dir': self.rewrite_archive('electrodes', reverse_electrodes)})

        def drop_author_test(arrays):
            del arrays['author_test']
        with self.assertRaisesRegex(ValueError, 'lacks arrays'):
            load_dataset('si_hom', {'data_dir': self.rewrite_archive('missing', drop_author_test)})

        def reverse_blocks(arrays):
            arrays['blocks'] = arrays['blocks'][::-1].copy()
        with self.assertRaisesRegex(ValueError, 'blocks do not match'):
            load_dataset('si_hom', {'data_dir': self.rewrite_archive('blocks', reverse_blocks)})

    def test_relative_data_folder_resolves_against_the_project_folder(self):
        project = Path(si_hom.__file__).resolve().parents[2]
        self.assertEqual(si_hom.project_root(), project)
        self.assertEqual(si_hom.DEFAULT_DATA_DIR, '../AGFL_speech_data')
        expected = (project.parent / 'AGFL_speech_data').resolve()
        self.addCleanup(os.chdir, os.getcwd())
        before = si_hom.resolve_data_dir(si_hom.DEFAULT_DATA_DIR)
        os.chdir(self.root)
        after = si_hom.resolve_data_dir(si_hom.DEFAULT_DATA_DIR)
        self.assertEqual(before, expected)
        self.assertEqual(after, expected)
        self.assertEqual(si_hom.resolve_data_dir('data/copy'), (project / 'data' / 'copy').resolve())
        self.assertEqual(si_hom.resolve_data_dir(self.data_dir), Path(self.data_dir).resolve())
        # The loader looks in the resolved folder, not under the working directory.
        missing = 'tests/no_such_si_hom_folder'
        with self.assertRaises(FileNotFoundError) as raised:
            load_dataset('si_hom', {'data_dir': missing})
        self.assertIn(str((project / missing).resolve()), str(raised.exception))


class SignalContractTests(unittest.TestCase):
    def test_fractional_labels_duplicate_ids_bad_signals_and_modality_are_rejected(self):
        metadata = {'modality': 'eeg', 'num_classes': 2}
        groups = np.array(['S01', 'S02'])
        for labels, ids, message in [([0.5, 1], ['a', 'b'], 'integer'), ([0, 1], ['a', 'a'], 'unique')]:
            with self.assertRaisesRegex(ValueError, message):
                SignalDataset(np.zeros((2, 1, 16)), np.asarray(labels), groups, ids, dict(metadata))
        bad = np.zeros((2, 1, 16))
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            SignalDataset(bad, np.array([0, 1]), groups, ['a', 'b'], dict(metadata))
        with self.assertRaisesRegex(ValueError, 'modality eeg'):
            SignalDataset(np.zeros((2, 1, 16)), np.array([0, 1]), groups, ['a', 'b'],
                          {'modality': 'ecg', 'num_classes': 2})

    def test_train_channel_normalization_is_deferred_to_the_engine(self):
        x = np.arange(32, dtype=np.float32).reshape(2, 1, 16)
        np.testing.assert_array_equal(normalize_samples(x, 'train_channel'), x)
        np.testing.assert_array_equal(normalize_samples(x, 'none'), x)
        normalized = normalize_samples(x, 'per_sample')
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_allclose(normalized.mean(axis=-1), 0, atol=1e-6)
        np.testing.assert_allclose(normalized.std(axis=-1), 1, atol=1e-5)
        with self.assertRaisesRegex(ValueError, 'normalization'):
            normalize_samples(x, 'train_sample')


class SplitTests(FixtureCase):
    def cells(self, bundle, indices):
        return Counter(zip(bundle.groups[indices].tolist(), bundle.y[indices].tolist()))

    def assert_partition(self, bundle, split):
        indices = [index for name in PARTITIONS for index in split[name]]
        self.assertEqual(sorted(indices), list(range(len(bundle))))

    def test_stratified_split_covers_every_subject_and_class_in_every_partition(self):
        bundle = self.load()
        split = get_split(bundle, {'protocol': 'stratified'}, 5, self.splits)
        self.assertEqual(split['protocol'], 'stratified')
        self.assertFalse(split['subject_independent'])
        self.assert_partition(bundle, split)
        every_cell = {(subject, label) for subject in SUBJECT_IDS for label in range(8)}
        # Four trials per cell and 0.6/0.2/0.2 give two, one and one.
        for name, expected in (('train', 2), ('validation', 1), ('test', 1)):
            counts = self.cells(bundle, split[name])
            self.assertEqual(set(counts), every_cell, name)
            self.assertEqual(set(counts.values()), {expected}, name)
            self.assertEqual(split['class_counts'][name], [7 * expected] * 8)
            self.assertEqual(split['group_counts'][name], {subject: 8 * expected for subject in SUBJECT_IDS})
            self.assertEqual(split['groups'][name], SUBJECT_IDS)
        other = get_split(bundle, {'protocol': 'stratified'}, 6, self.splits)
        self.assertNotEqual(other['split_id'], split['split_id'])
        self.assertNotEqual(other['train'], split['train'])

    def test_default_and_alias_protocols_are_the_stratified_split(self):
        bundle = self.load()
        split = get_split(bundle, {'protocol': 'stratified'}, 5, self.splits)
        for settings in ({}, None, {'protocol': 'trial'}, {'protocol': 'subject_dependent'},
                         {'train_fraction': 0.6, 'validation_fraction': 0.2, 'test_fraction': 0.2}):
            with self.subTest(settings=settings):
                self.assertEqual(get_split(bundle, settings, 5, self.splits)['split_id'], split['split_id'])
        self.assertEqual(len(list(self.splits.glob('*.json'))), 1)

    def test_group_split_holds_out_whole_subjects(self):
        bundle = self.load()
        split = get_split(bundle, {'protocol': 'group'}, 8, self.splits)
        self.assertTrue(split['subject_independent'])
        self.assert_partition(bundle, split)
        groups = split['groups']
        self.assertEqual([len(groups[name]) for name in PARTITIONS], [5, 1, 1])
        self.assertEqual(sorted(groups['train'] + groups['validation'] + groups['test']), SUBJECT_IDS)
        for name in PARTITIONS:
            self.assertEqual(sorted(set(bundle.groups[split[name]].tolist())), groups[name])
        self.assertEqual(get_split(bundle, {'protocol': 'subject'}, 8, self.splits)['split_id'], split['split_id'])
        leaking = deepcopy(split)
        leaking['train'].append(leaking['test'].pop())
        with self.assertRaisesRegex(ValueError, 'Subject groups overlap'):
            validate_split(bundle, leaking)

    def test_split_is_shared_by_models_and_persisted_once(self):
        config = {'dataset': 'si_hom', 'data': {'data_dir': self.data_dir},
                  'split': {'protocol': 'group'}, 'split_dir': str(self.splits)}
        bundle, first = prepare_data(dict(config, model='eegnet', attention='agfl'), 8)
        _, second = prepare_data(dict(config, model='signal_transformer', attention='hcann'), 8)
        self.assertEqual(first, second)
        self.assertEqual(len(list(self.splits.glob('*.json'))), 1)
        third = get_split(bundle, {'protocol': 'group'}, 9, self.splits)
        self.assertNotEqual(first['split_id'], third['split_id'])
        self.assertEqual(len(list(self.splits.glob('*.json'))), 2)

    def test_chronological_split_follows_sample_order_and_ignores_the_seed(self):
        bundle = self.load()
        first = get_split(bundle, {'protocol': 'chronological'}, 0, self.splits)
        second = get_split(bundle, {'protocol': 'chronological'}, 1, self.splits)
        self.assertNotEqual(first['split_id'], second['split_id'])
        for name in PARTITIONS:
            self.assertEqual(first[name], second[name])
        self.assert_partition(bundle, first)
        validate_split(bundle, first)
        validate_split(bundle, second)
        order = np.asarray(bundle.metadata['sample_order'])
        for subject in SUBJECT_IDS:
            spans = [order[[index for index in first[name] if bundle.groups[index] == subject]]
                     for name in PARTITIONS]
            # 32 trials per subject and 0.6/0.2/0.2 give 20, 6 and 6.
            self.assertEqual([len(span) for span in spans], [20, 6, 6])
            self.assertLess(spans[0].max(), spans[1].min())
            self.assertLess(spans[1].max(), spans[2].min())
        # Both swapped trials belong to the first subject: its earliest trial
        # becomes a test trial and one of its latest trials a training trial.
        swapped = deepcopy(first)
        swapped['train'][0], swapped['test'][0] = first['test'][0], first['train'][0]
        with self.assertRaisesRegex(ValueError, 'Chronological'):
            validate_split(bundle, swapped)

    def test_authors_split_tests_exactly_on_the_author_test_trials(self):
        bundle = self.load()
        flags = np.asarray(bundle.metadata['author_test'])
        settings = {'protocol': 'authors', 'validation_within_train': 0.2}
        split = get_split(bundle, settings, 4, self.splits)
        self.assert_partition(bundle, split)
        self.assertEqual(split['test'], np.flatnonzero(flags).tolist())
        self.assertEqual(len(split['test']), 7 * 8)
        self.assertFalse(flags[split['train'] + split['validation']].any())
        # Three author-training trials per cell: one validates, two train.
        every_cell = {(subject, label) for subject in SUBJECT_IDS for label in range(8)}
        for name, expected in (('train', 2), ('validation', 1), ('test', 1)):
            counts = self.cells(bundle, split[name])
            self.assertEqual(set(counts), every_cell, name)
            self.assertEqual(set(counts.values()), {expected}, name)
        other = get_split(bundle, settings, 5, self.splits)
        self.assertEqual(other['test'], split['test'])
        self.assertNotEqual(other['split_id'], split['split_id'])
        # Fractions belong to the other protocols and are not part of this identity.
        self.assertEqual(split['identity']['config'], settings)
        for spelling in ({'protocol': 'authors'},
                         {'protocol': 'authors', 'train': .5, 'validation': .25, 'test': .25,
                          'validation_within_train': 0.2}):
            self.assertEqual(get_split(bundle, spelling, 4, self.splits)['split_id'], split['split_id'])
        moved = deepcopy(split)
        moved['train'].append(moved['test'].pop())
        with self.assertRaisesRegex(ValueError, 'Authors split'):
            validate_split(bundle, moved)
        with self.assertRaisesRegex(ValueError, 'validation_within_train'):
            get_split(bundle, {'protocol': 'authors', 'validation_within_train': 1.0}, 4, self.splits)

    def test_persisted_split_is_reused_and_tampering_is_detected(self):
        bundle = self.load()
        split = get_split(bundle, {}, 3, self.splits)
        path = self.splits / f"{split['split_id']}.json"
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text()), split)
        self.assertEqual(get_split(bundle, {}, 3, self.splits), split)
        self.assertEqual(len(list(self.splits.glob('*.json'))), 1)
        overlapping = deepcopy(split)
        overlapping['train'][0] = overlapping['test'][0]
        path.write_text(json.dumps(overlapping))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            get_split(bundle, {}, 3, self.splits)
        reseeded = deepcopy(split)
        reseeded['identity']['seed'] = 4
        path.write_text(json.dumps(reseeded))
        with self.assertRaisesRegex(ValueError, 'identity was modified'):
            get_split(bundle, {}, 3, self.splits)
        path.write_text(json.dumps(split))
        self.assertEqual(get_split(bundle, {}, 3, self.splits), split)

    def test_removed_protocols_unknown_settings_and_bad_fractions_are_rejected(self):
        bundle = self.load()
        for protocol in ('session', 'diagnostic_eeg', 'random'):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, 'split.protocol'):
                    get_split(bundle, {'protocol': protocol}, 0, self.splits)
        with self.assertRaisesRegex(ValueError, 'Unknown split settings'):
            get_split(bundle, {'folds': 5}, 0, self.splits)
        with self.assertRaisesRegex(ValueError, 'sum to 1'):
            get_split(bundle, {'train': 0.6, 'validation': 0.3, 'test': 0.3}, 0, self.splits)
        self.assertEqual(list(self.splits.glob('*.json')), [])

    def test_training_cannot_omit_a_declared_class(self):
        bundle = self.load()
        metadata = deepcopy(bundle.metadata)
        metadata['num_classes'] = 9
        declared = SignalDataset(bundle.x, bundle.y, bundle.groups, bundle.sample_ids, metadata)
        with self.assertRaisesRegex(ValueError, 'Training split lacks classes'):
            get_split(declared, {}, 1, self.splits)


class CohortTests(FixtureCase):
    def test_pooled_cohort_is_one_experiment_without_a_subject(self):
        experiments = resolve_experiments({'data': {'data_dir': self.data_dir}})
        self.assertEqual(len(experiments), 1)
        experiment = experiments[0]
        self.assertIsNone(experiment['subject_id'])
        self.assertEqual(experiment['dataset'], 'si_hom')
        self.assertEqual(experiment['data']['cohort'], 'pooled')
        self.assertEqual(experiment['data']['subjects'], [1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(experiment['data']['data_dir'], self.data_dir)
        self.assertEqual(experiment['split']['protocol'], 'stratified')
        self.assertEqual(resolve_experiments(experiment), [experiment])

    def test_individual_cohort_is_one_experiment_per_subject(self):
        original = {'data': {'data_dir': self.data_dir, 'cohort': 'individual', 'subjects': [3, 1]},
                    'seeds': [7]}
        experiments = resolve_experiments(original)
        self.assertEqual(original['data']['subjects'], [3, 1])
        self.assertEqual([experiment['subject_id'] for experiment in experiments], ['S01', 'S03'])
        self.assertEqual([experiment['data']['subjects'] for experiment in experiments], [[1], [3]])
        for experiment in experiments:
            self.assertEqual(experiment['seeds'], [7])
            self.assertEqual(experiment['data']['cohort'], 'individual')
            self.assertEqual(experiment['split']['protocol'], 'stratified')
            # A saved, already expanded configuration replays as itself.
            self.assertEqual(resolve_experiments(experiment), [experiment])
            bundle = load_dataset(experiment['dataset'], experiment['data'])
            self.assertEqual(set(bundle.groups.tolist()), {experiment['subject_id']})
            self.assertEqual(len(bundle), 8 * PER_CELL)
        everyone = resolve_experiments({'data': {'cohort': 'individual'}})
        self.assertEqual([experiment['subject_id'] for experiment in everyone], SUBJECT_IDS)

    def test_incompatible_cohort_and_split_settings_are_rejected_before_loading(self):
        # None of these configurations names an existing data folder.
        for protocol in ('group', 'subject'):
            with self.subTest(protocol=protocol):
                with self.assertRaisesRegex(ValueError, 'Individual SI_Hom subjects'):
                    resolve_experiments({'data': {'cohort': 'individual'}, 'split': {'protocol': protocol}})
        with self.assertRaisesRegex(ValueError, 'three subjects'):
            resolve_experiments({'data': {'subjects': [1, 2]}, 'split': {'protocol': 'group'}})
        self.assertEqual(len(resolve_experiments({'data': {'subjects': [1, 2, 3]},
                                                  'split': {'protocol': 'group'}})), 1)
        with self.assertRaisesRegex(ValueError, 'no subject_id'):
            resolve_experiments({'subject_id': 'S01'})
        with self.assertRaisesRegex(ValueError, 'subject_id'):
            resolve_experiments({'subject_id': 'S02', 'data': {'cohort': 'individual', 'subjects': [1]}})
        with self.assertRaisesRegex(ValueError, 'cohort'):
            resolve_experiments({'data': {'cohort': 'per_subject'}})
        with self.assertRaisesRegex(ValueError, 'Unknown dataset settings'):
            resolve_experiments({'data': {'sessions': ['T']}})


@unittest.skipUnless(os.environ.get('AGFL_SPEECH_REAL_DATA') == '1',
                     'Set AGFL_SPEECH_REAL_DATA=1 on a machine holding the released archive')
class ReleasedArchiveTests(unittest.TestCase):
    def test_default_archive_is_the_released_recordings(self):
        bundle = load_dataset('si_hom')
        expected = si_hom.EXPECTED_RELEASE
        self.assertEqual(len(bundle), expected['trials'])
        self.assertEqual(bundle.x.shape, (expected['trials'], 19, expected['samples']))
        self.assertEqual(bundle.metadata['subject_trials'], expected['subject_trials'])
        self.assertEqual(int(sum(bundle.metadata['author_test'])), expected['author_test_trials'])
        self.assertEqual(bundle.metadata['channel_names'], CHANNEL_NAMES)
        self.assertEqual(bundle.metadata['num_classes'], 8)
        self.assertEqual(sorted(set(bundle.y.tolist())), list(range(8)))
        self.assertEqual(sorted(set(bundle.groups.tolist())), SUBJECT_IDS)
        self.assertEqual(sorted(set(bundle.metadata['sample_blocks'])),
                         [f'B{block:02d}' for block in range(1, 10)])
        self.assertEqual(len(set(bundle.sample_ids)), expected['trials'])


if __name__ == '__main__':
    unittest.main()
