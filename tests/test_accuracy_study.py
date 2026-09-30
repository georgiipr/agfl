"""Optional cluster-only checks for averaging and checkpoint-free training.

Uses unittest and installed training dependencies. Never part of the sbatch
launch and never run in the local workspace.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from agfl.accuracy_study import (MODELS, SUBJECTS, SEEDS, EXPECTED_FITS,
                                 accuracy_rows, study_configs, write_accuracy_tables)
from agfl.config import resolve_config
from agfl.engine import run_experiment, run_training


class AccuracyStudyTests(unittest.TestCase):
    def records(self):
        # Different subject test sizes deliberately make trial pooling wrong.
        values = {'A03': (.8, 50), 'A04': (.4, 20), 'A09': (.9, 60)}
        return [{'model': model, 'attention': attention, 'subject_id': subject, 'seed': seed,
                 'status': 'completed', 'test': {'accuracy': values[subject][0] - (.1 if attention == 'mha' else 0),
                                                'n_samples': values[subject][1]}}
                for model in MODELS for attention in ('agfl', 'mha')
                for subject in SUBJECTS for seed in SEEDS]

    def test_means_weight_subjects_equally_and_keep_all_models(self):
        records = self.records()
        self.assertEqual(len(records), EXPECTED_FITS)
        rows = accuracy_rows(records)
        self.assertEqual(len(rows), 5)
        for row in rows:
            self.assertAlmostEqual(row['mean_agfl_percent'], 70.)
            self.assertAlmostEqual(row['mean_mha_percent'], 60.)
            self.assertAlmostEqual(row['agfl_minus_mha_pp'], 10.)
            self.assertEqual(row['completed_fits'], 30)

    def test_missing_seed_withholds_subject_mean_and_overall_comparison(self):
        records = self.records()
        records.pop(0)
        row = accuracy_rows(records)[0]
        self.assertIsNone(row['A03_agfl_percent'])
        self.assertIsNone(row['mean_agfl_percent'])
        self.assertIsNone(row['agfl_minus_mha_pp'])
        self.assertAlmostEqual(row['mean_mha_percent'], 60.)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            accuracy_rows(records + [records[0]])

    def test_tables_replace_stale_full_scores_with_current_partial_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_accuracy_tables(root, self.records(), [], status='complete')
            write_accuracy_tables(root, [], [], status='running')
            with (root / 'accuracy_table.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(all(row['mean_agfl_percent'] == '' for row in rows))
            self.assertEqual(json.loads((root / 'accuracy_status.json').read_text())['completed_fits'], 0)
            self.assertIn('pending', (root / 'accuracy_table.md').read_text())

    def test_preset_resolves_to_matched_independent_subjects(self):
        with tempfile.TemporaryDirectory() as directory:
            configs = study_configs(Path(directory) / 'data', Path(directory) / 'out')
        self.assertEqual(len(configs), 30)
        self.assertEqual(sum(len(c['seeds']) for c in configs.values()), 150)
        for (model, attention, subject), c in configs.items():
            self.assertEqual(c['data']['subjects'], [int(subject[1:])])
            self.assertFalse(c['training']['save_checkpoints'])
            self.assertFalse(c['attention_options'].get('temporal_bias', False))
            if model == 'eegnet':
                self.assertEqual(c['model_options']['electrode_architecture'], 'spatial_fusion')

    def test_checkpoint_retention_option_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, 'save_checkpoints'):
            resolve_config({'training': {'save_checkpoints': 'false'}})

    def test_validation_only_search_cannot_discard_required_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'Validation-only'):
                run_training({'training': {'save_checkpoints': False}}, None, None,
                             Path(directory), validation_only=True)

    def test_memory_selection_matches_disk_selection_and_never_calls_torch_save(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {'dataset': 'synthetic_eeg', 'model': 'signal_transformer', 'attention': 'mha',
                   'device': 'cpu', 'seeds': [3], 'threads': 1,
                   'output_dir': str(root / 'disk'), 'split_dir': str(root / 'splits'),
                   'data': {'subjects': 6, 'samples_per_subject': 8, 'channels': 4, 'samples': 32},
                   'model_options': {'dim': 8, 'depth': 1, 'eeg_kernel_size': 3, 'eeg_temporal_bins': 2},
                   'attention_options': {'heads': 2}, 'training': {'epochs': 3, 'batch_size': 8}}
            disk, = run_experiment(cfg)
            memory_config = deepcopy(cfg)
            memory_config['output_dir'] = str(root / 'memory')
            memory_config['training']['save_checkpoints'] = False
            with patch('torch.save', side_effect=AssertionError('Checkpoint write is forbidden')):
                memory, = run_experiment(memory_config)
            self.assertTrue(disk['checkpoint_retained'])
            self.assertFalse(memory['checkpoint_retained'])
            self.assertEqual(memory['best_checkpoint_epoch'], disk['best_checkpoint_epoch'])
            self.assertEqual(memory['validation'], disk['validation'])
            self.assertEqual(memory['test'], disk['test'])
            self.assertFalse(list((root / 'memory').rglob('*.pt')))
            left = next((root / 'disk' / 'artifacts').rglob('predictions.npz'))
            right = next((root / 'memory' / 'artifacts').rglob('predictions.npz'))
            with np.load(left) as saved, np.load(right) as in_memory:
                for key in saved.files:
                    np.testing.assert_array_equal(saved[key], in_memory[key])
            # The general engine can recognize completed accuracy-only runs;
            # the queued study itself deliberately always retrains.
            with patch('agfl.engine.run_training', side_effect=AssertionError('Unexpected retraining')):
                skipped, = run_experiment(memory_config, skip_completed=True)
            self.assertEqual(skipped['test'], memory['test'])


if __name__ == '__main__':
    unittest.main()
