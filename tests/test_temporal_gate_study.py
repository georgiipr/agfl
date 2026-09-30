"""Target-machine checks of matched arms, validation selection and reports."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from agfl.analysis import analyze_results, attention_display_name
from agfl.config import comparison_identity, experiment_identity, merge, resolve_experiments
from tests.references.presets import load_archived_preset as load_preset, archived_experiments
from agfl.temporal_gate_study import temporal_gate_control


class TemporalGateStudyTests(unittest.TestCase):
    def configs(self):
        document = load_preset('eegnet-a03-temporal-gate')
        return [archived_experiments(merge(document['base'], arm))[0] for arm in document['experiments']]

    def test_only_two_attention_flags_differ_from_retained_recipe(self):
        previous, current = load_preset('eegnet-a03-token-agfl'), load_preset('eegnet-a03-temporal-gate')
        base = deepcopy(current['base'])
        base['output_dir'] = previous['base']['output_dir']
        self.assertEqual(base, previous['base'])
        for arm in current['experiments']:
            original = deepcopy(arm)
            original['attention_options'].pop('temporal_bias')
            original['attention_options'].pop('output_gate')
            self.assertEqual(original, previous['experiments'][1])
        configs = self.configs()
        self.assertEqual(sum(len(c['seeds']) for c in configs), 20)
        self.assertEqual(len({comparison_identity(c) for c in configs}), 1)
        self.assertEqual(len({experiment_identity(c) for c in configs}), 4)
        self.assertEqual(len({attention_display_name('agfl', c['attention_options']) for c in configs}), 4)
        self.assertEqual(sum(temporal_gate_control(a, b) for a in configs for b in configs), 5)

    def save_fixture(self, root, *, mismatch=False, incomplete=False):
        configs = self.configs()
        for arm, config in enumerate(configs):
            for seed in range(5):
                if incomplete and arm == 3 and seed == 4:
                    continue
                config = deepcopy(config)
                config.update(seeds=[seed], dataset_fingerprint='fixture',
                              expected_split_id=f'split-{seed}' + ('-other' if mismatch and arm == 3 and seed == 4 else ''),
                              provenance={'source_sha256': 'same-code', 'packages': {}})
                val = [.80, .84, .81, .82][arm]  # bias wins validation; both wins test
                test = [.75, .76, .77, .83][arm]
                record = {'schema_version': 2, 'status': 'completed', 'config': config,
                          **{k: config[k] for k in ('dataset', 'model', 'attention', 'model_variant', 'subject_id', 'dataset_fingerprint')},
                          'seed': seed, 'split_id': config['expected_split_id'], 'parameter_count': [8244, 8256, 8312, 8324][arm],
                          'experiment_id': experiment_identity(config), 'comparison_id': comparison_identity(config),
                          'validation': {'accuracy': val, 'loss': .2, 'f1': val, 'roc_auc': val},
                          'test': {'accuracy': test, 'f1': test, 'roc_auc': test}}
                directory = root / f'arm-{arm}' / f'seed-{seed}'
                directory.mkdir(parents=True)
                (directory / 'result.json').write_text(json.dumps(record))

    def test_reports_pair_five_contrasts_and_never_select_using_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_fixture(root / 'runs')
            result = analyze_results(root / 'runs', root / 'analysis')
            self.assertEqual(len(result['comparisons']), 15)  # five contrasts, three metrics
            self.assertTrue(all(r['n_pairs'] == 5 for r in result['comparisons']))
            study, = result['temporal_gate_studies']
            self.assertTrue(study['complete'])
            self.assertEqual(study['selected_arm'], 'temporal_bias')
            for row in study['per_seed']:
                self.assertAlmostEqual(row['interaction'], .05 if row['partition'] == 'test' else -.03)
            self.assertTrue((root / 'analysis/temporal_gate_study.md').is_file())

    def test_partial_or_mismatched_matrix_cannot_select_an_architecture(self):
        for flags in ({'mismatch': True}, {'incomplete': True}):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.save_fixture(root / 'runs', **flags)
                result = analyze_results(root / 'runs', root / 'analysis')
                study, = result['temporal_gate_studies']
                self.assertFalse(study['complete'])
                self.assertIsNone(study['selected_arm'])
                self.assertEqual(study['matched_seeds'], [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main()
