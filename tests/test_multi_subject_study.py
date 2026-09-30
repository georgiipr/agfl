"""Optional target-machine verification; do not execute on the local workspace."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from agfl.analysis import attention_display_name, subject_aggregates
from agfl.config import merge, resolve_experiments, experiment_identity, comparison_identity
from agfl.presets import load_preset
from agfl.multi_subject_study import summarize_subject_studies, write_subject_study_reports, validate_study_config
from agfl.visualization.common import FigureWriter
from agfl.visualization.subject_study import plot_subject_studies


class SubjectStudyTests(unittest.TestCase):
    def test_original_enhanced_and_mha_preset_has_all_three_declared_contrasts(self):
        from agfl.attention.agfl.config import DEFAULTS as AGFL_DEFAULTS
        from agfl.attention.mha.config import DEFAULTS as MHA_DEFAULTS
        doc = load_preset('eegnet-a03-a04-interchannel')
        original, enhanced, mha = doc['experiments']
        self.assertEqual(original['attention_options'], AGFL_DEFAULTS)
        self.assertEqual(mha['attention_options'], MHA_DEFAULTS)
        self.assertEqual(enhanced['attention_options']['projection'], 'qkv')
        self.assertEqual(enhanced['attention_options']['coefficient_conditioning'], 'token_contrast')
        self.assertTrue(enhanced['attention_options']['output_gate'])
        runs = []
        for arm in doc['experiments']:
            for config in resolve_experiments(merge(doc['base'], arm)):
                for seed in config['seeds']:
                    c = deepcopy(config)
                    subject = c['subject_id']
                    c.update(seeds=[seed], dataset_fingerprint=subject, expected_split_id=f'{subject}-{seed}',
                             provenance={'source_sha256': 'same-source', 'packages': {}})
                    score = {'agfl_base': .7, 'agfl_enhanced': .72, 'mha_base': .71}[c['study_arm']]
                    score += .1 if subject == 'A04' else 0
                    runs.append({'config': c, 'seed': seed, 'split_id': c['expected_split_id'],
                                 'dataset_fingerprint': subject, 'comparison_id': comparison_identity(c),
                                 'token_axis': 'electrode',
                                 'test': {m: score for m in ('accuracy', 'roc_auc', 'f1')},
                                 'validation': {m: .5 for m in ('accuracy', 'roc_auc', 'f1')}})
        self.assertEqual(len(runs), 30)
        study, = summarize_subject_studies(runs)
        self.assertTrue(study['complete'])
        self.assertEqual(study['plan']['subjects'], ['A03', 'A04'])
        self.assertEqual(study['expected_runs'], 30)
        self.assertEqual(len(study['comparisons']), 9)
        self.assertEqual({(r['candidate'], r['baseline']) for r in study['comparisons']},
                         {tuple(p) for p in doc['base']['study']['comparisons']})
        contrast = next(r for r in study['comparisons'] if r['candidate'] == 'agfl_enhanced'
                        and r['baseline'] == 'agfl_base' and r['metric'] == 'accuracy')
        self.assertAlmostEqual(contrast['mean_delta'], .02)
        self.assertEqual(contrast['n_pairs'], 2)
        self.assertEqual(contrast['contrast_selection'], 'declared')
        missing, = summarize_subject_studies([r for r in runs if r['config']['study_arm'] != 'agfl_base'])
        self.assertFalse(missing['complete'])
        self.assertEqual(len(missing['comparisons']), 9)
        self.assertTrue(all(r['mean_delta'] is None and r['t_pvalue'] is None for r in missing['comparisons']))

    def test_explicit_contrasts_reject_unknown_self_and_duplicate_pairs(self):
        doc = load_preset('eegnet-a03-a04-interchannel')
        config = merge(doc['base'], doc['experiments'][0])
        for pairs in ([], ['agfl_base'], [['agfl_base', 'unknown']], [['agfl_base', 'agfl_base']],
                      [['agfl_base', 'mha_base'], ['agfl_base', 'mha_base']],
                      [['agfl_base', 'mha_base'], ['mha_base', 'agfl_base']]):
            with self.subTest(pairs=pairs):
                changed = deepcopy(config)
                changed['study']['comparisons'] = pairs
                with self.assertRaisesRegex(ValueError, 'study.comparisons'):
                    validate_study_config(changed)

    def test_interchannel_matrix_is_315_fits_with_matched_gate_controls(self):
        new = load_preset('eegnet-interchannel-attentions')
        configs = [c for arm in new['experiments'] for c in resolve_experiments(merge(new['base'], arm))]
        self.assertEqual(len(configs), 63)
        self.assertEqual(sum(len(c['seeds']) for c in configs), 315)
        self.assertEqual(len({experiment_identity(c) for c in configs}), 63)
        self.assertTrue(all(c['model_options']['attention_axis'] == 'electrode' for c in configs))
        self.assertTrue(all(not c['attention_options'].get('temporal_bias', False) for c in configs))
        for subject in new['base']['study']['subjects']:
            group = [c for c in configs if c['subject_id'] == subject]
            self.assertEqual(len(group), 7)
            self.assertEqual(len({comparison_identity(c) for c in group}), 1)
            self.assertEqual(len({attention_display_name(c['attention'], c['attention_options']) for c in group}), 7)

    def fixtures(self):
        doc = load_preset('eegnet-interchannel-comparison')
        doc['experiments'] = doc['experiments'][:3]
        base = doc['base']
        base.update(seeds=[0, 1])
        base['data']['subjects'] = [1, 2]
        base['study'].update(subjects=['A01', 'A02'], seeds=[0, 1], arms=[a['study_arm'] for a in doc['experiments']])
        runs = []
        for arm in doc['experiments']:
            for config in resolve_experiments(merge(base, arm)):
                for seed in config['seeds']:
                    c = deepcopy(config)
                    subject = c['subject_id']
                    c.update(seeds=[seed], dataset_fingerprint=subject, expected_split_id=f'{subject}-{seed}',
                             provenance={'source_sha256': 'same-source', 'packages': {}})
                    score = (.7 if subject == 'A01' else .9) + .01 * seed
                    if c['study_arm'] == 'agfl_gated':
                        score += .03 if subject == 'A01' else .01
                    elif c['study_arm'] == 'mha_gated':
                        score += .01
                    runs.append({'config': c, 'seed': seed, 'split_id': c['expected_split_id'],
                                 'dataset_fingerprint': subject, 'comparison_id': comparison_identity(c),
                                 'dataset': 'eeg', 'backbone_key': 'eegnet', 'attention_key': c['attention'],
                                 'test': {m: score for m in ('accuracy', 'roc_auc', 'f1')},
                                 'validation': {m: .5 for m in ('accuracy', 'roc_auc', 'f1')}})
        return runs

    def test_equal_subject_weights_and_subject_paired_statistics(self):
        study, = summarize_subject_studies(self.fixtures())
        self.assertTrue(study['complete'])
        self.assertEqual(study['expected_runs'], 12)
        self.assertEqual(len(study['comparisons']), 9)  # three contrasts, three metrics
        average = next(r for r in study['averages'] if r['arm'] == 'agfl_gated' and r['partition'] == 'test' and r['metric'] == 'accuracy')
        self.assertAlmostEqual(average['mean'], .825)
        comparison = next(r for r in study['comparisons'] if r['candidate'] == 'agfl_gated' and r['baseline'] == 'mha' and r['metric'] == 'accuracy')
        self.assertEqual(comparison['n_pairs'], 2)  # two people, never four seed rows
        self.assertAlmostEqual(comparison['mean_delta'], .02)
        for row in study['comparisons']:
            self.assertIsNotNone(row['wilcoxon_pvalue_holm'])

    def test_whole_missing_subject_arm_or_single_seed_is_disclosed(self):
        runs = self.fixtures()
        for filtered in ([r for r in runs if r['config']['subject_id'] != 'A02'],
                         [r for r in runs if r['config']['study_arm'] != 'mha_gated'], runs[:-1]):
            with self.subTest(count=len(filtered)):
                study, = summarize_subject_studies(filtered)
                self.assertFalse(study['complete'])
                self.assertTrue(study['missing_runs'])
                self.assertTrue(all(r['t_pvalue'] is None for r in study['comparisons']))
        partial = [r for r in runs if r['config']['subject_id'] != 'A02']
        self.assertTrue(all(r['mean_of_subject_means'] is None for r in subject_aggregates(partial)))

    def test_mismatched_split_or_source_cannot_produce_a_matched_average(self):
        for mismatch in ('split', 'source'):
            runs = self.fixtures()
            if mismatch == 'split':
                runs[-1]['split_id'] = 'different'
            else:
                runs[-1]['config']['provenance']['source_sha256'] = 'different'
            study, = summarize_subject_studies(runs)
            self.assertFalse(study['complete'])
            self.assertTrue(study['issues'])
            self.assertTrue(all(r['mean'] is None for r in study['averages']))
        runs = self.fixtures()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize_subject_studies(runs + [deepcopy(runs[0])])

    def test_tables_and_all_seven_aggregate_figures_are_written(self):
        studies = summarize_subject_studies(self.fixtures())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_subject_study_reports(root, studies)
            self.assertIn('COMPLETE', (root / 'subject_study.md').read_text())
            writer = FigureWriter(root / 'figures')
            plot_subject_studies(writer, studies)
            self.assertEqual(len(writer.figures), 7)  # three heatmaps, one overview, three contrasts
            for figure in writer.figures:
                for filename in figure['files']:
                    self.assertTrue((writer.root / filename).is_file())


if __name__ == '__main__':
    unittest.main()
