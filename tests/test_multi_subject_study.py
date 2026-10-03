"""Declared subject studies: one model per subject, subjects weighted equally.

Development checks for the experiment machine only; they are not a prerequisite
for training. A study is declared by the ``si-hom-individual`` preset: three
attention arms x seven subjects x five seeds on the individual cohort. The
declaration is validated when a configuration is resolved, and the summary of
saved runs is complete only when every declared arm/subject/seed is present.
No model is trained here; run records are constructed by hand.

Run from the project folder: ``python -m unittest tests.test_multi_subject_study``.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from agfl_speech.config import comparison_identity, merge, resolve_config, resolve_experiments
from agfl_speech.multi_subject_study import (summarize_subject_studies, validate_study_config,
                                             write_subject_study_reports)
from agfl_speech.presets import load_preset


ARMS = ['agfl', 'mha', 'hcann']
SUBJECT_IDS = [f'S{subject:02d}' for subject in range(1, 8)]
METRICS = ('accuracy', 'roc_auc', 'f1')


def arm_configs():
    """Unresolved configurations of the three study arms, as the CLI merges them."""
    document = load_preset('si-hom-individual')
    return [merge(document['base'], experiment) for experiment in document['experiments']]


def expanded():
    return [experiment for config in arm_configs() for experiment in resolve_experiments(config)]


def study_runs():
    """A complete, hand-made result matrix with subject-dependent arm differences."""
    runs = []
    for config in expanded():
        subject = config['subject_id']
        number = int(subject[1:])
        for seed in config['seeds']:
            saved = deepcopy(config)
            saved.update(seeds=[seed], dataset_fingerprint=subject, expected_split_id=f'{subject}-{seed}',
                         provenance={'source_sha256': 'same-source', 'packages': {}})
            score = {'agfl': .40, 'mha': .40, 'hcann': .38}[saved['study_arm']] + .03 * number + .002 * seed
            if saved['study_arm'] == 'agfl':
                score += .01 * number
            runs.append({'config': saved, 'seed': seed, 'split_id': saved['expected_split_id'],
                         'dataset_fingerprint': subject, 'comparison_id': comparison_identity(saved),
                         'token_axis': 'electrode',
                         'test': {metric: score for metric in METRICS},
                         'validation': {metric: .5 for metric in METRICS}})
    return runs


class StudyDeclarationTests(unittest.TestCase):
    def test_individual_preset_arms_are_accepted_after_expansion(self):
        experiments = expanded()
        self.assertEqual([(e['study_arm'], e['subject_id']) for e in experiments],
                         [(arm, subject) for arm in ARMS for subject in SUBJECT_IDS])
        for experiment in experiments:
            self.assertIsNone(validate_study_config(experiment))
            self.assertEqual(experiment['data']['cohort'], 'individual')
            self.assertEqual(experiment['data']['subjects'], [int(experiment['subject_id'][1:])])
            self.assertEqual(experiment['study']['subjects'], SUBJECT_IDS)
            self.assertEqual(experiment['study']['arms'], ARMS)
        # The unexpanded arm (no subject yet, all seven subjects) is valid too.
        for config in arm_configs():
            self.assertIsNone(validate_study_config(resolve_config(config)))

    def test_pooled_cohort_is_rejected(self):
        experiment = expanded()[0]
        experiment['data']['cohort'] = 'pooled'
        with self.assertRaisesRegex(ValueError, 'cohort=individual'):
            validate_study_config(experiment)
        config = arm_configs()[0]
        config['data']['cohort'] = 'pooled'
        with self.assertRaisesRegex(ValueError, 'cohort=individual'):
            resolve_experiments(config)

    def test_subject_outside_the_declared_subjects_is_rejected(self):
        experiment = expanded()[0]
        self.assertEqual(experiment['subject_id'], 'S01')
        experiment['study']['subjects'] = SUBJECT_IDS[1:]
        with self.assertRaisesRegex(ValueError, 'subject_id is not declared'):
            validate_study_config(experiment)
        relabelled = expanded()[0]
        relabelled['subject_id'] = 'S08'
        with self.assertRaisesRegex(ValueError, 'subject_id is not declared'):
            validate_study_config(relabelled)
        # Before expansion there is no subject_id; data.subjects is checked instead.
        config = arm_configs()[0]
        config['study']['subjects'] = SUBJECT_IDS[:6]
        with self.assertRaisesRegex(ValueError, 'data.subjects must belong'):
            resolve_experiments(config)
        subset = arm_configs()[0]
        subset['data']['subjects'] = [2, 5]
        self.assertEqual([e['subject_id'] for e in resolve_experiments(subset)], ['S02', 'S05'])

    def test_seeds_outside_the_declared_seeds_are_rejected(self):
        experiment = expanded()[0]
        experiment['seeds'] = [0, 7]
        with self.assertRaisesRegex(ValueError, 'subset of study.seeds'):
            validate_study_config(experiment)
        config = arm_configs()[0]
        config['seeds'] = [5]
        with self.assertRaisesRegex(ValueError, 'subset of study.seeds'):
            resolve_experiments(config)
        config['seeds'] = [0, 1]
        self.assertEqual(len(resolve_experiments(config)), 7)

    def test_arm_and_declaration_must_be_complete(self):
        experiment = expanded()[0]
        undeclared = deepcopy(experiment)
        undeclared['study_arm'] = 'performer'
        with self.assertRaisesRegex(ValueError, 'study_arm'):
            validate_study_config(undeclared)
        without_arm = deepcopy(experiment)
        del without_arm['study_arm']
        with self.assertRaisesRegex(ValueError, 'study_arm'):
            validate_study_config(without_arm)
        without_study = deepcopy(experiment)
        del without_study['study']
        with self.assertRaisesRegex(ValueError, 'study requires'):
            resolve_config(without_study)
        for key in ('name', 'subjects', 'seeds', 'arms'):
            incomplete = deepcopy(experiment)
            del incomplete['study'][key]
            with self.subTest(missing=key):
                with self.assertRaisesRegex(ValueError, 'study requires'):
                    validate_study_config(incomplete)
        extended = deepcopy(experiment)
        extended['study']['notes'] = 'undeclared field'
        with self.assertRaisesRegex(ValueError, 'study requires'):
            validate_study_config(extended)

    def test_explicit_contrasts_reject_unknown_self_and_duplicate_pairs(self):
        experiment = expanded()[0]
        for pairs in ([], ['agfl'], [['agfl', 'unknown']], [['agfl', 'agfl']],
                      [['agfl', 'mha'], ['agfl', 'mha']], [['agfl', 'mha'], ['mha', 'agfl']]):
            with self.subTest(pairs=pairs):
                changed = deepcopy(experiment)
                changed['study']['comparisons'] = pairs
                with self.assertRaisesRegex(ValueError, 'study.comparisons'):
                    validate_study_config(changed)
        optional = deepcopy(experiment)
        del optional['study']['comparisons']
        self.assertIsNone(validate_study_config(optional))


class StudySummaryTests(unittest.TestCase):
    def test_complete_matrix_weights_subjects_equally_and_pairs_by_subject(self):
        runs = study_runs()
        self.assertEqual(len(runs), 3 * 7 * 5)
        study, = summarize_subject_studies(runs)
        self.assertEqual(study['name'], 'si-hom-individual-eegnet')
        self.assertTrue(study['complete'])
        self.assertEqual(study['issues'], [])
        self.assertEqual(study['missing_runs'], [])
        self.assertEqual((study['expected_runs'], study['completed_runs']), (105, 105))
        self.assertEqual(study['token_axis'], 'electrode')
        self.assertEqual(study['plan']['subjects'], SUBJECT_IDS)
        # Two declared contrasts (AGFL against each baseline) x three metrics.
        self.assertEqual(len(study['comparisons']), 6)
        self.assertEqual({(row['candidate'], row['baseline']) for row in study['comparisons']},
                         {('agfl', 'mha'), ('agfl', 'hcann')})
        average = next(row for row in study['averages'] if row['arm'] == 'agfl'
                       and row['partition'] == 'test' and row['metric'] == 'accuracy')
        # Subject mean: .40 + .03 s + .004 + .01 s; mean over s = 1..7 is .564.
        self.assertAlmostEqual(average['mean'], .564)
        self.assertEqual((average['n_subjects'], average['expected_subjects']), (7, 7))
        for baseline, delta in (('mha', .04), ('hcann', .06)):
            contrast = next(row for row in study['comparisons'] if row['baseline'] == baseline
                            and row['metric'] == 'accuracy')
            self.assertEqual(contrast['candidate'], 'agfl')
            self.assertEqual(contrast['n_pairs'], 7)  # seven people, never 35 seed rows
            self.assertAlmostEqual(contrast['mean_delta'], delta)
            self.assertEqual(contrast['contrast_selection'], 'declared')
            self.assertEqual(contrast['matched_subjects'], SUBJECT_IDS)
        for row in study['comparisons']:
            self.assertIsNotNone(row['t_pvalue_holm'])
            self.assertIsNotNone(row['wilcoxon_pvalue_holm'])
        labels = {row['arm']: row['attention_label'] for row in study['averages']}
        self.assertEqual(labels, {'agfl': 'AGFL (static)', 'mha': 'MHA', 'hcann': 'HCANN attention'})

    def test_missing_arm_subject_or_seed_is_disclosed_without_statistics(self):
        runs = study_runs()
        for filtered in ([run for run in runs if run['config']['subject_id'] != 'S07'],
                         [run for run in runs if run['config']['study_arm'] != 'hcann'],
                         [run for run in runs if run['config']['study_arm'] != 'agfl'],
                         runs[:-1]):
            with self.subTest(count=len(filtered)):
                study, = summarize_subject_studies(filtered)
                self.assertFalse(study['complete'])
                self.assertEqual(len(study['missing_runs']), 105 - len(filtered))
                self.assertEqual(len(study['comparisons']), 6)
                self.assertTrue(all(row['mean_delta'] is None and row['t_pvalue'] is None
                                    for row in study['comparisons']))

    def test_mismatched_split_or_source_cannot_produce_a_matched_average(self):
        for mismatch in ('split', 'source'):
            runs = study_runs()
            if mismatch == 'split':
                runs[-1]['split_id'] = 'different'
            else:
                runs[-1]['config']['provenance']['source_sha256'] = 'different'
            with self.subTest(mismatch=mismatch):
                study, = summarize_subject_studies(runs)
                self.assertFalse(study['complete'])
                self.assertTrue(study['issues'])
                self.assertTrue(all(row['mean'] is None for row in study['averages']))
        runs = study_runs()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            summarize_subject_studies(runs + [deepcopy(runs[0])])

    def test_report_tables_state_completeness(self):
        runs = study_runs()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_subject_study_reports(root, summarize_subject_studies(runs))
            text = (root / 'subject_study.md').read_text()
            self.assertIn('**COMPLETE: 105/105 fits.**', text)
            self.assertIn('| AGFL (static) | 7/7 |', text)
            for name in ('subject_study.json', 'subject_study_per_subject.csv',
                         'subject_study_averages.csv', 'subject_study_comparisons.csv'):
                self.assertTrue((root / name).is_file(), name)
            partial = [run for run in runs if run['config']['subject_id'] != 'S07']
            write_subject_study_reports(root, summarize_subject_studies(partial))
            text = (root / 'subject_study.md').read_text()
            self.assertIn('INCOMPLETE', text)
            self.assertIn('90/105 fits', text)


if __name__ == '__main__':
    unittest.main()
