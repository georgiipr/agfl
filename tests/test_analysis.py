"""Pairing must follow split identities, never file order or rounded means."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scipy import stats
from agfl_speech.analysis import analyze_results, conditioning_control, holm, paired_statistics
from agfl_speech.config import comparison_identity, experiment_identity, resolve_config


def save_run(root, attention, seed, value, split=None, model='eegnet', **overrides):
    # resolve_config never opens the data archive, so saved-result fixtures need no data folder.
    config = resolve_config({'dataset': 'si_hom', 'model': model, 'attention': attention, 'seeds': [seed], 'device': 'cpu'})
    config.update(dataset_fingerprint='fixture-content', expected_split_id=split or f'split-{seed}')
    config.update(overrides)
    record = {'schema_version': 2, 'status': 'completed', 'dataset': config['dataset'], 'model': model, 'attention': attention,
              'model_variant': 'eeg', 'seed': seed, 'split_id': config['expected_split_id'],
              'dataset_fingerprint': config['dataset_fingerprint'], 'parameter_count': 100,
              'comparison_id': comparison_identity(config), 'experiment_id': experiment_identity(config),
              'config': config, 'validation': dict(accuracy=value, roc_auc=value, f1=value),
              'subject_id': config.get('subject_id'),
              'test': dict(accuracy=value, roc_auc=value, f1=value)}
    path = root / model / attention / str(seed)
    path.mkdir(parents=True)
    (path / 'result.json').write_text(json.dumps(record))
    return record


def test_paired_tests_match_scipy():
    a, b = [0.8, .9, .7, .85, .78], [.75, .88, .73, .79, .76]
    result = paired_statistics(a, b)
    expected = stats.ttest_rel(a, b)
    assert result['t_pvalue'] == pytest.approx(expected.pvalue)
    assert result['cohen_dz'] == pytest.approx(np.mean(np.subtract(a, b)) / np.std(np.subtract(a, b), ddof=1))
    assert result['wilcoxon_pvalue'] == pytest.approx(stats.wilcoxon(np.round(np.subtract(a, b), 12)).pvalue)
    assert paired_statistics([1, 1], [1, 1])['t_pvalue'] == 1
    assert paired_statistics([1], [0])['t_pvalue'] is None
    assert paired_statistics([1, 1], [0, 0])['cohen_dz'] is None


def test_individual_subject_summary_does_not_pool_people_and_seeds(tmp_path):
    for subject, scores in [('S01', [.6, .8]), ('S02', [.9, 1.0])]:
        for seed, score in enumerate(scores):
            save_run(tmp_path / 'runs' / subject, 'agfl', seed, score,
                     subject_id=subject, data={'subjects': [int(subject[1:])]})
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert {r['subject_id'] for r in result['experiments']} == {'S01', 'S02'}
    row = next(r for r in result['across_subjects'] if r['partition'] == 'test' and r['metric'] == 'accuracy')
    assert row['subjects'] == ['S01', 'S02'] and row['n_subjects'] == 2
    assert row['mean_of_subject_means'] == pytest.approx(.825)
    assert row['between_subject_sd'] == pytest.approx(np.std([.7, .95], ddof=1))
    save_run(tmp_path / 'runs' / 'S01', 'agfl', 2, .7, subject_id='S01', data={'subjects': [1]})
    partial = analyze_results(tmp_path / 'runs', tmp_path / 'partial')
    assert all(r['mean_of_subject_means'] is None and r['reason'] for r in partial['across_subjects'])


def test_holm_respects_order_and_missing_values():
    values = holm([.04, None, .01, .03])
    assert values[1] is None
    assert [values[i] for i in (0, 2, 3)] == pytest.approx([.06, .03, .06])


def test_aggregation_uses_saved_files_and_exact_split_pairs(tmp_path):
    for seed in range(5):
        save_run(tmp_path / 'runs', 'agfl', seed, .7 + seed * .01)
        save_run(tmp_path / 'runs', 'mha', seed, .68 + seed * .008,
                 split='unmatched' if seed == 4 else None)
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['n_runs'] == 10
    assert len(result['comparisons']) == 3
    for row in result['comparisons']:
        assert row['n_pairs'] == 4
        assert row['n_unmatched_agfl'] == 1
        assert row['n_unmatched_baseline'] == 1
        assert row['t_pvalue_holm'] >= row['t_pvalue']
    summary = next(row for row in result['experiments'] if row['attention'] == 'agfl')
    assert summary['metrics']['test_accuracy']['std'] == pytest.approx(np.std([.7, .71, .72, .73, .74], ddof=1))
    for name in ['aggregation.json', 'per_model.csv', 'attention_comparison.csv', 'agfl_ablations.csv', 'statistical_comparisons.csv', 'report.md']:
        assert (tmp_path / 'analysis' / name).is_file()


def test_agfl_is_paired_with_every_primary_baseline_of_the_same_backbone(tmp_path):
    for seed in range(3):
        save_run(tmp_path / 'runs', 'agfl', seed, .7 + seed * .01)
        save_run(tmp_path / 'runs', 'mha', seed, .65 + seed * .02)
        save_run(tmp_path / 'runs', 'hcann', seed, .6 + seed * .03)
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['n_runs'] == 9
    assert len(result['comparisons']) == 6  # Two baselines, three metrics; no baseline-versus-baseline rows.
    assert {row['baseline'] for row in result['comparisons']} == {'mha', 'hcann'}
    assert {row['baseline_label'] for row in result['comparisons']} == {'MHA', 'HCANN attention'}
    for row in result['comparisons']:
        assert row['comparison_kind'] == 'attention' and row['n_pairs'] == 3
        assert row['agfl_label'] == 'AGFL (static)'
    assert result['experiments_without_comparable_counterpart'] == []


def test_duplicate_seeds_cannot_inflate_sample_size(tmp_path):
    record = save_run(tmp_path / 'runs', 'agfl', 0, .7)
    duplicate = tmp_path / 'runs' / 'copy'
    duplicate.mkdir()
    (duplicate / 'result.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='pseudoreplication'):
        analyze_results(tmp_path / 'runs', tmp_path / 'analysis')


def test_different_training_protocols_are_not_paired(tmp_path):
    save_run(tmp_path / 'runs', 'agfl', 0, .7)
    cfg = resolve_config({'model': 'eegnet', 'attention': 'mha', 'dataset': 'si_hom'})
    cfg['training']['epochs'] = 100
    save_run(tmp_path / 'runs', 'mha', 0, .6, training=cfg['training'])
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['comparisons'] == []
    assert len(result['experiments_without_comparable_counterpart']) == 2


def test_different_backbones_are_not_paired_as_attention_effect(tmp_path):
    save_run(tmp_path / 'runs', 'agfl', 0, .7, model='eegnet')
    save_run(tmp_path / 'runs', 'mha', 0, .6, model='signal_transformer')
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['comparisons'] == []
    assert {row['model'] for row in result['experiments']} == {'eegnet', 'signal_transformer'}


def test_attention_settings_change_experiment_but_not_controlled_protocol():
    first = resolve_config({'model': 'eegnet', 'attention': 'agfl'})
    other = resolve_config({'model': 'eegnet', 'attention': 'mha'})
    assert comparison_identity(first) == comparison_identity(other)
    assert experiment_identity(first) != experiment_identity(other)
    other['model_options']['f2'] *= 2
    assert comparison_identity(first) != comparison_identity(other)


def test_results_saved_before_schema_version_two_are_rejected(tmp_path):
    record = save_run(tmp_path / 'runs', 'agfl', 0, .7)
    config = record['config']
    config['schema_version'] = 1
    record.update(schema_version=1, experiment_id=experiment_identity(config),
                  comparison_id=comparison_identity(config))
    next((tmp_path / 'runs').rglob('result.json')).write_text(json.dumps(record))
    with pytest.raises(ValueError, match='predates schema_version 2'):
        analyze_results(tmp_path / 'runs', tmp_path / 'analysis')


@pytest.mark.parametrize('mismatch', [
    {'coefficient_conditioning_scale': .25}, {'top_k': 3}, {'K': 3},
    {'coefficient_conditioning': 'trial_power'}, {'coefficient_conditioning': 'feature_contrast'},
])
def test_feature_to_token_pair_rejects_unmatched_options(mismatch):
    token = {'coefficient_conditioning': 'token_contrast', 'coefficient_conditioning_scale': .5, 'top_k': 5, 'K': 2}
    feature = {**token, 'coefficient_conditioning': 'feature_contrast'}
    assert conditioning_control({'attention_options': feature}, {'attention_options': token})
    assert not conditioning_control({'attention_options': feature}, {'attention_options': {**token, **mismatch}})
