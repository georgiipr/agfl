"""Pairing must follow split identities, never file order or rounded means."""
from copy import deepcopy
import json
import numpy as np
import pytest
from scipy import stats
from agfl.analysis import analyze_results, holm, paired_statistics
from agfl.config import comparison_identity, experiment_identity, resolve_config


def save_run(root, attention, seed, value, split=None, model='eegnet', **overrides):
    config = resolve_config({'dataset': 'synthetic_eeg', 'model': model, 'attention': attention, 'seeds': [seed], 'device': 'cpu'})
    config.update(dataset_fingerprint='fixture-content', expected_split_id=split or f'split-{seed}')
    config.update(overrides)
    record = {'schema_version': 2, 'status': 'completed', 'dataset': config['dataset'], 'model': model, 'attention': attention,
              'model_variant': 'eeg', 'seed': seed, 'split_id': config['expected_split_id'],
              'dataset_fingerprint': config['dataset_fingerprint'], 'parameter_count': 100,
              'comparison_id': comparison_identity(config), 'experiment_id': experiment_identity(config),
              'config': config, 'validation': dict(accuracy=value, roc_auc=value, f1=value),
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


def test_duplicate_seeds_cannot_inflate_sample_size(tmp_path):
    record = save_run(tmp_path / 'runs', 'agfl', 0, .7)
    duplicate = tmp_path / 'runs' / 'copy'
    duplicate.mkdir()
    (duplicate / 'result.json').write_text(json.dumps(record))
    with pytest.raises(ValueError, match='pseudoreplication'):
        analyze_results(tmp_path / 'runs', tmp_path / 'analysis')


def test_different_training_protocols_are_not_paired(tmp_path):
    save_run(tmp_path / 'runs', 'agfl', 0, .7)
    cfg = resolve_config({'model': 'eegnet', 'attention': 'mha', 'dataset': 'synthetic_eeg'})
    cfg['training']['epochs'] = 100
    save_run(tmp_path / 'runs', 'mha', 0, .6, training=cfg['training'])
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['comparisons'] == []
    assert len(result['experiments_without_comparable_counterpart']) == 2


def test_different_backbones_are_not_paired_as_attention_effect(tmp_path):
    save_run(tmp_path / 'runs', 'agfl', 0, .7, model='eegnet')
    save_run(tmp_path / 'runs', 'mha', 0, .6, model='conformer')
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['comparisons'] == []
    assert {row['model'] for row in result['experiments']} == {'eegnet', 'conformer'}


def test_attention_settings_change_experiment_but_not_controlled_protocol():
    first = resolve_config({'model': 'eegnet', 'attention': 'agfl'})
    other = resolve_config({'model': 'eegnet', 'attention': 'mha'})
    assert comparison_identity(first) == comparison_identity(other)
    assert experiment_identity(first) != experiment_identity(other)
    other['model_options']['f2'] *= 2
    assert comparison_identity(first) != comparison_identity(other)


def test_v1_results_keep_saved_identities_and_correct_backbone_label(tmp_path):
    record = save_run(tmp_path / 'runs', 'agfl', 0, .7, model='signal_transformer')
    config = record['config']
    config['schema_version'], config['model'] = 1, 'agfl'
    config['comparison_family'] = 'signal_transformer_v1'
    config['model_options'].update(config.pop('attention_options'))
    del config['attention']
    record.update(schema_version=1, model='agfl', experiment_id=experiment_identity(config),
                  comparison_id=comparison_identity(config))
    del record['attention']
    path = next((tmp_path / 'runs').rglob('result.json'))
    path.write_text(json.dumps(record))
    before = path.read_bytes()
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    assert result['experiments'][0]['model'] == 'signal_transformer'
    assert result['experiments'][0]['attention'] == 'agfl'
    assert result['experiments'][0]['experiment_id'] == record['experiment_id']
    assert path.read_bytes() == before
