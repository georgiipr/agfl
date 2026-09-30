"""Cluster-only checks for the fixed conditioning comparison and its reports."""
from copy import deepcopy
import json

import pytest

from agfl.analysis import analyze_results, attention_display_name
from agfl.config import comparison_identity, experiment_identity, merge, resolve_config, resolve_experiments
from agfl.models.eegnet.search import search_configs
from tests.references.presets import load_archived_preset as load_preset, archived_experiments
from agfl.visualization.results import selection_label
from tests.test_analysis import save_run


def test_comparison_retains_the_control_and_strongest_observed_baseline():
    document = load_preset('eegnet-a03-conditioned-agfl')
    original = load_preset('eegnet-a03-all-attentions')
    shared = deepcopy(document['base'])
    shared['output_dir'] = original['base']['output_dir']
    assert shared == original['base']
    control, conditioned, baseline = document['experiments']
    previous = deepcopy(next(e for e in original['experiments'] if e['attention'] == 'agfl'))
    previous['attention_options'].update(coefficient_conditioning='static', coefficient_conditioning_scale=.5)
    assert control == previous
    expected = deepcopy(control)
    expected['attention_options']['coefficient_conditioning'] = 'trial_power'
    assert conditioned == expected
    assert baseline == next(e for e in original['experiments'] if e['attention'] == 'linformer'
                            and e['attention_options']['projection_rank'] == 4)
    configs = [config for variant in document['experiments']
               for config in archived_experiments(merge(document['base'], variant))]
    assert len(configs) == 3
    assert sum(len(config['seeds']) for config in configs) == 15
    assert {config['subject_id'] for config in configs} == {'A03'}
    assert {config['model'] for config in configs} == {'eegnet'}
    assert all(config['training']['epochs'] == 250 for config in configs)
    assert len({comparison_identity(config) for config in configs}) == 1
    assert len({experiment_identity(config) for config in configs}) == 3
    assert all(1000 // config['model_options']['pk1'] // config['model_options']['pk2'] == 7
               for config in configs)


def test_retired_temporal_search_cannot_be_started(tmp_path):
    with pytest.raises(ValueError, match='candidates'):
        search_configs('../ml', tmp_path, [3], [0], ['spatial_qkv_sparse'])
    current = search_configs('../ml', tmp_path, [3], [0])
    assert all(c['model_options']['attention_axis'] == 'electrode'
               for group in current.values() for c in group)


def save_comparison(root, name, attention, seed, accuracy, *, options=None, split=None):
    config = resolve_config({'dataset': 'synthetic_eeg', 'model': 'eegnet',
                             'attention': attention, 'attention_options': options or {}, 'device': 'cpu'})
    return save_run(root / name, attention, seed, accuracy, split=split,
                    attention_options=config['attention_options'], subject_id='A03', data={'subjects': [3]})


def test_variants_remain_separate_and_only_matching_controls_are_paired(tmp_path):
    source, output = tmp_path / 'runs', tmp_path / 'report'
    ids = {}
    for seed in range(3):
        for name, attention, score, options in [
            ('static', 'agfl', .80, {}),
            ('conditioned', 'agfl', .82, {'coefficient_conditioning': 'trial_power'}),
            ('baseline', 'linformer', .81, {'projection_rank': 4}),
        ]:
            record = save_comparison(source, name, attention, seed, score,
                                      options=options, split='different' if name == 'baseline' and seed == 2 else None)
            ids[name] = record['experiment_id']
    original_bytes = {p: p.read_bytes() for p in source.rglob('result.json')}
    result = analyze_results(source, output)
    assert result['n_runs'] == 9
    assert len(result['experiments']) == 3
    assert all(row['n_seeds'] == 3 for row in result['experiments'])
    assert len(result['comparisons']) == 9  # Three metrics for three directed contrasts.
    expected_pairs = {(ids['conditioned'], ids['static']): (3, .02, 'coefficient_conditioning'),
                      (ids['static'], ids['baseline']): (2, -.01, 'attention'),
                      (ids['conditioned'], ids['baseline']): (2, .01, 'attention')}
    for row in result['comparisons']:
        count, delta, kind = expected_pairs[(row['agfl_experiment_id'], row['baseline_experiment_id'])]
        assert (row['n_pairs'], row['comparison_kind']) == (count, kind)
        assert row['mean_delta'] == pytest.approx(delta)
        assert row['agfl_label'] != row['baseline_label']
    assert {row['attention_label'] for row in result['across_subjects']} == {
        'AGFL (static)', 'AGFL (trial-conditioned, scale 0.5)', 'Linformer (rank 4)'}
    report = (output / 'report.md').read_text()
    assert all(label in report for label in ('AGFL (static)', 'AGFL (trial-conditioned', 'Linformer (rank 4)'))
    assert all(path.read_bytes() == before for path, before in original_bytes.items())
    assert json.loads((output / 'aggregation.json').read_text())['n_runs'] == 9


def test_different_graph_settings_are_not_labeled_as_a_conditioning_effect(tmp_path):
    for seed in range(2):
        save_comparison(tmp_path / 'runs', 'static', 'agfl', seed, .8, options={'top_k': 3})
        save_comparison(tmp_path / 'runs', 'conditioned', 'agfl', seed, .9,
                        options={'top_k': 5, 'coefficient_conditioning': 'trial_power'})
    result = analyze_results(tmp_path / 'runs', tmp_path / 'report')
    assert result['comparisons'] == []


def test_conditioning_contrast_rejects_different_training_settings(tmp_path):
    save_comparison(tmp_path / 'runs', 'static', 'agfl', 0, .8)
    config = resolve_config({'model': 'eegnet', 'attention': 'agfl', 'dataset': 'synthetic_eeg',
                             'attention_options': {'coefficient_conditioning': 'trial_power'}})
    config['training']['epochs'] = 100
    save_run(tmp_path / 'runs' / 'conditioned', 'agfl', 0, .9,
             attention_options=config['attention_options'], training=config['training'],
             subject_id='A03', data={'subjects': [3]})
    assert analyze_results(tmp_path / 'runs', tmp_path / 'report')['comparisons'] == []


def test_saved_options_drive_readable_plot_labels():
    config = resolve_config({'model': 'eegnet', 'attention': 'agfl',
                             'attention_options': {'coefficient_conditioning': 'trial_power'}})
    config['subject_id'] = 'A03'
    assert selection_label({'config': config}) == 'eegnet / AGFL (trial-conditioned, scale 0.5) / electrode graph / A03'
    assert attention_display_name('agfl', {}) == 'AGFL (static)'
    assert attention_display_name('linformer', {'projection_rank': 4}) == 'Linformer (rank 4)'
