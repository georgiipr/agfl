"""Target-machine reporting checks for token routing and its matched controls."""
from copy import deepcopy

import pytest

from agfl.analysis import analyze_results, attention_display_name
from agfl.config import resolve_config
from agfl.visualization.results import selection_label
from tests.test_analysis import save_run
from tests.test_eegnet_conditioning_comparison import save_comparison


def test_token_and_old_trial_effects_are_separate_and_only_matched_pairs_enter(tmp_path):
    source, output = tmp_path / 'runs', tmp_path / 'analysis'
    ids = {}
    for seed in range(3):
        for name, attention, score, options in [
            ('static', 'agfl', .80, {}),
            ('token', 'agfl', .83, {'coefficient_conditioning': 'token_contrast'}),
            ('trial', 'agfl', .82, {'coefficient_conditioning': 'trial_power'}),
            ('mha', 'mha', .79, {}),
            ('linformer', 'linformer', .81, {'projection_rank': 4}),
        ]:
            record = save_comparison(
                source, name, attention, seed, score, options=options,
                split='mismatched-split' if name == 'mha' and seed == 2 else None)
            ids[name] = record['experiment_id']
    unchanged = {path: path.read_bytes() for path in source.rglob('result.json')}
    result = analyze_results(source, output)
    assert result['n_runs'] == 15 and len(result['experiments']) == 5
    expected = {
        (ids['token'], ids['static']): (3, .03, 'coefficient_conditioning'),
        (ids['trial'], ids['static']): (3, .02, 'coefficient_conditioning'),
        (ids['static'], ids['mha']): (2, .01, 'attention'),
        (ids['token'], ids['mha']): (2, .04, 'attention'),
        (ids['trial'], ids['mha']): (2, .03, 'attention'),
        (ids['static'], ids['linformer']): (3, -.01, 'attention'),
        (ids['token'], ids['linformer']): (3, .02, 'attention'),
        (ids['trial'], ids['linformer']): (3, .01, 'attention'),
    }
    assert len(result['comparisons']) == len(expected) * 3
    for row in result['comparisons']:
        count, delta, kind = expected[row['agfl_experiment_id'], row['baseline_experiment_id']]
        assert row['n_pairs'] == count
        assert row['mean_delta'] == pytest.approx(delta)
        assert row['comparison_kind'] == kind
        assert row['agfl_label'] != row['baseline_label']
    assert all(row['n_seeds'] == 3 for row in result['experiments'])
    labels = {row['attention_label'] for row in result['experiments']}
    assert labels == {'AGFL (static)', 'AGFL (token routing, scale 0.5)',
                      'AGFL (trial-conditioned, scale 0.5)', 'MHA', 'Linformer (rank 4)'}
    report = (output / 'report.md').read_text()
    assert all(label in report for label in labels)
    assert all(path.read_bytes() == content for path, content in unchanged.items())


@pytest.mark.parametrize('static_options,token_options', [
    ({'top_k': 3}, {'top_k': 5}),
    ({'K': 1}, {'K': 2}),
    ({'projection': 'split_input'}, {'projection': 'qkv'}),
    ({'filter_projection': 'none'}, {'filter_projection': 'separate'}),
    ({'graph_normalization': 'softmax'}, {'graph_normalization': 'row'}),
])
def test_attention_mismatches_cannot_be_reported_as_a_token_routing_effect(
        tmp_path, static_options, token_options):
    for seed in range(2):
        save_comparison(tmp_path / 'runs', 'static', 'agfl', seed, .8,
                        options=static_options)
        save_comparison(tmp_path / 'runs', 'token', 'agfl', seed, .9,
                        options={**token_options, 'coefficient_conditioning': 'token_contrast'})
    assert analyze_results(tmp_path / 'runs', tmp_path / 'analysis')['comparisons'] == []


@pytest.mark.parametrize('change', ['training', 'backbone', 'data', 'source'])
def test_protocol_changes_exclude_token_static_contrasts(tmp_path, change):
    control = save_comparison(tmp_path / 'runs', 'static', 'agfl', 0, .8)
    config = deepcopy(control['config'])
    config['attention_options']['coefficient_conditioning'] = 'token_contrast'
    if change == 'training':
        config['training']['learning_rate'] *= 2
    elif change == 'backbone':
        config['model_options']['temp_kernel'] *= 2
    elif change == 'data':
        config['data']['window'] = 900
    else:
        config['provenance'] = {'source_sha256': 'different-source'}
    save_run(tmp_path / 'runs' / 'token', 'agfl', 0, .9,
             **{key: config[key] for key in (
                 'attention_options', 'model_options', 'training', 'data', 'subject_id')},
             **({'provenance': config['provenance']} if 'provenance' in config else {}))
    assert analyze_results(tmp_path / 'runs', tmp_path / 'analysis')['comparisons'] == []


def test_token_plot_labels_are_explicit_and_old_labels_do_not_change():
    config = resolve_config({'model': 'eegnet', 'attention': 'agfl',
                             'attention_options': {'coefficient_conditioning': 'token_contrast'}})
    config['subject_id'] = 'A03'
    assert selection_label({'config': config}) == 'eegnet / AGFL (token routing, scale 0.5) / A03'
    assert attention_display_name('agfl', {}) == 'AGFL (static)'
    assert attention_display_name('agfl', {'coefficient_conditioning': 'trial_power'}) == 'AGFL (trial-conditioned, scale 0.5)'
    assert attention_display_name('agfl', {'coefficient_conditioning': 'token_contrast',
                                         'coefficient_conditioning_scale': .25}) == 'AGFL (token routing, scale 0.25)'
