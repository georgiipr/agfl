"""Matched comparison/reporting contracts for the optional feature router."""
from copy import deepcopy

import pytest

from agfl.analysis import analyze_results, attention_display_name, conditioning_control
from agfl.config import comparison_identity, experiment_identity, merge, resolve_experiments
from tests.references.presets import load_archived_preset as load_preset, archived_experiments
from tests.test_eegnet_conditioning_comparison import save_comparison


def test_feature_comparison_retains_all_previous_controls_and_training():
    old, current = load_preset('eegnet-a03-token-agfl'), load_preset('eegnet-a03-feature-agfl')
    assert current['base'] == {**old['base'], 'output_dir': 'results/eegnet-A03-feature-agfl-v1'}
    assert current['experiments'][:2] + current['experiments'][3:] == old['experiments']
    expected = deepcopy(old['experiments'][1])
    expected['attention_options']['coefficient_conditioning'] = 'feature_contrast'
    assert current['experiments'][2] == expected
    configs = [archived_experiments(merge(current['base'], e))[0] for e in current['experiments']]
    assert len({comparison_identity(c) for c in configs}) == 1
    assert len({experiment_identity(c) for c in configs}) == 5
    assert sum(len(c['seeds']) for c in configs) == 25
    assert {c['subject_id'] for c in configs} == {'A03'}


def test_feature_effect_uses_matching_token_static_and_baseline_controls(tmp_path):
    ids = {}
    for seed in range(3):
        for name, attention, score, mode in [
            ('static', 'agfl', .80, 'static'), ('token', 'agfl', .82, 'token_contrast'),
            ('feature', 'agfl', .83, 'feature_contrast'), ('mha', 'mha', .81, None),
            ('linformer', 'linformer', .84, None),
        ]:
            options = ({'coefficient_conditioning': mode} if mode else
                       {'projection_rank': 4} if attention == 'linformer' else {})
            record = save_comparison(tmp_path / 'runs', name, attention, seed, score, options=options,
                                      split='different-split' if name == 'token' and seed == 2 else None)
            ids[name] = record['experiment_id']
    result = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    rows = [r for r in result['comparisons'] if r['agfl_experiment_id'] == ids['feature']]
    expected = {ids['static']: (3, .03), ids['token']: (2, .01),
                ids['mha']: (3, .02), ids['linformer']: (3, -.01)}
    assert len(rows) == 12  # Four controls, three metrics; no reverse or self pairs.
    for row in rows:
        count, delta = expected[row['baseline_experiment_id']]
        assert row['n_pairs'] == count and row['mean_delta'] == pytest.approx(delta)
        assert row['agfl_label'] == 'AGFL (feature routing, scale 0.5)'
    assert len(result['experiments']) == 5
    assert attention_display_name('agfl', {'coefficient_conditioning': 'token_contrast'}) != rows[0]['agfl_label']


@pytest.mark.parametrize('mismatch', [
    {'coefficient_conditioning_scale': .25}, {'top_k': 3}, {'K': 3},
    {'coefficient_conditioning': 'trial_power'}, {'coefficient_conditioning': 'feature_contrast'},
])
def test_feature_to_token_pair_rejects_unmatched_options(mismatch):
    token = {'coefficient_conditioning': 'token_contrast', 'coefficient_conditioning_scale': .5, 'top_k': 5, 'K': 2}
    feature = {**token, 'coefficient_conditioning': 'feature_contrast'}
    assert conditioning_control({'attention_options': feature}, {'attention_options': token})
    assert not conditioning_control({'attention_options': feature}, {'attention_options': {**token, **mismatch}})
