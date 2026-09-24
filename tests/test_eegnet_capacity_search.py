"""Target-machine checks for the opt-in A03 capacity study."""
from copy import deepcopy

import pytest

from agfl.config import comparison_identity, experiment_identity
from agfl.models.eegnet.search import CANDIDATE_SETS, DEFAULT_CANDIDATES, search_configs


RECIPES = {
    'spatial_qkv_capacity_focal_005': ('focal', .005),
    'spatial_qkv_capacity_focal_001': ('focal', .001),
    'spatial_qkv_capacity_ce_005': ('cross_entropy', .005),
    'spatial_qkv_capacity_ce_001': ('cross_entropy', .001),
}


@pytest.mark.parametrize('name', RECIPES)
def test_capacity_recipe_changes_only_declared_architecture_and_training(name, tmp_path):
    configs = search_configs(tmp_path / 'ml', tmp_path / 'out', [3], list(range(5)),
                             ['spatial_qkv_sparse', name], epochs=250)
    control, actual = configs['spatial_qkv_sparse'][0], configs[name][0]
    expected = deepcopy(control)
    expected['model_options']['pk2'] = 8
    expected['attention_options'].update(filter_projection='separate',
                                         filter_projection_init='identity')
    loss, learning_rate = RECIPES[name]
    expected['training'].update(loss=loss, learning_rate=learning_rate)
    # Catch accidental changes to trial partitions, preprocessing, augmentation,
    # score scaling, graph order, checkpoint selection, or the epoch budget.
    assert actual == expected
    assert actual['data'] == control['data']
    assert actual['split'] == control['split']
    assert actual['training']['checkpoint_criterion'] == 'accuracy'
    assert actual['attention_options']['score_scaling'] == 'sqrt_dim'
    assert actual['attention_options']['coefficient_init'] == 'one_hop'
    assert actual['attention_options']['top_k'] == 'scheduled'
    assert actual['attention_options']['K'] == 2
    assert 1000 // actual['model_options']['pk1'] // actual['model_options']['pk2'] == 15
    assert 1000 // control['model_options']['pk1'] // control['model_options']['pk2'] == 7
    assert experiment_identity(actual) != experiment_identity(control)
    # Capacity and training differ: old fixed-backbone attention comparisons
    # must not silently pool these new recipes with their historical control.
    assert comparison_identity(actual) != comparison_identity(control)


def test_capacity_group_is_bounded_with_unchanged_control_first(tmp_path):
    assert CANDIDATE_SETS['capacity'] == ('spatial_qkv_sparse', *RECIPES)
    configs = search_configs(tmp_path / 'ml', tmp_path / 'out', [3], list(range(5)),
                             CANDIDATE_SETS['capacity'], epochs=250)
    assert sum(len(config['seeds']) for variants in configs.values() for config in variants) == 25
    for variants in configs.values():
        assert len(variants) == 1
        config = variants[0]
        assert config['subject_id'] == 'A03' and config['data']['subjects'] == [3]
        assert config['data']['sessions'] == ['T']
        assert config['seeds'] == [0, 1, 2, 3, 4]
        assert config['training']['epochs'] == 250
        assert (config['model'], config['attention']) == ('eegnet', 'agfl')
    control = configs['spatial_qkv_sparse'][0]
    assert control['attention_options']['filter_projection'] == 'none'
    assert control['model_options']['pk2'] == 16
    assert control['training']['loss'] == 'focal'
    assert control['training']['learning_rate'] == .005
    assert not (tmp_path / 'out').exists()


def test_capacity_candidates_do_not_expand_original_default_study(tmp_path):
    assert DEFAULT_CANDIDATES == ('spatial_control', 'compact_train_channel',
                                  'compact_trialnorm', 'compact_recombine',
                                  'compact_recombine_slow')
    defaults = search_configs(tmp_path / 'ml', tmp_path / 'out', [3], [0])
    assert tuple(defaults) == DEFAULT_CANDIDATES
    assert set(CANDIDATE_SETS['capacity']).isdisjoint(defaults)
