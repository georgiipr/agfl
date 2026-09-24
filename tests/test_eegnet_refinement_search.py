"""Target-machine checks for the bounded A03 graph/filter refinement plan."""
from copy import deepcopy
import json

import pytest

from agfl.cli import main
from agfl.config import comparison_identity, experiment_identity
from agfl.models.eegnet.search import CANDIDATES, DEFAULT_CANDIDATES, search_configs


CONTROL = 'spatial_qkv_sparse'
REFINEMENTS = {
    'spatial_qkv_temp1': (1., 'one_hop'),
    'spatial_qkv_temp05': (.5, 'one_hop'),
    'spatial_qkv_identity_temp1': (1., 'identity_one_hop'),
    'spatial_qkv_identity_temp05': (.5, 'identity_one_hop'),
}


@pytest.mark.parametrize('candidate', REFINEMENTS)
def test_refinement_changes_only_declared_attention_settings(candidate, tmp_path):
    configs = search_configs(tmp_path, tmp_path / 'out', [3], list(range(5)),
                             [CONTROL, candidate], epochs=250)
    control = configs[CONTROL][0]
    actual = configs[candidate][0]
    temperature, coefficient_init = REFINEMENTS[candidate]
    expected = deepcopy(control)
    expected['attention_options'].update(
        score_scaling='temperature', temperature_init=temperature,
        coefficient_init=coefficient_init,
    )
    # Whole-configuration equality catches hidden changes to QKV, graph pruning,
    # polynomial order, normalization, model, loss, augmentation or trial splits.
    assert actual == expected
    assert control['attention_options']['score_scaling'] == 'sqrt_dim'
    assert control['attention_options']['coefficient_init'] == 'one_hop'
    assert control['attention_options']['top_k'] == 'scheduled'
    assert control['attention_options']['K'] == 2
    # Distinct configurations remain comparable under shared source and training.
    assert experiment_identity(actual) != experiment_identity(control)
    assert comparison_identity(actual) == comparison_identity(control)


def test_refinement_preview_is_bounded_and_does_not_execute(monkeypatch, capsys, tmp_path):
    import agfl.datasets
    import agfl.models.eegnet.search

    def forbidden(*args, **kwargs):
        raise AssertionError('A search preview loaded data or launched training')

    monkeypatch.setattr(agfl.datasets, 'load_dataset', forbidden)
    monkeypatch.setattr(agfl.models.eegnet.search, 'run_search', forbidden)
    candidates = [CONTROL, *REFINEMENTS]
    argv = ['tune-eegnet', '--dry-run', '--data-dir', str(tmp_path / 'ml'),
            '--output-dir', str(tmp_path / 'eegnet-A03-agfl-refinement-v1'),
            '--subjects', '3', '--seeds', '0', '1', '2', '3', '4', '--epochs', '250']
    for candidate in candidates:
        argv.extend(['--candidate', candidate])
    main(argv)
    configs = json.loads(capsys.readouterr().out)
    assert list(configs) == candidates  # The unchanged control wins exact ties.
    assert sum(len(cfg['seeds']) for variants in configs.values() for cfg in variants) == 25
    for variants in configs.values():
        assert len(variants) == 1
        cfg = variants[0]
        assert (cfg['model'], cfg['attention'], cfg['dataset']) == ('eegnet', 'agfl', 'eeg')
        assert cfg['subject_id'] == 'A03' and cfg['data']['subjects'] == [3]
        assert cfg['data']['sessions'] == ['T']
        assert cfg['seeds'] == [0, 1, 2, 3, 4]
        assert cfg['training']['epochs'] == 250
        assert cfg['model_options']['attention_axis'] == 'time'
        assert cfg['split'] == {'protocol': 'stratified', 'train': .6, 'validation': .2, 'test': .2}
    assert not (tmp_path / 'eegnet-A03-agfl-refinement-v1').exists()


def test_refinement_does_not_expand_defaults_or_change_existing_qkv_candidates(tmp_path):
    original_names = ('spatial_control', 'compact_train_channel', 'compact_trialnorm',
                      'compact_recombine', 'compact_recombine_slow')
    assert DEFAULT_CANDIDATES == original_names
    defaults = search_configs(tmp_path, tmp_path / 'default', [3], [0])
    assert tuple(defaults) == original_names
    assert set(REFINEMENTS).isdisjoint(defaults)
    for name in ('spatial_qkv_dense', CONTROL, 'spatial_qkv_renorm', 'spatial_qkv_top3'):
        options = CANDIDATES[name]['attention_options']
        assert options['score_scaling'] == 'sqrt_dim'
        assert options['coefficient_init'] == 'one_hop'
        assert 'temperature_init' not in options
