"""All launch presets must resolve before any dataset is loaded."""
import json
import pytest
from agfl.cli import main
from agfl.config import merge, resolve_config, resolve_experiments
from agfl.presets import load_preset, preset_names


@pytest.mark.parametrize('name', preset_names())
def test_presets_resolve_with_five_seeds(name):
    document = load_preset(name)
    configs = [merge(document['base'], variant) for variant in document['experiments']] if 'base' in document else [document]
    for config in configs:
        resolved = resolve_config(config)
        assert resolved['seeds'] == [0, 1, 2, 3, 4]
        assert resolved['model'] != 'none'
        assert resolved['model_variant'] in {'eeg', 'ecg'}


def test_no_arguments_suggests_a_launch_without_running(capsys):
    main([])
    assert 'run --preset eeg' in capsys.readouterr().out


def test_plan_resolves_comparison_without_loading_datasets(monkeypatch, capsys):
    import agfl.datasets
    def forbidden(*args, **kwargs):
        raise AssertionError('A plan must not load data')
    monkeypatch.setattr(agfl.datasets, 'prepare_data', forbidden)
    main(['plan', '--preset', 'eeg-comparison'])
    output = capsys.readouterr().out
    assert 'sweep --preset eeg-comparison' in output
    assert '"attention": "nystromformer"' in output


def test_plan_selects_model_and_attention_independently(capsys):
    main(['plan', '--preset', 'eeg', '--model', 'eegencoder', '--attention', 'mha'])
    output = capsys.readouterr().out
    assert '"model": "eegencoder"' in output
    assert '"attention": "mha"' in output
    assert '"attention_options"' in output


def test_comparison_keeps_selected_backbone_for_all_five_attentions(capsys):
    main(['plan', '--preset', 'eeg-comparison', '--model', 'conformer'])
    configurations = json.loads(capsys.readouterr().out.split('\nLaunch on')[0])
    assert {config['model'] for config in configurations} == {'conformer'}
    assert {config['attention'] for config in configurations} == {'agfl', 'mha', 'performer', 'linformer', 'nystromformer'}


@pytest.mark.parametrize('key', ['agfl', 'performer', 'transformer', 'legacy_eegnet'])
def test_attention_and_removed_adapter_keys_cannot_select_a_backbone(key):
    with pytest.raises(ValueError):
        resolve_config({'model': key})


def test_model_settings_cannot_hide_an_attention_selection():
    with pytest.raises(ValueError, match='Unknown options'):
        resolve_config({'model': 'eegnet', 'model_options': {'attention_type': 'standard'}})
    with pytest.raises(ValueError, match='Unknown options'):
        resolve_config({'attention': 'mha', 'attention_options': {'K': 2}})


def test_duplicate_sweep_after_attention_override_runs_each_configuration_once(capsys):
    main(['plan', '--preset', 'eeg-comparison', '--attention', 'agfl'])
    configurations = json.loads(capsys.readouterr().out.split('\nLaunch on')[0])
    assert len(configurations) == 9
    assert {row['subject_id'] for row in configurations} == {f'A{s:02d}' for s in range(1, 10)}
    assert all(row['attention'] == 'agfl' for row in configurations)


def test_subjects_expand_without_mutating_config_or_loading_data():
    original = {'dataset': 'eeg', 'data': {'subjects': [3, 1]}, 'seeds': [7]}
    experiments = resolve_experiments(original)
    assert original['data']['subjects'] == [3, 1]
    assert [r['data']['subjects'] for r in experiments] == [[1], [3]]
    assert [r['subject_id'] for r in experiments] == ['A01', 'A03']
    assert all(r['split']['protocol'] == 'stratified' and r['seeds'] == [7] for r in experiments)
    assert resolve_experiments(experiments[0]) == [experiments[0]]


def test_individual_subjects_reject_pooled_protocol_and_filter_leakage():
    with pytest.raises(ValueError, match='individually'):
        resolve_experiments({'dataset': 'eeg', 'split': {'protocol': 'group'}})
    with pytest.raises(ValueError, match='held-out trials'):
        resolve_experiments({'dataset': 'eeg', 'data': {'filter_scope': 'run'}})
    with pytest.raises(ValueError, match='subject_id'):
        resolve_experiments({'dataset': 'eeg', 'subject_id': 'A02', 'data': {'subjects': [1]}})


def test_comparison_expands_all_subject_attention_pairs(capsys):
    main(['plan', '--preset', 'eeg-comparison'])
    configurations = json.loads(capsys.readouterr().out.split('\nLaunch on')[0])
    assert len(configurations) == 45
    assert len({(r['subject_id'], r['attention']) for r in configurations}) == 45
    assert all(len(r['data']['subjects']) == 1 and len(r['seeds']) == 5 for r in configurations)


@pytest.mark.parametrize('key', ['eegnet', 'eegencoder', 'dstseegencoder', 'conformer'])
def test_ablation_matrix_resolves_once_per_backbone_default(key, capsys):
    main(['plan', '--preset', 'eeg-ablations', '--model', key])
    configurations = json.loads(capsys.readouterr().out.split('\nLaunch on')[0])
    assert {config['model'] for config in configurations} == {key}
    assert len({json.dumps(config, sort_keys=True) for config in configurations}) == len(configurations)
    assert {config['attention_options']['K'] for config in configurations} >= {0, 1, 2, 3}
