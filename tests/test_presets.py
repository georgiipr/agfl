"""All launch presets must resolve before any dataset is loaded."""
import json
import pytest
from agfl.cli import main
from agfl.config import merge, resolve_config
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
    assert len(configurations) == 1
    assert configurations[0]['attention'] == 'agfl'


@pytest.mark.parametrize('key', ['eegnet', 'eegencoder', 'dstseegencoder', 'conformer'])
def test_ablation_matrix_resolves_once_per_backbone_default(key, capsys):
    main(['plan', '--preset', 'eeg-ablations', '--model', key])
    configurations = json.loads(capsys.readouterr().out.split('\nLaunch on')[0])
    assert {config['model'] for config in configurations} == {key}
    assert len({json.dumps(config, sort_keys=True) for config in configurations}) == len(configurations)
    assert {config['attention_options']['K'] for config in configurations} >= {0, 1, 2, 3}
