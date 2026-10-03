"""Every packaged preset and the command line resolve without loading data.

Development checks for the experiment machine only; they are not a prerequisite
for training. Presets are resolved through the path the CLI uses (merge the
``base`` with each experiment, then ``resolve_experiments``), so the counts
below are the numbers of configurations a launch would train, each with its
declared seeds.

Run from the project folder: ``python -m pytest tests/test_presets.py -q``.
"""
from itertools import product
import json
import os

import pytest

from agfl_speech.cli import main
from agfl_speech.config import digest, merge, resolve_config, resolve_experiments
from agfl_speech.multi_subject_study import validate_study_config
from agfl_speech.presets import load_preset, preset_names


MODELS = ('eegnet', 'eegnet_original', 'signal_transformer')
ATTENTIONS = ('agfl', 'mha', 'hcann')
SUBJECT_IDS = [f'S{subject:02d}' for subject in range(1, 8)]
# Resolved configurations per preset.
EXPECTED_CONFIGS = {
    'si-hom': 1,
    'si-hom-attentions': 9,
    'si-hom-confirmatory': 8,
    'si-hom-authors-recipe': 3,
    'si-hom-eegnet': 3,
    'si-hom-eegnet-original': 3,
    'si-hom-individual': 21,
    'si-hom-signal-transformer': 3,
}
# The confirmatory experiment uses seeds the exploratory presets never touch.
CONFIRMATORY_SEEDS = list(range(5, 25))


def preset_experiments(name):
    """Resolve a preset as the CLI does, without its deduplication."""
    document = load_preset(name)
    configs = ([merge(document['base'], experiment) for experiment in document['experiments']]
               if 'base' in document else [document])
    return [experiment for config in configs for experiment in resolve_experiments(config)]


def planned(capsys):
    """The JSON printed by ``plan`` and the launch hint after it."""
    output = capsys.readouterr().out
    document, _, hint = output.partition('\nLaunch on')
    return json.loads(document), hint


def forbid_data_loading(monkeypatch):
    import agfl_speech.datasets
    import agfl_speech.datasets.registry
    import agfl_speech.datasets.si_hom
    import agfl_speech.datasets.splits

    def forbidden(*args, **kwargs):
        raise AssertionError('Resolving a configuration must not load data')
    monkeypatch.setattr(agfl_speech.datasets, 'prepare_data', forbidden)
    monkeypatch.setattr(agfl_speech.datasets, 'load_dataset', forbidden)
    monkeypatch.setattr(agfl_speech.datasets.registry, 'load_dataset', forbidden)
    monkeypatch.setattr(agfl_speech.datasets.splits, 'prepare_data', forbidden)
    monkeypatch.setattr(agfl_speech.datasets.si_hom, 'read_archive', forbidden)


def test_packaged_presets_are_the_documented_catalogue():
    assert preset_names() == sorted(EXPECTED_CONFIGS)
    with pytest.raises(ValueError, match='Unknown preset'):
        load_preset('eeg-comparison')
    sweeps = [name for name in preset_names() if 'base' in load_preset(name)]
    assert sorted(set(preset_names()) - set(sweeps)) == ['si-hom']
    # Only the individual-subject preset declares a subject study.
    assert [name for name in sweeps if 'study' in load_preset(name)['base']] == ['si-hom-individual']


@pytest.mark.parametrize('name', sorted(EXPECTED_CONFIGS))
def test_presets_resolve_with_declared_seeds_and_electrode_attention(name, monkeypatch):
    forbid_data_loading(monkeypatch)
    experiments = preset_experiments(name)
    assert len(experiments) == EXPECTED_CONFIGS[name]
    assert len({digest(experiment) for experiment in experiments}) == len(experiments)
    for experiment in experiments:
        assert experiment['dataset'] == 'si_hom'
        assert experiment['seeds'] == (CONFIRMATORY_SEEDS if name == 'si-hom-confirmatory' else [0, 1, 2, 3, 4])
        assert experiment['model_variant'] == 'eeg'
        assert experiment['model'] in MODELS and experiment['attention'] in ATTENTIONS
        assert experiment['model_options'].get('attention_axis', 'electrode') == 'electrode'
        assert not experiment['attention_options'].get('temporal_bias', False)
        assert experiment['attention_options']['heads'] == 4
        assert experiment['device'] == 'cuda'
        # The sibling data folder is stored as written, not as an absolute path.
        assert experiment['data']['data_dir'] == '../AGFL_speech_data'
        assert experiment['data']['classes'] == list(range(8))
        assert experiment['output_dir'] == os.path.realpath(os.path.join('results', name))
        # A resolved configuration replays as itself.
        assert resolve_experiments(experiment) == [experiment]


@pytest.mark.parametrize('name,model', [('si-hom-eegnet', 'eegnet'), ('si-hom-eegnet-original', 'eegnet_original'),
                                        ('si-hom-signal-transformer', 'signal_transformer'),
                                        ('si-hom-authors-recipe', 'eegnet_original')])
def test_single_backbone_presets_compare_the_three_attentions_on_the_pooled_cohort(name, model):
    experiments = preset_experiments(name)
    assert [experiment['attention'] for experiment in experiments] == list(ATTENTIONS)
    assert {experiment['model'] for experiment in experiments} == {model}
    assert all(experiment['subject_id'] is None and experiment['data']['cohort'] == 'pooled'
               and experiment['data']['subjects'] == [1, 2, 3, 4, 5, 6, 7] for experiment in experiments)
    # A resolved split holds only the settings its protocol uses.
    split = ({'protocol': 'authors', 'validation_within_train': 0.2} if name == 'si-hom-authors-recipe'
             else {'protocol': 'stratified', 'train': 0.6, 'validation': 0.2, 'test': 0.2})
    assert all(experiment['split'] == split for experiment in experiments)
    # Everything except the attention is shared, so the arms are comparable.
    for key in ('data', 'split', 'training', 'model_options', 'seeds', 'output_dir'):
        assert all(experiment[key] == experiments[0][key] for experiment in experiments), key
    if model == 'eegnet':
        assert experiments[0]['model_options']['electrode_architecture'] == 'pre_spatial'


def test_attention_preset_is_the_full_backbone_attention_matrix():
    experiments = preset_experiments('si-hom-attentions')
    assert [(experiment['model'], experiment['attention']) for experiment in experiments] \
        == list(product(MODELS, ATTENTIONS))
    assert all(experiment['subject_id'] is None and experiment['data']['cohort'] == 'pooled'
               for experiment in experiments)
    assert {experiment['training']['epochs'] for experiment in experiments} == {200}
    for key in ('data', 'split', 'training', 'seeds', 'output_dir'):
        assert all(experiment[key] == experiments[0][key] for experiment in experiments), key


def test_confirmatory_preset_is_the_declared_eight_arms_on_new_seeds():
    experiments = preset_experiments('si-hom-confirmatory')
    arms = [(experiment['model'], experiment['attention'], experiment['training']['optimizer'])
            for experiment in experiments]
    assert arms == ([(model, attention, 'adamw') for model in ('eegnet', 'eegnet_original') for attention in ATTENTIONS]
                    + [('eegnet_original', 'mha', 'adam'), ('eegnet_original', 'hcann', 'adam')])
    exploratory = {(experiment['model'], experiment['attention']): experiment
                   for experiment in preset_experiments('si-hom-attentions')}
    for experiment in experiments[:6]:
        # The first run's arm, with only the epoch budget and the seeds changed.
        first = exploratory[experiment['model'], experiment['attention']]
        assert experiment['training'] == {**first['training'], 'epochs': 60}
        for key in ('data', 'split', 'model_options', 'attention_options'):
            assert experiment[key] == first[key], key
    for experiment in experiments[6:]:
        training = experiment['training']
        assert (training['epochs'], training['batch_size'], training['weight_decay'], training['class_weights'],
                training['scheduler']) == (50, 16, 0.0, 'none', 'none')
        assert experiment['model_options'] == experiments[4]['model_options']
    assert all(experiment['subject_id'] is None and experiment['seeds'] == CONFIRMATORY_SEEDS
               and experiment['split'] == experiments[0]['split'] and experiment['data'] == experiments[0]['data']
               for experiment in experiments)
    assert not set(CONFIRMATORY_SEEDS) & {0, 1, 2, 3, 4}


def test_individual_preset_is_three_arms_for_each_of_seven_subjects():
    experiments = preset_experiments('si-hom-individual')
    assert [(experiment['study_arm'], experiment['subject_id']) for experiment in experiments] \
        == list(product(ATTENTIONS, SUBJECT_IDS))
    for experiment in experiments:
        assert experiment['attention'] == experiment['study_arm']
        assert experiment['model'] == 'eegnet'
        assert experiment['data']['cohort'] == 'individual'
        assert experiment['data']['subjects'] == [int(experiment['subject_id'][1:])]
        assert experiment['split']['protocol'] == 'stratified'
        assert experiment['study']['subjects'] == SUBJECT_IDS
        assert experiment['study']['seeds'] == [0, 1, 2, 3, 4]
        assert experiment['study']['arms'] == list(ATTENTIONS)
        assert experiment['study']['comparisons'] == [['agfl', 'mha'], ['agfl', 'hcann']]
        validate_study_config(experiment)
    assert len({digest(experiment['study']) for experiment in experiments}) == 1


def test_single_preset_is_eegnet_with_agfl_and_backbone_defaults():
    config, = preset_experiments('si-hom')
    assert (config['model'], config['attention']) == ('eegnet', 'agfl')
    assert config['subject_id'] is None and config['data']['cohort'] == 'pooled'
    assert config['training']['epochs'] == 200
    assert config['model_options']['temp_kernel'] == 250
    assert config['model_options']['electrode_architecture'] == 'spatial_fusion'


def test_no_arguments_suggests_check_and_a_launch_without_running(capsys):
    main([])
    output = capsys.readouterr().out
    assert 'python -m agfl_speech check' in output
    assert 'python -m agfl_speech sweep --preset si-hom-attentions' in output
    assert 'python -m agfl_speech plan --preset si-hom-attentions' in output
    assert 'main.py' not in output


def test_list_and_presets_commands_print_the_kept_selections(capsys):
    main(['list'])
    assert capsys.readouterr().out.splitlines() == [
        'Models: eegnet, eegnet_original, signal_transformer',
        'Attentions: agfl, hcann, mha',
        'Datasets: si_hom']
    main(['presets'])
    assert capsys.readouterr().out.splitlines() == sorted(EXPECTED_CONFIGS)


def test_plan_resolves_a_sweep_without_loading_data(monkeypatch, capsys):
    forbid_data_loading(monkeypatch)
    main(['plan', '--preset', 'si-hom-attentions'])
    configurations, hint = planned(capsys)
    assert len(configurations) == 9
    assert {(config['model'], config['attention']) for config in configurations} == set(product(MODELS, ATTENTIONS))
    assert 'python -m agfl_speech sweep --preset si-hom-attentions' in hint
    main(['plan', '--preset', 'si-hom-individual'])
    configurations, hint = planned(capsys)
    assert len(configurations) == 21
    assert {config['subject_id'] for config in configurations} == set(SUBJECT_IDS)
    assert 'python -m agfl_speech sweep --preset si-hom-individual' in hint


@pytest.mark.parametrize('model', MODELS)
@pytest.mark.parametrize('attention', ATTENTIONS)
def test_plan_selects_model_and_attention_independently(model, attention, capsys):
    main(['plan', '--preset', 'si-hom', '--model', model, '--attention', attention])
    config, hint = planned(capsys)
    assert config['model'] == model and config['attention'] == attention
    assert config['comparison_family'] == {'eegnet': 'eegnet_v2', 'eegnet_original': 'eegnet_original_v1',
                                           'signal_transformer': 'signal_transformer_v2'}[model]
    assert config['attention_options']['heads'] == 4
    assert ('K' in config['attention_options']) == (attention == 'agfl')
    assert ('pre_norm' in config['attention_options']) == (attention == 'hcann')
    assert f'python -m agfl_speech run --preset si-hom --model {model} --attention {attention}' in hint


def test_plan_applies_seed_data_folder_and_set_overrides(monkeypatch, capsys):
    forbid_data_loading(monkeypatch)
    main(['plan', '--preset', 'si-hom', '--seeds', '7', '8', '--data-dir', 'elsewhere/si_hom',
          '--set', 'training.epochs=3', '--set', 'data.cohort=individual', '--set', 'data.subjects=[5,2]'])
    configurations, _ = planned(capsys)
    assert [config['subject_id'] for config in configurations] == ['S02', 'S05']
    for config in configurations:
        assert config['seeds'] == [7, 8]
        assert config['training']['epochs'] == 3
        assert config['data']['data_dir'] == 'elsewhere/si_hom'
        assert config['data']['cohort'] == 'individual'


def test_dry_run_prints_the_matrix_and_never_starts_training(monkeypatch, capsys):
    import agfl_speech.engine
    forbid_data_loading(monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError('A dry run must not train')
    monkeypatch.setattr(agfl_speech.engine, 'run_experiment', forbidden)
    main(['sweep', '--preset', 'si-hom-eegnet', '--dry-run'])
    configurations = json.loads(capsys.readouterr().out)
    assert [config['attention'] for config in configurations] == list(ATTENTIONS)
    main(['run', '--preset', 'si-hom', '--dry-run'])
    assert json.loads(capsys.readouterr().out)['attention'] == 'agfl'


@pytest.mark.parametrize('arguments', [
    ['tune-eegnet'], ['audit-eeg'], ['plot-eeg-audit'],
    ['plan', '--preset', 'si-hom', '--dataset', 'eeg'],
    ['plan', '--preset', 'si-hom', '--labels-dir', 'labels'],
    ['plan', '--preset', 'si-hom', '--data-path', 'signals.npz'],
    ['run', '--preset', 'si-hom-attentions', '--dry-run'],
    ['sweep', '--preset', 'si-hom', '--dry-run'],
])
def test_removed_commands_flags_and_mismatched_launch_kinds_exit_with_usage_errors(arguments, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    capsys.readouterr()


@pytest.mark.parametrize('key', ['agfl', 'mha', 'hcann', 'transformer', 'performer', 'conformer',
                                 'eegencoder', 'dstseegencoder', 'legacy_eegnet', 'none'])
def test_attention_and_removed_keys_cannot_select_a_backbone(key):
    with pytest.raises(ValueError):
        resolve_config({'model': key})


@pytest.mark.parametrize('key', ['performer', 'linformer', 'nystromformer', 'eegnet', 'none'])
def test_removed_and_backbone_keys_cannot_select_an_attention(key):
    with pytest.raises(ValueError, match='Unknown attention'):
        resolve_config({'attention': key})


def test_settings_cannot_hide_an_attention_selection_or_use_removed_options():
    for config in ({'model': 'eegnet', 'model_options': {'attention_type': 'standard'}},
                   {'model': 'eegnet', 'model_options': {'temporal_statistics': 'mean_logvar'}},
                   {'model': 'signal_transformer', 'model_options': {'ecg_patch_size': 16}},
                   {'model': 'eegnet_original', 'model_options': {'electrode_dim': 32}},
                   {'attention': 'mha', 'attention_options': {'K': 2}},
                   {'attention': 'hcann', 'attention_options': {'temporal_bias': False}},
                   {'attention': 'agfl', 'attention_options': {'pre_norm': True}}):
        with pytest.raises(ValueError, match='Unknown options'):
            resolve_config(config)
    with pytest.raises(ValueError, match='Unknown configuration keys'):
        resolve_config({'synthetic': True})
    with pytest.raises(ValueError, match='Unknown dataset'):
        resolve_config({'dataset': 'eeg'})
    with pytest.raises(ValueError, match='temporal_bias'):
        resolve_config({'attention': 'mha', 'attention_options': {'temporal_bias': True}})
    with pytest.raises(ValueError, match='across electrodes'):
        resolve_config({'model': 'eegnet', 'model_options': {'attention_axis': 'time'}})


def test_a_declared_study_is_rejected_for_the_pooled_cohort_at_resolve_time():
    document = load_preset('si-hom-individual')
    config = merge(document['base'], document['experiments'][0])
    assert len(resolve_experiments(config)) == 7
    config['data']['cohort'] = 'pooled'
    with pytest.raises(ValueError, match='cohort=individual'):
        resolve_config(config)
    with pytest.raises(ValueError, match='cohort=individual'):
        main(['plan', '--preset', 'si-hom-individual', '--set', 'data.cohort=pooled'])


def test_split_protocol_is_validated_and_canonicalised_at_resolve_time():
    for protocol in ('session', 'diagnostic_eeg', 'random'):
        with pytest.raises(ValueError, match='split.protocol'):
            resolve_config({'split': {'protocol': protocol}})
    with pytest.raises(ValueError, match='split.protocol'):
        main(['plan', '--preset', 'si-hom', '--set', 'split.protocol=session'])
    for alias, protocol in (('subject', 'group'), ('trial', 'stratified'), ('subject_dependent', 'stratified'),
                            ('stratified', 'stratified'), ('chronological', 'chronological'),
                            ('group', 'group'), ('authors', 'authors')):
        assert resolve_config({'split': {'protocol': alias}})['split']['protocol'] == protocol
    fractions = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
    assert resolve_config({})['split'] == {'protocol': 'stratified', **fractions}
    assert resolve_config({'split': {'protocol': 'chronological', 'train': 0.5, 'validation': 0.25, 'test': 0.25}})['split'] \
        == {'protocol': 'chronological', 'train': 0.5, 'validation': 0.25, 'test': 0.25}
    # The default fractions are not settings of the authors protocol.
    assert resolve_config({'split': {'protocol': 'authors'}})['split'] \
        == {'protocol': 'authors', 'validation_within_train': 0.2}
    for split, message in (({'folds': 5}, 'Unknown split settings'),
                           ({'train': 0.6, 'validation': 0.3, 'test': 0.3}, 'sum to 1'),
                           ({'protocol': 'authors', 'validation_within_train': 1.0}, 'validation_within_train')):
        with pytest.raises(ValueError, match=message):
            resolve_config({'split': split})


def test_data_folder_is_stored_as_given_while_output_folders_become_absolute(tmp_path, monkeypatch):
    assert resolve_config({})['data']['data_dir'] == '../AGFL_speech_data'
    monkeypatch.chdir(tmp_path)
    config = resolve_config({'data': {'data_dir': '../somewhere/else'},
                             'output_dir': 'results/x', 'split_dir': 'splits'})
    assert config['data']['data_dir'] == '../somewhere/else'
    assert os.path.isabs(config['output_dir']) and os.path.isabs(config['split_dir'])
    assert config['output_dir'] == os.path.realpath(os.path.join('results', 'x'))
    assert config['split_dir'] == os.path.realpath('splits')
    # Resolving again from another working directory changes nothing.
    monkeypatch.chdir(tmp_path.parent)
    assert resolve_config(config) == config
