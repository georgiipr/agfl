"""Scientific plotting invariants; execute on the target machine only."""
import copy
import json

import numpy as np
import pytest
import torch

from agfl.analysis import analyze_results
from agfl.config import comparison_identity, experiment_identity, resolve_config
from agfl.datasets import load_dataset, get_split
from agfl.metrics import classification_metrics
from agfl.models import get_model_spec
from agfl.storage import write_json
from agfl.visualization.diagnostics import diagnose_run, selected_indices
from agfl.visualization.maps import mixer_map, map_statistics
from agfl.visualization.results import generate_result_plots, paired_differences, read_predictions
from tests.model_fixtures import small_model_options


def saved_fixture(tmp_path, model_key='signal_transformer', attention='agfl', axis=None):
    directory = tmp_path / 'runs' / 'seed_0'
    directory.mkdir(parents=True)
    config = resolve_config({
        'model': model_key, 'attention': attention, 'dataset': 'synthetic_eeg', 'device': 'cpu', 'seeds': [0],
        'data': {'subjects': 3, 'samples_per_subject': 8, 'channels': 4, 'samples': 65},
        'model_options': {**small_model_options(model_key, 'eeg'), **({'attention_axis': axis} if axis else {})},
        'attention_options': {'heads': 2},
        'output_dir': str(tmp_path / 'runs'), 'split_dir': str(tmp_path / 'splits'),
    })
    bundle = load_dataset(config['dataset'], config['data'])
    split = get_split(bundle, config['split'], 0, config['split_dir'])
    config.update(dataset_fingerprint=bundle.fingerprint, expected_split_id=split['split_id'],
                  resolved_metadata=bundle.metadata, provenance={'source_sha256': 'fixture'})
    torch.manual_seed(13)
    model = get_model_spec(config['model']).build(config['model_options'], bundle.metadata, config['attention'], config['attention_options']).eval()
    predictions, metrics = {}, {}
    with torch.no_grad():
        for partition in ('validation', 'test'):
            indices = split[partition]
            targets = bundle.y[indices]
            probabilities = model(torch.from_numpy(bundle.x[indices])).softmax(-1).numpy()
            predictions.update({partition + '_targets': targets, partition + '_probabilities': probabilities,
                                partition + '_ids': np.asarray(split['sample_ids'][partition])})
            metrics[partition] = {**classification_metrics(targets, probabilities), 'loss': 1.}
    np.savez_compressed(directory / 'predictions.npz', **predictions)
    torch.save({'model': model.state_dict(), 'config': config, 'split_id': split['split_id'],
                'dataset_fingerprint': bundle.fingerprint, 'normalization': None, 'epoch': 2}, directory / 'checkpoint.pt')
    record = {'schema_version': 2, 'status': 'completed', 'dataset': config['dataset'], 'model': model_key, 'attention': attention, 'model_variant': 'eeg',
              'seed': 0, 'split_id': split['split_id'], 'dataset_fingerprint': bundle.fingerprint,
              'comparison_id': comparison_identity(config), 'experiment_id': experiment_identity(config),
              'parameter_count': sum(p.numel() for p in model.parameters()), 'config': config,
              'best_checkpoint_epoch': 2, **metrics}
    write_json(directory / 'result.json', record)
    write_json(directory / 'config.json', config)
    write_json(directory / 'split.json', split)
    write_json(directory / 'history.json', [{'epoch': i, 'train_loss': 1.1 / i,
                                           'validation': metrics['validation'], 'learning_rate': .001 / i} for i in (1, 2)])
    return directory, {**record, 'source': str(directory / 'result.json')}


def test_predictions_must_match_saved_split_order(tmp_path):
    directory, run = saved_fixture(tmp_path)
    read_predictions(run, 'test')
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        values = {name: saved[name] for name in saved.files}
    values['test_ids'] = values['test_ids'][::-1]
    np.savez_compressed(directory / 'predictions.npz', **values)
    with pytest.raises(ValueError, match='sample order'):
        read_predictions(run, 'test')


def test_paired_plot_excludes_mismatched_splits_and_undefined_metrics():
    a = [{'seed': i, 'split_id': str(i), 'dataset_fingerprint': 'data', 'comparison_id': 'protocol',
          'test': {'accuracy': .8}} for i in range(3)]
    b = copy.deepcopy(a)
    b[0]['test']['accuracy'] = .7
    b[1]['split_id'] = 'different'
    b[2]['test']['accuracy'] = None
    assert paired_differences(a, b, 'accuracy') == [(0, pytest.approx(.1))]


def test_diagnostic_sampling_is_repeatable_and_keeps_partition_membership():
    labels = np.tile(np.arange(4), 12)
    selected = selected_indices(labels, np.arange(8, 40), 12, 42)
    np.testing.assert_array_equal(selected, selected_indices(labels, np.arange(8, 40), 12, 42))
    assert set(selected) <= set(range(8, 40))
    assert np.bincount(labels[selected]).tolist() == [3, 3, 3, 3]


@pytest.mark.parametrize('key', ['mha', 'performer', 'linformer', 'nystromformer'])
def test_effective_diagnostic_map_recovers_mixer_output(key):
    from agfl.attention import get_attention_spec
    spec = get_attention_spec(key)
    options = {**spec.defaults, 'heads': 2}
    if 'landmarks' in options:
        options['landmarks'] = 3
    if 'projection_rank' in options:
        options['projection_rank'] = 3
    torch.manual_seed(29)
    mixer = spec.build(options, 8, 7).double()
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    with torch.no_grad():
        expected = mixer(x)
        matrix, _ = mixer_map(mixer, x)
        _, _, values = mixer.project(x)
        actual = mixer.combine(matrix @ values)
    torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)


def test_signed_or_unnormalized_maps_do_not_get_probability_entropy():
    signed = np.array([[[[1.2, -.2], [.4, .6]]]])
    assert map_statistics(signed)['entropy_per_head'] == [None]
    assert map_statistics(np.ones((1, 1, 2, 2)))['entropy_per_head'] == [None]
    assert map_statistics(np.full((1, 1, 2, 2), .5))['entropy_per_head'][0] == pytest.approx(np.log(2))


def test_result_figures_need_no_dataset_or_checkpoint_execution(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    directory, _ = saved_fixture(tmp_path)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    import agfl.datasets
    import agfl.models
    def forbidden(*args, **kwargs):
        raise AssertionError('Result plotting must only read saved artifacts')
    monkeypatch.setattr(agfl.datasets, 'load_dataset', forbidden)
    monkeypatch.setattr(agfl.models, 'get_model_spec', forbidden)
    aggregation = analyze_results(tmp_path / 'runs', tmp_path / 'analysis')
    manifest = generate_result_plots(aggregation, tmp_path / 'figures')
    assert manifest['figures']
    for entry in manifest['figures']:
        png, pdf = [tmp_path / 'figures' / name for name in entry['files']]
        assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
        assert pdf.read_bytes().startswith(b'%PDF')
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before


@pytest.mark.parametrize('model_key,attention,axis', [
    ('signal_transformer', 'agfl', None), ('eegnet', 'agfl', None),
    ('eegencoder', 'mha', None), ('dstseegencoder', 'linformer', None),
    ('conformer', 'performer', None), ('eegnet', 'nystromformer', 'time'),
])
def test_checkpoint_diagnostics_preserve_run_and_use_selected_partition(tmp_path, monkeypatch, model_key, attention, axis):
    pytest.importorskip('matplotlib')
    directory, _ = saved_fixture(tmp_path, model_key, attention, axis)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    import agfl.engine
    def forbidden(*args, **kwargs):
        raise AssertionError('Diagnostics must not train')
    monkeypatch.setattr(agfl.engine, 'run_training', forbidden)
    manifest = diagnose_run(directory, tmp_path / 'diagnostics', max_samples=6, embedding='pca')
    split = json.loads((directory / 'split.json').read_text())
    assert set(manifest['sample_ids']) <= set(split['sample_ids']['validation'])
    assert not set(manifest['sample_ids']) & set(manifest['csp_training_sample_ids'])
    assert manifest['prediction_check']['matches']
    assert manifest['model'] == model_key and manifest['attention'] == attention
    statistics = json.loads((tmp_path / 'diagnostics' / 'mixer_statistics.json').read_text())
    assert statistics
    assert {row['token_axis'] for row in statistics.values()} == {axis or 'electrode'}
    assert manifest['figures']
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
