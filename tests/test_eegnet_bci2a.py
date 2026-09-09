"""Target-machine checks for the focused BCI IV-2a improvement workflow."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from agfl.cli import main
from agfl.config import resolve_config
from agfl.datasets import SignalDataset, get_split, load_dataset
from agfl.engine import Partition, run_training
from agfl.models import get_model_spec
from agfl.models.eegnet.search import search_configs, select_candidate, run_search
from agfl.presets import load_preset
from agfl.storage import write_json
from tests.test_engine import tiny_config


def record(seed, accuracy, f1=None):
    return {'status': 'validation_completed', 'seed': seed,
            'validation': {'accuracy': accuracy, 'f1': accuracy if f1 is None else f1},
            'best_checkpoint_epoch': 1}


def test_compact_eegnet_keeps_spatial_filters_and_feature_gradients():
    cfg = resolve_config(load_preset('eegnet-bci2a'))
    metadata = {'modality': 'eeg', 'channels': 22, 'samples': 1000, 'num_classes': 4}
    spec = get_model_spec('eegnet')
    torch.manual_seed(9)
    model = spec.build(cfg['model_options'], metadata, 'agfl', cfg['attention_options'])
    assert model.block2[0].kernel_size == (22, 1)
    assert model.token_axis == 'time' and model.num_tokens == 31
    assert model.fc.in_features == 16 * 31
    old = spec.build({}, metadata, 'agfl')
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in old.parameters()) / 4
    logits = model(torch.randn(4, 22, 1000))
    assert logits.shape == (4, 4)
    torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    # AGFL starts with zero filter taps; the residual must still train EEGNet.
    for layer in (model.block1[0], model.block2[0], model.fc):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert torch.count_nonzero(layer.weight.grad) > 0
    state = deepcopy(model.state_dict())
    replay = spec.build(cfg['model_options'], metadata, 'agfl', cfg['attention_options'])
    replay.load_state_dict(state)
    model.eval()
    replay.eval()
    x = torch.randn(2, 22, 1000)
    torch.testing.assert_close(model(x), replay(x), rtol=0, atol=0)


def test_search_preview_does_not_load_data_or_train(monkeypatch, capsys, tmp_path):
    import agfl.datasets
    import agfl.models.eegnet.search
    def forbidden(*args, **kwargs):
        raise AssertionError('Preview executed the experiment')
    monkeypatch.setattr(agfl.datasets, 'load_dataset', forbidden)
    monkeypatch.setattr(agfl.models.eegnet.search, 'run_search', forbidden)
    main(['tune-eegnet', '--dry-run', '--data-dir', str(tmp_path), '--output-dir', str(tmp_path / 'out')])
    configs = json.loads(capsys.readouterr().out)
    assert len(configs) == 5
    for variants in configs.values():
        assert len(variants) == 9
        for cfg in variants:
            assert cfg['model'] == 'eegnet' and cfg['attention'] == 'agfl' and cfg['dataset'] == 'eeg'
            assert cfg['model_options']['attention_axis'] == 'time'
            assert len(cfg['data']['subjects']) == 1 and cfg['seeds'] == list(range(5))
            assert cfg['data']['sessions'] == ['T'] and cfg['data']['filter_scope'] == 'trial'
            assert cfg['split'] == {'protocol': 'stratified', 'train': .6, 'validation': .2, 'test': .2}


def test_recombination_uses_only_same_subject_class_training_donors():
    # Every channel/time position identifies its original donor unambiguously.
    x = np.arange(6, dtype=np.float32)[:, None, None] * 1000
    x = x + np.arange(2, dtype=np.float32)[None, :, None] * 100 + np.arange(12, dtype=np.float32)[None, None, :]
    bundle = SignalDataset(x, np.array([0, 0, 1, 0, 0, 1]),
                           np.array(['A01', 'A01', 'A01', 'A02', 'A01', 'A02']),
                           [f'trial_{i}' for i in range(6)], {'modality': 'eeg', 'num_classes': 2})
    original = bundle.x.copy()
    augmentation = {'shift': 0, 'scale': 0., 'noise': 0., 'recombine_segments': 3, 'recombine_probability': 1.}
    train = Partition(bundle, [0, 1, 2, 3, 5], augmentation=augmentation)
    assert train.donor_pools[('A01', 0)] == [0, 1]  # Trial 4 is held out.
    np.random.seed(7)
    samples = [train[0][0].numpy() for _ in range(12)]
    np.random.seed(7)
    repeated = [train[0][0].numpy() for _ in range(12)]
    for sample, replay in zip(samples, repeated):
        np.testing.assert_array_equal(sample, replay)
        for left in (0, 4, 8):
            segment = sample[:, left:left + 4]
            assert any(np.array_equal(segment, bundle.x[donor, :, left:left + 4]) for donor in (0, 1))
    np.testing.assert_array_equal(bundle.x, original)
    np.testing.assert_array_equal(Partition(bundle, [4])[0][0].numpy(), bundle.x[4])


def test_validation_only_training_never_evaluates_test_and_stops_on_plateau(tmp_path, monkeypatch):
    from agfl.models._shared import ModelSpec
    import agfl.engine
    cfg = resolve_config(tiny_config(tmp_path))
    cfg['training'].update(epochs=10, early_stopping_patience=2, early_stopping_min_epochs=0)
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    split = get_split(bundle, cfg['split'], cfg['seeds'][0], cfg['split_dir'])
    path = tmp_path / 'candidate'
    path.mkdir()
    calls = []
    original_loaders = agfl.engine.make_loaders
    def guarded_loaders(*args, **kwargs):
        assert kwargs['include_test'] is False
        loaders, norm = original_loaders(*args, **kwargs)
        assert set(loaders) == {'train', 'validation'}
        return loaders, norm
    def constant_validation(self, model, loader, device, loss_fn):
        assert loader.dataset.indices == split['validation']
        calls.append('validation')
        return {'accuracy': .5, 'f1': .5, 'loss': 1., 'roc_auc': .5}, None, None
    monkeypatch.setattr(agfl.engine, 'make_loaders', guarded_loaders)
    monkeypatch.setattr(ModelSpec, 'evaluate', constant_validation)
    result = run_training(cfg, bundle, split, path, validation_only=True)
    assert result['epochs_trained'] == 3 and result['best_checkpoint_epoch'] == 1
    assert calls == ['validation'] * 3
    assert 'test' not in result and (path / 'selection.json').exists()
    assert not (path / 'result.json').exists() and not (path / 'predictions.npz').exists()


def test_candidate_selection_ignores_incomparable_losses_and_rejects_cross_seed_pooling():
    focal, ce = record(0, .6, .55), record(0, .6, .65)
    focal['validation']['loss'], ce['validation']['loss'] = .01, 100.
    assert select_candidate({'focal': [focal], 'ce': [ce]})[0] == 'ce'
    assert select_candidate({'first': [ce], 'second': [deepcopy(ce)]})[0] == 'first'
    with pytest.raises(ValueError, match='one matching seed'):
        select_candidate({'first': [record(0, .9), record(1, .9)]})
    with pytest.raises(ValueError, match='one matching seed'):
        select_candidate({'first': [record(0, .9)], 'second': [record(1, .9)]})
    with pytest.raises(ValueError, match='validation-only'):
        select_candidate({'first': [{**record(0, .9), 'test': {'accuracy': 1.}}]})
    with pytest.raises(ValueError, match='finite'):
        select_candidate({'first': [record(0, float('nan'))]})


def test_search_selects_each_split_before_testing_and_resumes(tmp_path, monkeypatch):
    import agfl.datasets
    import agfl.engine
    import agfl.reproducibility
    from agfl.datasets.base import normalize_samples

    root = tmp_path / 'search'
    configs = search_configs(tmp_path, root, [1], [0, 1],
                             ['compact_train_channel', 'compact_trialnorm'], epochs=2)
    for experiments in configs.values():
        experiments[0]['split_dir'] = str(tmp_path / 'splits')
    x = np.random.default_rng(10).normal(size=(40, 2, 32)).astype(np.float32)
    def data_loader(key, options):
        assert key == 'eeg' and options['subjects'] == [1]
        return SignalDataset(normalize_samples(x, options['normalization']), np.tile(np.arange(4), 10),
                             np.repeat('A01', 40), [f'trial_{i}' for i in range(40)],
                             {'modality': 'eeg', 'num_classes': 4, 'preprocessing': options})
    fits, tests, splits = [], [], {}
    def train(cfg, bundle, split, path, *, validation_only):
        assert validation_only
        seed, norm = cfg['seeds'][0], cfg['data']['normalization']
        assert split['sample_ids'] == splits.setdefault(seed, split['sample_ids'])
        fits.append((seed, norm))
        (path / 'checkpoint.pt').write_text(f'{seed}:{norm}')
        accuracy = .9 if (seed == 0) == (norm == 'train_channel') else .4
        result = {**record(seed, accuracy), 'config': cfg, 'subject_id': 'A01', 'elapsed_seconds': 1.}
        write_json(path / 'history.json', [{'epoch': 1, 'validation': result['validation']}])
        return result
    def evaluate(cfg, bundle, split, path, **kwargs):
        # All four candidates must finish and choices must exist before ANY test.
        assert len(fits) >= 4 and (root / 'selection_report.json').exists()
        seed, norm = cfg['seeds'][0], cfg['data']['normalization']
        assert norm == ('train_channel' if seed == 0 else 'per_sample')
        tests.append((seed, norm))
        (path / 'predictions.npz').write_bytes(b'mocked predictions')
        return {'status': 'completed', 'config': cfg, 'subject_id': 'A01', 'seed': seed,
                'validation': {'accuracy': .9}, 'test': {'accuracy': .5}, 'best_checkpoint_epoch': 1}
    def completed(path, cfg):
        return json.loads((path / 'result.json').read_text()) if (path / 'result.json').exists() else None
    monkeypatch.setattr(agfl.datasets, 'load_dataset', data_loader)
    monkeypatch.setattr(agfl.engine, 'run_training', train)
    monkeypatch.setattr(agfl.engine, 'evaluate_saved_checkpoint', evaluate)
    monkeypatch.setattr(agfl.engine, 'completed_run', completed)
    monkeypatch.setattr(agfl.reproducibility, 'provenance', lambda: {'source_sha256': 'test'})
    report = run_search(configs)
    assert report['choices'] == {'A01': {'0': 'compact_train_channel', '1': 'compact_trialnorm'}}
    assert report['mean_test_accuracy'] == .5 and not report['target_met']
    assert len(fits) == 4 and len(tests) == 2
    assert not list((root / 'candidates').rglob('result.json'))
    assert len(list((root / 'selected').rglob('result.json'))) == 2
    assert run_search(configs) == report
    assert len(fits) == 4 and len(tests) == 2
    # An aborted candidate is restarted in place, preserving unrelated files.
    candidate = root / 'candidates' / 'compact_trialnorm' / 'A01' / 'seed_1'
    write_json(candidate / 'failure.json', {'type': 'KeyboardInterrupt'})
    (candidate / 'notes.txt').write_text('keep')
    run_search(configs)
    assert len(fits) == 5 and not (candidate / 'failure.json').exists()
    assert len(tests) == 3  # Replaced candidates invalidate the old selected run.
    assert (candidate / 'notes.txt').read_text() == 'keep'
    changed = deepcopy(configs)
    for experiments in changed.values():
        experiments[0]['seeds'] = [0]
    with pytest.raises(ValueError, match='different search settings'):
        run_search(changed)


@pytest.mark.parametrize('augmentation', [{'recombine_segments': 1}, {'recombine_probability': 1.1}])
def test_invalid_recombination_fails_before_launch(augmentation):
    with pytest.raises(ValueError, match='recombine'):
        resolve_config({'dataset': 'eeg', 'model': 'eegnet', 'training': {'augmentation': augmentation}})
