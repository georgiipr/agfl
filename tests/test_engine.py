"""Research protocol invariants, to run only on the experiment machine."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from agfl.config import resolve_config
from agfl.datasets import SignalDataset, get_split, load_dataset
from agfl.engine import ClassificationLoss, make_loaders, make_scheduler, run_experiment
from agfl.metrics import classification_metrics


def tiny_config(tmp_path, **changes):
    config = {
        'dataset': 'synthetic_eeg', 'model': 'signal_transformer', 'attention': 'mha', 'attention_options': {'heads': 2}, 'device': 'cpu', 'seeds': [3],
        'output_dir': str(tmp_path / 'results'), 'split_dir': str(tmp_path / 'splits'),
        'data': {'subjects': 6, 'samples_per_subject': 8, 'channels': 4, 'samples': 32},
        'model_options': {'dim': 8, 'depth': 1, 'eeg_kernel_size': 3, 'eeg_temporal_bins': 2},
        'training': {'epochs': 2, 'batch_size': 8},
    }
    config.update(changes)
    return config


def test_focal_probability_is_unweighted():
    weights = torch.tensor([.25, 4.])
    logits = torch.tensor([[2., -1.], [-.5, 1.]], requires_grad=True)
    y = torch.tensor([0, 1])
    loss = ClassificationLoss(weights, 'focal', 2)(logits, y)
    probabilities = logits.softmax(-1)[torch.arange(2), y]
    expected = (-weights[y] * (1 - probabilities)**2 * probabilities.log()).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_undefined_auc_and_macro_f1_use_all_declared_classes():
    metrics = classification_metrics([0, 0], [[.9, .1], [.7, .3]])
    assert metrics['accuracy'] == 1
    assert metrics['f1'] == .5
    assert metrics['roc_auc'] is None
    assert metrics['roc_auc_reason']
    with pytest.raises(ValueError, match='normalized'):
        classification_metrics([0, 1], [[2., 1.], [0., 3.]])


def test_train_normalization_ignores_changed_test_values(tmp_path):
    cfg = resolve_config(tiny_config(tmp_path))
    cfg['data']['normalization'] = 'train_channel'
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    split = get_split(bundle, cfg['split'], 3, cfg['split_dir'])
    _, original = make_loaders(bundle, split, cfg, 3)
    values = bundle.x.copy()
    values[split['test']] += 10000
    changed = SignalDataset(values, bundle.y, bundle.groups, bundle.sample_ids, deepcopy(bundle.metadata))
    # Deliberately assign the same indices to altered held-out samples; refresh
    # identity because a production loader rightly rejects mismatched content.
    modified_split = deepcopy(split)
    modified_split['fingerprint'] = changed.fingerprint
    _, actual = make_loaders(changed, modified_split, cfg, 3)
    np.testing.assert_array_equal(actual['mean'], original['mean'])
    np.testing.assert_array_equal(actual['std'], original['std'])


def test_short_scheduler_does_not_have_negative_period(tmp_path):
    cfg = resolve_config(tiny_config(tmp_path))
    cfg['training']['epochs'] = 1
    parameter = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.AdamW([parameter], lr=.001)
    scheduler = make_scheduler(opt, cfg['training'])
    assert opt.param_groups[0]['lr'] > 0
    opt.step()
    scheduler.step()
    assert opt.param_groups[0]['lr'] >= 0


def test_saved_configuration_reproduces_selected_checkpoint_and_predictions(tmp_path):
    first = run_experiment(tiny_config(tmp_path))[0]
    result_path = next((tmp_path / 'results').rglob('result.json'))
    history = json.loads(result_path.with_name('history.json').read_text())
    expected_best = max(history, key=lambda epoch: epoch['validation']['accuracy'])['epoch']
    assert first['best_checkpoint_epoch'] == expected_best
    assert first['split_id'] == first['config']['expected_split_id']
    assert first['test']['n_samples'] > 0
    replay = deepcopy(first['config'])
    replay['output_dir'] = str(tmp_path / 'replay')
    second = run_experiment(replay)[0]
    assert first['test'] == second['test']
    assert first['validation'] == second['validation']
    original = np.load(result_path.with_name('predictions.npz'))
    repeated = np.load(next((tmp_path / 'replay').rglob('predictions.npz')))
    for key in original.files:
        np.testing.assert_array_equal(original[key], repeated[key])
    skipped = run_experiment(replay, skip_completed=True)[0]
    assert skipped['experiment_id'] == second['experiment_id']
    replaced = run_experiment(replay)[0]
    assert replaced['test'] == second['test']
    assert replaced['experiment_id'] == second['experiment_id']
    assert len(list((tmp_path / 'replay').rglob('result.json'))) == 1


@pytest.mark.parametrize('interruption', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('skip_completed', [False, True])
def test_interrupted_seed_restarts_without_stale_artifacts(tmp_path, monkeypatch, interruption, skip_completed):
    from agfl.models._shared import ModelSpec

    config = tiny_config(tmp_path)
    directories = []
    def interrupted_run(self, resolved, bundle, split, run_dir):
        directories.append(run_dir)
        for name in ('checkpoint.pt', 'history.json', 'predictions.npz', 'result.json', 'history.json.tmp'):
            (run_dir / name).write_text('interrupted output')
        (run_dir / 'notes.txt').write_text('keep my notes')
        raise interruption('stopped by test')
    monkeypatch.setattr(ModelSpec, 'run', interrupted_run)
    with pytest.raises(interruption):
        run_experiment(config)
    run_dir = directories[0]
    assert json.loads((run_dir / 'failure.json').read_text())['type'] == interruption.__name__
    saved_split = (run_dir / 'split.json').read_bytes()
    sibling = run_dir.parent / 'seed_99'
    sibling.mkdir()
    (sibling / 'result.json').write_text('keep the other seed')

    def restarted_run(self, resolved, bundle, split, destination):
        assert destination == run_dir
        assert (destination / 'split.json').read_bytes() == saved_split
        for name in ('checkpoint.pt', 'history.json', 'predictions.npz', 'result.json', 'failure.json', 'history.json.tmp'):
            assert not (destination / name).exists(), name
        assert (destination / 'config.json').is_file()
        assert (destination / 'notes.txt').read_text() == 'keep my notes'
        return {'status': 'restarted'}
    monkeypatch.setattr(ModelSpec, 'run', restarted_run)
    assert run_experiment(config, skip_completed=skip_completed) == [{'status': 'restarted'}]
    assert (sibling / 'result.json').read_text() == 'keep the other seed'


def test_skip_completed_restarts_an_incomplete_completion_record(tmp_path, monkeypatch):
    from agfl.models._shared import ModelSpec

    config = tiny_config(tmp_path)
    run_experiment(config)
    result_path = next((tmp_path / 'results').rglob('result.json'))
    # A completion record alone must not prevent recovery of a damaged run.
    result_path.with_name('predictions.npz').unlink()
    def restarted_run(self, resolved, bundle, split, run_dir):
        assert not (run_dir / 'result.json').exists()
        assert not (run_dir / 'checkpoint.pt').exists()
        return {'status': 'restarted'}
    monkeypatch.setattr(ModelSpec, 'run', restarted_run)
    assert run_experiment(config, skip_completed=True) == [{'status': 'restarted'}]


def test_abort_during_replacement_cannot_leave_old_completed_results(tmp_path, monkeypatch):
    from agfl.models._shared import ModelSpec

    config = tiny_config(tmp_path)
    run_experiment(config)
    result_path = next((tmp_path / 'results').rglob('result.json'))
    def interrupted_run(self, resolved, bundle, split, run_dir):
        assert not (run_dir / 'result.json').exists()
        assert not (run_dir / 'checkpoint.pt').exists()
        raise KeyboardInterrupt('replacement interrupted')
    monkeypatch.setattr(ModelSpec, 'run', interrupted_run)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config)
    assert not result_path.exists()
    assert result_path.with_name('failure.json').is_file()


def test_active_seed_cannot_be_overwritten_and_lock_releases_after_abort(tmp_path):
    from agfl.storage import run_directory

    destination = tmp_path / 'seed_0'
    with pytest.raises(KeyboardInterrupt):
        with run_directory(destination):
            (destination / 'checkpoint.pt').write_text('active checkpoint')
            with pytest.raises(RuntimeError, match='Another process'):
                with run_directory(destination):
                    raise AssertionError('Concurrent writer entered')
            assert (destination / 'checkpoint.pt').read_text() == 'active checkpoint'
            raise KeyboardInterrupt
    with run_directory(destination):
        assert (destination / 'checkpoint.pt').read_text() == 'active checkpoint'


def test_defaults_keep_five_seeds_and_separate_domain_budgets():
    eeg, ecg = resolve_config({'dataset': 'eeg'}), resolve_config({'dataset': 'ecg'})
    assert eeg['seeds'] == [0, 1, 2, 3, 4]
    assert eeg['training']['epochs'] == 250
    assert ecg['training']['epochs'] == 50
    assert eeg['model_variant'] == 'eeg' and ecg['model_variant'] == 'ecg'
    with pytest.raises(ValueError, match='Unknown'):
        resolve_config({'training': {'learnng_rate': .001}})


def test_worker_augmentation_repeats_with_same_loader_seed(tmp_path):
    cfg = resolve_config(tiny_config(tmp_path))
    cfg['training']['num_workers'] = 2
    cfg['training']['augmentation'] = {'shift': 2, 'scale': .1, 'noise': .05}
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    split = get_split(bundle, cfg['split'], 3, cfg['split_dir'])
    first, _ = make_loaders(bundle, split, cfg, 3)
    second, _ = make_loaders(bundle, split, cfg, 3)
    for (x, y), (xx, yy) in zip(first['train'], second['train']):
        torch.testing.assert_close(x, xx, rtol=0, atol=0)
        torch.testing.assert_close(y, yy, rtol=0, atol=0)
