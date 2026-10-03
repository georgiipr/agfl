"""Training-engine protocol invariants on the tiny SI_Hom fixture archive.

Development checks for the experiment machine only; they are not a prerequisite
for training. Every run uses ``tests.si_hom_fixture.tiny_config``: a two-epoch
CPU ``signal_transformer`` + ``mha`` fit on 7 subjects x 8 classes x 4 trials.
The numbers these runs produce are meaningless as accuracy; only the protocol
(isolation of held-out data, saved identities, restart behaviour) is checked.

Run from the project folder: ``python -m pytest tests/test_engine.py -q``.
"""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from agfl_speech.config import resolve_config
from agfl_speech.datasets import SignalDataset, get_split, load_dataset
from agfl_speech.engine import (ClassificationLoss, Partition, evaluate_saved_checkpoint, make_loaders,
                                make_scheduler, run_experiment, run_training)
from agfl_speech.metrics import classification_metrics
from tests.si_hom_fixture import tiny_config


SUBJECT_IDS = [f'S{subject:02d}' for subject in range(1, 8)]


def report_run_directory(session, run_dir):
    """The downloadable copy of one artifacts seed directory."""
    return session / 'report' / 'runs' / run_dir.relative_to(session / 'artifacts')


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


def test_python_launch_expands_individual_subjects_and_keeps_a_pooled_cohort_whole(monkeypatch):
    import agfl_speech.engine
    launched = []

    def record(config, skip_completed=False):
        launched.append(config)
        return [{'subject_id': config['subject_id']}]
    monkeypatch.setattr(agfl_speech.engine, '_run_experiment', record)
    results = run_experiment({'data': {'cohort': 'individual', 'subjects': [2, 1]}})
    assert results == [{'subject_id': 'S01'}, {'subject_id': 'S02'}]
    assert [config['data']['subjects'] for config in launched] == [[1], [2]]
    assert all(config['seeds'] == [0, 1, 2, 3, 4] for config in launched)
    launched.clear()
    assert run_experiment({'data': {'subjects': [2, 1]}}) == [{'subject_id': None}]
    assert [config['data']['subjects'] for config in launched] == [[2, 1]]


def test_undefined_auc_and_macro_f1_use_all_declared_classes():
    metrics = classification_metrics([0, 0], [[.9, .1], [.7, .3]])
    assert metrics['accuracy'] == 1
    assert metrics['f1'] == .5
    assert metrics['roc_auc'] is None
    assert metrics['roc_auc_reason']
    with pytest.raises(ValueError, match='normalized'):
        classification_metrics([0, 1], [[2., 1.], [0., 3.]])


def test_defaults_keep_five_seeds_and_the_eeg_training_policy():
    config = resolve_config({})
    assert config['dataset'] == 'si_hom' and config['model_variant'] == 'eeg'
    assert config['seeds'] == [0, 1, 2, 3, 4]
    assert config['split'] == {'protocol': 'stratified', 'train': .6, 'validation': .2, 'test': .2}
    assert config['data']['cohort'] == 'pooled' and config['subject_id'] is None
    training = config['training']
    assert training['epochs'] == 500 and training['loss'] == 'cross_entropy'
    assert training['learning_rate'] == 1e-3 and training['checkpoint_criterion'] == 'loss'
    with pytest.raises(ValueError, match='Unknown'):
        resolve_config({'training': {'learnng_rate': .001}})
    with pytest.raises(ValueError, match='save_checkpoints'):
        resolve_config({'training': {'save_checkpoints': 'false'}})


@pytest.mark.parametrize('augmentation', [{'recombine_segments': 1}, {'recombine_probability': 1.1}])
def test_invalid_recombination_fails_before_launch(augmentation):
    with pytest.raises(ValueError, match='recombine'):
        resolve_config({'model': 'eegnet', 'training': {'augmentation': augmentation}})


def test_segment_recombination_is_limited_to_the_eegnet_backbones():
    augmentation = {'training': {'augmentation': {'recombine_segments': 2}}}
    for model in ('eegnet', 'eegnet_original'):
        resolved = resolve_config({'model': model, **augmentation})
        assert resolved['training']['augmentation']['recombine_segments'] == 2
    with pytest.raises(ValueError, match='EEGNet backbones'):
        resolve_config({'model': 'signal_transformer', **augmentation})


def test_train_normalization_ignores_changed_test_values(tmp_path):
    cfg = resolve_config(tiny_config(tmp_path))
    assert cfg['data']['normalization'] == 'train_channel'
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    split = get_split(bundle, cfg['split'], 3, cfg['split_dir'])
    _, original = make_loaders(bundle, split, cfg, 3)
    training = bundle.x[split['train']]
    assert original['mean'].shape == original['std'].shape == (19, 1)
    np.testing.assert_allclose(original['mean'][:, 0], training.mean(axis=(0, 2), dtype=np.float64), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(original['std'][:, 0], training.std(axis=(0, 2), dtype=np.float64), rtol=1e-5, atol=1e-6)
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
    with pytest.raises(ValueError, match='fingerprint'):
        make_loaders(changed, split, cfg, 3)


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


def test_recombination_uses_only_same_subject_class_training_donors():
    # Every channel/time position identifies its original donor unambiguously.
    x = np.arange(6, dtype=np.float32)[:, None, None] * 1000
    x = x + np.arange(2, dtype=np.float32)[None, :, None] * 100 + np.arange(12, dtype=np.float32)[None, None, :]
    bundle = SignalDataset(x, np.array([0, 0, 1, 0, 0, 1]),
                           np.array(['S01', 'S01', 'S01', 'S02', 'S01', 'S02']),
                           [f'trial_{i}' for i in range(6)], {'modality': 'eeg', 'num_classes': 2})
    original = bundle.x.copy()
    augmentation = {'shift': 0, 'scale': 0., 'noise': 0., 'recombine_segments': 3, 'recombine_probability': 1.}
    train = Partition(bundle, [0, 1, 2, 3, 5], augmentation=augmentation)
    assert train.donor_pools[('S01', 0)] == [0, 1]  # Trial 4 is held out.
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


def test_saved_configuration_reproduces_selected_checkpoint_and_predictions(tmp_path):
    first = run_experiment(tiny_config(tmp_path))[0]
    session = tmp_path / 'results'
    result_path = next((session / 'artifacts').rglob('result.json'))
    run_dir = result_path.parent
    # Pooled layout: artifacts/<dataset>/<model>-<attention>-<experiment id>/seed_<n>.
    assert run_dir.name == 'seed_3'
    assert run_dir.parent.name == f"signal_transformer-mha-{first['experiment_id']}"
    assert run_dir.parent.parent == session / 'artifacts' / 'si_hom'
    report_run = report_run_directory(session, run_dir)
    for name in ('result.json', 'config.json', 'split.json', 'history.json', 'predictions.npz'):
        assert (report_run / name).read_bytes() == result_path.with_name(name).read_bytes()
    assert not list((session / 'report').rglob('checkpoint.pt'))
    assert (run_dir / 'checkpoint.pt').is_file()
    history = json.loads(result_path.with_name('history.json').read_text())
    assert len(history) == 2
    assert all(0 <= row['train_accuracy'] <= 1 for row in history)
    # The EEG policy selects the first epoch with the lowest validation loss.
    assert first['config']['training']['checkpoint_criterion'] == 'loss'
    expected_best = min(history, key=lambda epoch: epoch['validation']['loss'])['epoch']
    assert first['best_checkpoint_epoch'] == expected_best
    assert first['split_id'] == first['config']['expected_split_id']
    assert first['dataset_fingerprint'] == first['config']['dataset_fingerprint']
    # Stratified 0.6/0.2/0.2 of 4 trials per subject and class.
    assert first['validation']['n_samples'] == 56 and first['test']['n_samples'] == 56
    assert first['training_class_counts'] == [14] * 8
    assert first['subject_id'] is None
    assert sorted(first['test_by_subject']) == SUBJECT_IDS
    assert all(metrics['n_samples'] == 8 for metrics in first['test_by_subject'].values())
    assert 'synthetic' not in first
    assert first['token_axis'] == 'electrode' and first['num_tokens'] == 19
    assert first['node_labels'] == first['config']['resolved_metadata']['channel_names']
    assert [(module['attention'], module['num_tokens'], module['token_axis'])
            for module in first['attention_modules']] == [('mha', 19, 'electrode')]
    assert json.loads(result_path.read_text())['test'] == first['test']
    replay = deepcopy(first['config'])
    replay['output_dir'] = str(tmp_path / 'replay')
    second = run_experiment(replay)[0]
    assert first['test'] == second['test']
    assert first['validation'] == second['validation']
    assert first['experiment_id'] == second['experiment_id']
    original = np.load(result_path.with_name('predictions.npz'))
    repeated = np.load(next((tmp_path / 'replay' / 'artifacts').rglob('predictions.npz')))
    for key in original.files:
        np.testing.assert_array_equal(original[key], repeated[key])
    skipped = run_experiment(replay, skip_completed=True)[0]
    assert skipped['experiment_id'] == second['experiment_id']
    replaced = run_experiment(replay)[0]
    assert replaced['test'] == second['test']
    assert replaced['experiment_id'] == second['experiment_id']
    assert len(list((tmp_path / 'replay' / 'artifacts').rglob('result.json'))) == 1
    assert len(list((tmp_path / 'replay' / 'report' / 'runs').rglob('result.json'))) == 1


def test_individual_cohort_trains_one_model_per_subject(tmp_path):
    config = tiny_config(tmp_path)
    config['data'].update(cohort='individual', subjects=[5, 2])
    results = run_experiment(config)
    assert [result['subject_id'] for result in results] == ['S02', 'S05']
    artifacts = tmp_path / 'results' / 'artifacts' / 'si_hom'
    assert sorted(path.name for path in artifacts.iterdir()) == ['subject_S02', 'subject_S05']
    for result in results:
        subject = result['subject_id']
        assert result['config']['data']['subjects'] == [int(subject[1:])]
        assert list(result['test_by_subject']) == [subject]
        # One subject: 32 trials, two/one/one of every class per partition.
        assert result['training_class_counts'] == [2] * 8
        assert result['validation']['n_samples'] == 8 and result['test']['n_samples'] == 8
        assert len(list((artifacts / f'subject_{subject}').rglob('result.json'))) == 1
    assert results[0]['split_id'] != results[1]['split_id']
    assert results[0]['dataset_fingerprint'] != results[1]['dataset_fingerprint']


def test_individual_experiment_rejects_data_of_another_subject(tmp_path):
    cfg = resolve_config(tiny_config(tmp_path))
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    cfg['subject_id'] = 'S01'
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    with pytest.raises(ValueError, match='another subject'):
        run_training(cfg, bundle, None, run_dir)
    assert list(run_dir.iterdir()) == []


@pytest.mark.parametrize('saved,message', [
    ({'provenance': {'source_sha256': 'another source', 'packages': {}}}, 'recorded source'),
    ({'dataset_fingerprint': '0' * 64}, 'Dataset bytes/preprocessing differ'),
    ({'expected_split_id': '0' * 64}, 'Generated split differs'),
])
def test_saved_identities_must_match_the_current_source_data_and_split(tmp_path, saved, message):
    with pytest.raises(ValueError, match=message):
        run_experiment(tiny_config(tmp_path, **saved))
    assert not list((tmp_path / 'results').rglob('config.json'))


def test_evaluation_rejects_a_checkpoint_of_another_split_or_normalization(tmp_path):
    result = run_experiment(tiny_config(tmp_path))[0]
    run_dir = next((tmp_path / 'results' / 'artifacts').rglob('checkpoint.pt')).parent
    config = result['config']
    bundle = load_dataset(config['dataset'], config['data'])
    split = get_split(bundle, config['split'], config['seeds'][0], config['split_dir'])
    assert split['split_id'] == result['split_id']
    checkpoint = torch.load(run_dir / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == result['best_checkpoint_epoch']
    assert checkpoint['normalization']['mean'].shape == (19, 1)
    # Control: the unchanged checkpoint reproduces the saved evaluation.
    repeated = evaluate_saved_checkpoint(config, bundle, split, run_dir)
    assert repeated['test'] == result['test'] and repeated['validation'] == result['validation']
    foreign = dict(checkpoint, split_id='another split')
    with pytest.raises(ValueError, match='does not match'):
        evaluate_saved_checkpoint(config, bundle, split, run_dir, checkpoint=foreign)
    shifted = dict(checkpoint, normalization={'mean': checkpoint['normalization']['mean'] + 1,
                                              'std': checkpoint['normalization']['std']})
    with pytest.raises(ValueError, match='normalization'):
        evaluate_saved_checkpoint(config, bundle, split, run_dir, checkpoint=shifted)
    unnormalized = dict(checkpoint, normalization=None)
    with pytest.raises(ValueError, match='normalization'):
        evaluate_saved_checkpoint(config, bundle, split, run_dir, checkpoint=unnormalized)


def test_validation_only_search_cannot_discard_required_weights(tmp_path):
    with pytest.raises(ValueError, match='Validation-only'):
        run_training({'training': {'save_checkpoints': False}}, None, None, tmp_path, validation_only=True)


def test_validation_only_training_never_evaluates_test_and_stops_on_plateau(tmp_path, monkeypatch):
    from agfl_speech.models._shared import ModelSpec
    import agfl_speech.engine
    cfg = resolve_config(tiny_config(tmp_path))
    cfg['training'].update(epochs=10, early_stopping_patience=2, early_stopping_min_epochs=0)
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    split = get_split(bundle, cfg['split'], cfg['seeds'][0], cfg['split_dir'])
    path = tmp_path / 'candidate'
    path.mkdir()
    calls = []
    original_loaders = agfl_speech.engine.make_loaders

    def guarded_loaders(*args, **kwargs):
        assert kwargs['include_test'] is False
        loaders, norm = original_loaders(*args, **kwargs)
        assert set(loaders) == {'train', 'validation'}
        return loaders, norm

    def constant_validation(self, model, loader, device, loss_fn):
        assert loader.dataset.indices == split['validation']
        calls.append('validation')
        return {'accuracy': .5, 'f1': .5, 'loss': 1., 'roc_auc': .5}, None, None
    monkeypatch.setattr(agfl_speech.engine, 'make_loaders', guarded_loaders)
    monkeypatch.setattr(ModelSpec, 'evaluate', constant_validation)
    result = run_training(cfg, bundle, split, path, validation_only=True)
    assert result['status'] == 'validation_completed'
    assert result['epochs_trained'] == 3 and result['best_checkpoint_epoch'] == 1
    assert calls == ['validation'] * 3
    assert 'test' not in result and (path / 'selection.json').exists()
    assert not (path / 'result.json').exists() and not (path / 'predictions.npz').exists()


def test_memory_selection_matches_disk_selection_and_never_calls_torch_save(tmp_path, monkeypatch):
    import agfl_speech.engine
    cfg = tiny_config(tmp_path, output_dir=str(tmp_path / 'disk'))
    cfg['training']['epochs'] = 3
    disk, = run_experiment(cfg)
    memory_config = deepcopy(cfg)
    memory_config['output_dir'] = str(tmp_path / 'memory')
    memory_config['training']['save_checkpoints'] = False

    def forbidden_save(*args, **kwargs):
        raise AssertionError('Checkpoint write is forbidden')
    with monkeypatch.context() as patched:
        patched.setattr(torch, 'save', forbidden_save)
        memory, = run_experiment(memory_config)
    assert disk['checkpoint_retained'] is True
    assert memory['checkpoint_retained'] is False
    assert memory['best_checkpoint_epoch'] == disk['best_checkpoint_epoch']
    assert memory['validation'] == disk['validation']
    assert memory['test'] == disk['test']
    assert not list((tmp_path / 'memory').rglob('*.pt'))
    left = next((tmp_path / 'disk' / 'artifacts').rglob('predictions.npz'))
    right = next((tmp_path / 'memory' / 'artifacts').rglob('predictions.npz'))
    with np.load(left) as saved, np.load(right) as in_memory:
        for key in saved.files:
            np.testing.assert_array_equal(saved[key], in_memory[key])

    # A completed run without a checkpoint is still recognised by --skip-completed.
    def forbidden_training(*args, **kwargs):
        raise AssertionError('Unexpected retraining')
    monkeypatch.setattr(agfl_speech.engine, 'run_training', forbidden_training)
    skipped, = run_experiment(memory_config, skip_completed=True)
    assert skipped['test'] == memory['test']


@pytest.mark.parametrize('interruption', [RuntimeError, KeyboardInterrupt])
@pytest.mark.parametrize('skip_completed', [False, True])
def test_interrupted_seed_restarts_without_stale_artifacts(tmp_path, monkeypatch, interruption, skip_completed):
    from agfl_speech.models._shared import ModelSpec

    config = tiny_config(tmp_path)
    directories = []

    def interrupted_run(self, resolved, bundle, split, run_dir):
        directories.append(run_dir)
        for name in ('checkpoint.pt', 'history.json', 'predictions.npz', 'result.json', 'selection.json', 'history.json.tmp'):
            (run_dir / name).write_text('interrupted output')
        (run_dir / 'notes.txt').write_text('keep my notes')
        raise interruption('stopped by test')
    monkeypatch.setattr(ModelSpec, 'run', interrupted_run)
    with pytest.raises(interruption):
        run_experiment(config)
    run_dir = directories[0]
    assert json.loads((run_dir / 'failure.json').read_text())['type'] == interruption.__name__
    report_run = report_run_directory(tmp_path / 'results', run_dir)
    assert (report_run / 'failure.json').read_bytes() == (run_dir / 'failure.json').read_bytes()
    assert not (report_run / 'checkpoint.pt').exists()
    saved_split = (run_dir / 'split.json').read_bytes()
    sibling = run_dir.parent / 'seed_99'
    sibling.mkdir()
    (sibling / 'result.json').write_text('keep the other seed')

    def restarted_run(self, resolved, bundle, split, destination):
        assert destination == run_dir
        assert (destination / 'split.json').read_bytes() == saved_split
        for name in ('checkpoint.pt', 'history.json', 'predictions.npz', 'result.json', 'selection.json', 'failure.json', 'history.json.tmp'):
            assert not (destination / name).exists(), name
            assert not (report_run / name).exists(), name
        assert (destination / 'config.json').is_file()
        assert (report_run / 'config.json').read_bytes() == (destination / 'config.json').read_bytes()
        assert (destination / 'notes.txt').read_text() == 'keep my notes'
        return {'status': 'restarted'}
    monkeypatch.setattr(ModelSpec, 'run', restarted_run)
    assert run_experiment(config, skip_completed=skip_completed) == [{'status': 'restarted'}]
    assert (sibling / 'result.json').read_text() == 'keep the other seed'


def test_skip_completed_restarts_an_incomplete_completion_record(tmp_path, monkeypatch):
    from agfl_speech.models._shared import ModelSpec

    config = tiny_config(tmp_path)
    run_experiment(config)
    result_path = next((tmp_path / 'results' / 'artifacts').rglob('result.json'))
    report_run = report_run_directory(tmp_path / 'results', result_path.parent)
    # A completion record alone must not prevent recovery of a damaged run.
    # Leave the downloadable mirror intact: canonical damage must still restart.
    result_path.with_name('predictions.npz').unlink()
    assert (report_run / 'predictions.npz').is_file()

    def restarted_run(self, resolved, bundle, split, run_dir):
        assert not (run_dir / 'result.json').exists()
        assert not (run_dir / 'checkpoint.pt').exists()
        assert not (report_run / 'result.json').exists()
        assert not (report_run / 'predictions.npz').exists()
        return {'status': 'restarted'}
    monkeypatch.setattr(ModelSpec, 'run', restarted_run)
    assert run_experiment(config, skip_completed=True) == [{'status': 'restarted'}]


def test_abort_during_replacement_cannot_leave_old_completed_results(tmp_path, monkeypatch):
    from agfl_speech.models._shared import ModelSpec

    config = tiny_config(tmp_path)
    run_experiment(config)
    result_path = next((tmp_path / 'results' / 'artifacts').rglob('result.json'))
    report_run = report_run_directory(tmp_path / 'results', result_path.parent)

    def interrupted_run(self, resolved, bundle, split, run_dir):
        assert not (run_dir / 'result.json').exists()
        assert not (run_dir / 'checkpoint.pt').exists()
        assert not (report_run / 'result.json').exists()
        raise KeyboardInterrupt('replacement interrupted')
    monkeypatch.setattr(ModelSpec, 'run', interrupted_run)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config)
    assert not result_path.exists()
    assert result_path.with_name('failure.json').is_file()
    assert not (report_run / 'result.json').exists()
    assert (report_run / 'failure.json').read_bytes() == result_path.with_name('failure.json').read_bytes()


def test_active_seed_cannot_be_overwritten_and_lock_releases_after_abort(tmp_path):
    from agfl_speech.storage import run_directory

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


def test_worker_augmentation_repeats_with_same_loader_seed(tmp_path):
    cfg = resolve_config(tiny_config(tmp_path))
    cfg['training']['num_workers'] = 2
    cfg['training']['augmentation'].update(shift=2, scale=.1, noise=.05)
    bundle = load_dataset(cfg['dataset'], cfg['data'])
    split = get_split(bundle, cfg['split'], 3, cfg['split_dir'])
    first, _ = make_loaders(bundle, split, cfg, 3)
    second, _ = make_loaders(bundle, split, cfg, 3)
    batches = 0
    for (x, y), (xx, yy) in zip(first['train'], second['train']):
        torch.testing.assert_close(x, xx, rtol=0, atol=0)
        torch.testing.assert_close(y, yy, rtol=0, atol=0)
        batches += 1
    assert batches == 7  # 112 training trials in batches of 16
