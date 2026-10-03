"""Optional EMA invariants; execute only on the experiment machine."""
from copy import deepcopy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from agfl_speech.config import resolve_config
from agfl_speech.datasets import get_split, load_dataset
from agfl_speech.engine import (CalibrationInputs, ExponentialMovingAverage,
                                calibrate_batch_norm, checkpoint_is_better,
                                evaluate_saved_checkpoint, make_loaders, run_experiment)
from tests.si_hom_fixture import write_si_hom_fixture


def tiny_ema_config(tmp_path, decay=.5):
    # The fixture archive holds 224 trials of 19 electrodes x 64 samples; the
    # stratified split trains on 112 of them (14 full batches of 8).
    return {
        'dataset': 'si_hom', 'model': 'eegnet', 'attention': 'mha',
        'attention_options': {'heads': 2}, 'device': 'cpu', 'seeds': [3],
        'output_dir': str(tmp_path / 'results'), 'split_dir': str(tmp_path / 'splits'),
        'data': {'data_dir': write_si_hom_fixture(tmp_path / 'data')},
        'model_options': {'f1': 2, 'd': 2, 'f2': 4, 'temp_kernel': 8,
                          'pk1': 4, 'pk2': 4, 'attention_axis': 'electrode', 'electrode_dim': 8},
        # The f1/loss tie-breaker exists only for accuracy selection; the EEG
        # default criterion is the validation loss.
        'training': {'epochs': 3, 'batch_size': 8, 'loss': 'cross_entropy',
                     'checkpoint_criterion': 'accuracy',
                     'ema_decay': decay, 'checkpoint_tiebreaker': 'f1_loss'},
    }


def test_ema_initialization_and_updates_preserve_live_model_and_rng():
    model = nn.Sequential(nn.Linear(2, 2), nn.BatchNorm1d(2))
    before = {name: value.clone() for name, value in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    ema = ExponentialMovingAverage(model, .75)
    torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
    assert all(not parameter.requires_grad for parameter in ema.model.parameters())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2)
        model[1].running_mean.fill_(3)
        model[1].running_var.fill_(4)
        model[1].num_batches_tracked.fill_(7)
    live = {name: value.clone() for name, value in model.state_dict().items()}
    ema.update(model)
    assert ema.updates == 1
    for name, parameter in ema.model.named_parameters():
        torch.testing.assert_close(parameter, before[name] + .5)
    for name, buffer in ema.model.named_buffers():
        torch.testing.assert_close(buffer, live[name], rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, live[name], rtol=0, atol=0)
    assert model.training and model[1].training


class LabelsForbidden:
    def __init__(self, values):
        self.x = values

    @property
    def y(self):
        raise AssertionError('Calibration must not read labels')


class CalibrationProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2, momentum=.17)
        self.dropout = nn.Dropout(.9)
        self.observed_modes = []
        self.fail = False

    def forward(self, x):
        self.observed_modes.append((self.training, self.bn.training, self.dropout.training))
        # A defensive RNG-restoration test even if a future model consumes RNG
        # in evaluation. These draws do not change the calibration inputs.
        torch.rand(1)
        random.random()
        np.random.random()
        if self.fail:
            raise RuntimeError('deliberate calibration failure')
        return self.dropout(self.bn(x))


def test_calibration_is_train_only_sample_weighted_and_restores_modes_rng():
    values = np.arange(6 * 2 * 4, dtype=np.float32).reshape(6, 2, 4)
    values[[1, 3, 5]] = 1e8  # Held-out values must not affect the result.
    normalization = {'mean': np.array([[2.], [3.]], dtype=np.float32),
                     'std': np.array([[2.], [4.]], dtype=np.float32)}
    inputs = CalibrationInputs(LabelsForbidden(values), [0, 2, 4], normalization)
    loader = DataLoader(inputs, batch_size=2, shuffle=False,
                        generator=torch.Generator().manual_seed(99), drop_last=False)
    model = CalibrationProbe().train()
    model.dropout.eval()  # Mixed original modes must be restored exactly.
    modes = [module.training for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    metadata = calibrate_batch_norm(model, loader, torch.device('cpu'))
    x = torch.from_numpy((values[[0, 2, 4]] - normalization['mean']) / normalization['std'])
    expected_mean = x.mean(dim=(0, 2))
    expected_variance = (2 * x[:2].var(dim=(0, 2), unbiased=True)
                         + x[2:].var(dim=(0, 2), unbiased=True)) / 3
    torch.testing.assert_close(model.bn.running_mean, expected_mean)
    torch.testing.assert_close(model.bn.running_var, expected_variance)
    assert model.bn.num_batches_tracked.item() == 2
    assert metadata['samples'] == 3 and metadata['batches'] == 2
    assert metadata['partition'] == 'train'
    assert metadata['method'] == 'sample_weighted_batch_statistics'
    assert model.observed_modes == [(False, True, False)] * 2
    assert [module.training for module in model.modules()] == modes
    assert model.bn.momentum == .17
    assert random.getstate() == python_state
    assert np.random.get_state()[0] == numpy_state[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


def test_calibration_restores_modes_momentum_rng_after_failure():
    model = CalibrationProbe().train()
    model.fail = True
    loader = DataLoader(torch.ones(3, 2, 4), batch_size=2,
                        generator=torch.Generator().manual_seed(10))
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()
    with pytest.raises(RuntimeError, match='deliberate'):
        calibrate_batch_norm(model, loader, torch.device('cpu'))
    assert all(layer.training for layer in model.modules())
    assert model.bn.momentum == .17
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)


def test_calibration_loader_is_opt_in_and_includes_all_train_inputs_without_test(tmp_path):
    config = resolve_config(tiny_ema_config(tmp_path))
    config['training']['augmentation'].update({'shift': 5, 'scale': .4, 'noise': .3})
    bundle = load_dataset(config['dataset'], config['data'])
    split = get_split(bundle, config['split'], 3, config['split_dir'])
    loaders, normalization = make_loaders(bundle, split, config, 3, include_test=False)
    assert set(loaders) == {'train', 'validation', 'calibration'}
    assert loaders['calibration'].dataset.indices == split['train']
    actual = torch.cat(list(loaders['calibration'])).numpy()
    expected = bundle.x[split['train']]
    if normalization is not None:
        expected = (expected - normalization['mean']) / normalization['std']
    np.testing.assert_array_equal(actual, expected)
    config['training']['ema_decay'] = 0
    ordinary, _ = make_loaders(bundle, split, config, 3, include_test=False)
    assert set(ordinary) == {'train', 'validation'}


def test_checkpoint_tiebreaker_uses_f1_then_loss_only_on_accuracy_ties():
    best = {'accuracy': .8, 'f1': .79, 'loss': .5}
    assert checkpoint_is_better({'accuracy': .81, 'f1': .7, 'loss': .9}, best, 'accuracy', 'f1_loss')
    assert not checkpoint_is_better({'accuracy': .79, 'f1': .99, 'loss': .01}, best, 'accuracy', 'f1_loss')
    assert checkpoint_is_better({'accuracy': .8, 'f1': .8, 'loss': .9}, best, 'accuracy', 'f1_loss')
    assert checkpoint_is_better({'accuracy': .8, 'f1': .79, 'loss': .4}, best, 'accuracy', 'f1_loss')
    assert not checkpoint_is_better(best, best, 'accuracy', 'f1_loss')
    assert not checkpoint_is_better({'accuracy': .8, 'f1': 1., 'loss': .01}, best, 'accuracy')
    assert checkpoint_is_better({'loss': .4}, {'loss': .5}, 'loss')
    for name in ('accuracy', 'f1', 'loss'):
        invalid = {**best, name: float('nan')}
        with pytest.raises(ValueError, match='undefined'):
            checkpoint_is_better(invalid, None, 'accuracy', 'f1_loss')


@pytest.mark.parametrize('value', [-.1, 1., float('nan'), float('inf'), True, '.9'])
def test_invalid_ema_decay_is_rejected(value):
    with pytest.raises(ValueError, match='ema_decay'):
        resolve_config({'training': {'ema_decay': value}})


def test_ema_and_tiebreaker_are_disabled_by_default():
    config = resolve_config({})
    assert config['training']['ema_decay'] == 0
    assert config['training']['checkpoint_tiebreaker'] == 'none'
    assert config['training']['checkpoint_criterion'] == 'loss'
    with pytest.raises(ValueError, match='requires checkpoint_criterion=accuracy'):
        resolve_config({'training': {'checkpoint_criterion': 'loss', 'checkpoint_tiebreaker': 'f1_loss'}})


def test_selected_ema_state_is_replayed_unchanged_for_validation_and_test(tmp_path):
    result = run_experiment(tiny_ema_config(tmp_path))[0]
    result_path = next((tmp_path / 'results' / 'artifacts').rglob('result.json'))
    checkpoint_path = result_path.with_name('checkpoint.pt')
    original_checkpoint = checkpoint_path.read_bytes()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    history = json.loads(result_path.with_name('history.json').read_text())
    selected = max(history, key=lambda row: (row['validation']['accuracy'],
                                            row['validation']['f1'], -row['validation']['loss']))
    assert result['best_checkpoint_epoch'] == selected['epoch']
    assert result['checkpoint_policy'] == checkpoint['checkpoint_policy'] == selected['checkpoint_policy']
    assert result['checkpoint_policy']['weights'] == 'ema'
    assert result['checkpoint_policy']['training_metrics_weights'] == 'raw'
    assert checkpoint['validation'] == result['validation']
    split = json.loads(result_path.with_name('split.json').read_text())
    assert result['checkpoint_policy']['batch_norm_calibration']['samples'] == len(split['train'])
    expected_updates = selected['epoch'] * ((len(split['train']) + 7) // 8)
    assert result['checkpoint_policy']['ema_updates'] == expected_updates
    original_predictions = {key: value.copy() for key, value in
                            np.load(result_path.with_name('predictions.npz')).items()}
    bundle = load_dataset(result['config']['dataset'], result['config']['data'])
    replay = evaluate_saved_checkpoint(deepcopy(result['config']), bundle, split, result_path.parent)
    assert replay['validation'] == result['validation']
    assert replay['test'] == result['test']
    for key, value in np.load(result_path.with_name('predictions.npz')).items():
        np.testing.assert_array_equal(value, original_predictions[key])
    assert checkpoint_path.read_bytes() == original_checkpoint


def test_ema_excludes_optimizer_steps_skipped_by_the_scaler(tmp_path, monkeypatch):
    class AlternatingScaler:
        def __init__(self):
            self.steps, self.current_scale = 0, 64.

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def get_scale(self):
            return self.current_scale

        def step(self, optimizer):
            self.steps += 1
            if self.steps % 2:
                optimizer.step()

        def update(self):
            if not self.steps % 2:
                self.current_scale /= 2

    scaler = AlternatingScaler()
    monkeypatch.setattr(torch.amp, 'GradScaler', lambda *args, **kwargs: scaler)
    run_experiment(tiny_ema_config(tmp_path))
    history_path = next((tmp_path / 'results' / 'artifacts').rglob('history.json'))
    history = json.loads(history_path.read_text())
    split = json.loads(history_path.with_name('split.json').read_text())
    steps_per_epoch = (len(split['train']) + 7) // 8
    for row in history:
        total_steps = row['epoch'] * steps_per_epoch
        assert row['checkpoint_policy']['ema_updates'] == (total_steps + 1) // 2
    assert sum(row['amp_skipped_steps'] for row in history) == scaler.steps // 2
