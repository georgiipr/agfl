"""Target-machine checks for retired power-study checkpoint reproduction.

These development checks are deliberately separate from experiment launches.
"""
from copy import deepcopy

import pytest
import torch
from torch.nn import functional as F

from agfl.attention import AttentionFactory
from agfl.models import get_model_spec
from agfl.models.eegnet.backbone import EEGNetBackbone
from agfl.models.eegnet.reproduction import TemporalStatisticsFusion, EEGPowerReproduction, ECGPowerReproduction
from agfl.models.eegnet.config import EEG_DEFAULTS
from agfl.models.eegnet.eeg import EEGModel
from agfl.models.eegnet.ecg import ECGModel


def small_options(**changes):
    return {**{k: v for k, v in deepcopy(EEG_DEFAULTS).items()
               if k not in {'electrode_dim', 'electrode_architecture'}}, 'attention_axis': 'time', 'temp_kernel': 9,
            'f1': 4, 'd': 2, 'f2': 8, 'pk1': 2, 'pk2': 4,
            'dropout_rate': 0., **changes}


def metadata(modality='eeg'):
    return {'modality': modality, 'channels': 3, 'samples': 65, 'num_classes': 4}


def build(options, attention='agfl', modality='eeg'):
    return get_model_spec('eegnet').rebuild_saved(
        options, metadata(modality), attention, {'heads': 2})


@pytest.mark.parametrize('modality', ['eeg', 'ecg'])
def test_active_eegnet_uses_original_backbone_and_archived_power_is_explicit(modality):
    restored = build(small_options(), modality=modality)
    assert type(restored) is (EEGModel if modality == 'eeg' else ECGModel)
    assert restored.convolve.__func__ is EEGNetBackbone.convolve
    assert not hasattr(restored, 'temporal_statistics')
    assert not any(name.startswith('temporal_statistics.') for name in restored.state_dict())
    archived = build(small_options(temporal_statistics='mean_logvar'), modality=modality)
    assert type(archived) is (EEGPowerReproduction if modality == 'eeg' else ECGPowerReproduction)
    assert archived.model_variant == restored.model_variant == modality
    assert set(archived.state_dict()) - set(restored.state_dict()) == {'temporal_statistics.projection.weight'}


@pytest.mark.parametrize('initialization', ['pytorch', 'xavier'])
def test_historical_default_has_no_new_keys_and_keeps_rng(initialization):
    options = small_options(initialization=initialization)
    historical = deepcopy(options)
    historical.pop('temporal_statistics')
    torch.manual_seed(42)
    old = EEGModel(historical, metadata(), AttentionFactory('agfl', {'heads': 2}, historical=True)).double().eval()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(42)
    current = build(options).double().eval()
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert not hasattr(old, 'temporal_statistics') and not hasattr(current, 'temporal_statistics')
    assert old.state_dict().keys() == current.state_dict().keys()
    assert not any('temporal_statistics' in key for key in current.state_dict())
    current.load_state_dict(old.state_dict(), strict=True)
    x = torch.randn(2, 3, 65, dtype=torch.float64)
    # Independent original time-mode feature/attention/readout expression.
    encoded = old.block3(old.block2(old.block1(x.unsqueeze(1)))).squeeze(2).transpose(1, 2)
    original = old.fc((encoded + old.attn_blocks[0](encoded)).flatten(1))
    torch.testing.assert_close(current(x), original, rtol=0, atol=0)


@pytest.mark.parametrize('initialization', ['pytorch', 'xavier'])
@pytest.mark.parametrize('axis', ['time', 'electrode'])
def test_zero_projection_preserves_backbone_initialization_outputs_and_gradients(initialization, axis):
    options = small_options(initialization=initialization, attention_axis=axis)
    torch.manual_seed(19)
    old = build(options).double().eval()
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(19)
    new = build({**options, 'temporal_statistics': 'mean_logvar'}).double().eval()
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert set(new.state_dict()) - set(old.state_dict()) == {'temporal_statistics.projection.weight'}
    for name, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[name], rtol=0, atol=0)
    x = torch.randn(3, 3, 65, dtype=torch.float64)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected, actual = old(a), new(b)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    F.cross_entropy(expected, torch.tensor([0, 1, 2])).backward()
    F.cross_entropy(actual, torch.tensor([0, 1, 2])).backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    new_parameters = dict(new.named_parameters())
    for name, parameter in old.named_parameters():
        if parameter.grad is None:
            assert new_parameters[name].grad is None, name
        else:
            torch.testing.assert_close(parameter.grad, new_parameters[name].grad,
                                       rtol=1e-12, atol=1e-12)
    gradient = new.temporal_statistics.projection.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_bins_cover_every_sample_and_match_population_variance():
    layer = TemporalStatisticsFusion(2, 2, 3, 11).double()
    values = torch.tensor([[[[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 25.]],
                            [[4., 1., 4., 1., 4., 1., 4., 1., 4., 1., 4.]]]], dtype=torch.float64)
    assert layer.bin_edges == (0, 3, 7, 11)
    assert [index for start, end in zip(layer.bin_edges[:-1], layer.bin_edges[1:])
            for index in range(start, end)] == list(range(11))
    expected = torch.stack([values[..., start:end].var(dim=-1, unbiased=False)
                            for start, end in ((0, 3), (3, 7), (7, 11))], dim=-1)
    torch.testing.assert_close(layer.log_variance(values), (expected + 1e-6).log())
    before = layer.log_variance(values)
    changed = values.clone()
    changed[..., -1] += 50
    after = layer.log_variance(changed)
    torch.testing.assert_close(before[..., :2], after[..., :2], rtol=0, atol=0)
    assert not torch.allclose(before[..., -1], after[..., -1])


def test_statistics_path_does_not_repeat_batchnorm_or_change_dropout_rng():
    options = small_options(dropout_rate=.4)
    torch.manual_seed(29)
    original = build(options).train()
    torch.manual_seed(29)
    extended = build({**options, 'temporal_statistics': 'mean_logvar'}).train()
    values = torch.randn(4, 3, 65)
    torch.manual_seed(137)
    expected = original(values)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(137)
    actual = extended(values)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), old_rng)
    buffers = dict(extended.named_buffers())
    for name, buffer in original.named_buffers():
        torch.testing.assert_close(buffers[name], buffer, rtol=0, atol=0)
        if name.endswith('num_batches_tracked'):
            assert int(buffer) == 1


def test_feature_power_contrasts_reach_the_fused_output_without_cross_trial_statistics():
    layer = TemporalStatisticsFusion(4, 4, 2, 12).double()
    with torch.no_grad():
        layer.projection.weight.copy_(torch.eye(4, dtype=torch.float64))
    values = torch.arange(12, dtype=torch.float64).reshape(1, 1, 1, 12).expand(2, 4, 1, 12).clone()
    values *= torch.tensor([1., 2., 3., 5.], dtype=torch.float64)[None, :, None, None]
    expected = F.layer_norm(layer.log_variance(values).movedim(1, -1), (4,), eps=1e-5).movedim(-1, 1)
    torch.testing.assert_close(layer(values), expected)
    modified = values.clone()
    modified[0, 0] *= 3
    assert not torch.allclose(layer(values)[0], layer(modified)[0])
    torch.testing.assert_close(layer(values)[1], layer(modified)[1], rtol=0, atol=0)
    # Additive offsets carry no population-variance information.
    torch.testing.assert_close(layer(values + 10.), layer(values), rtol=1e-12, atol=1e-12)
    values.requires_grad_()
    layer(values).square().sum().backward()
    assert torch.isfinite(values.grad).all() and torch.count_nonzero(values.grad) > 0


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_constant_and_large_half_precision_inputs_stay_finite(dtype):
    layer = TemporalStatisticsFusion(4, 4, 3, 12).to(dtype=dtype)
    with torch.no_grad():
        layer.projection.weight.copy_(torch.eye(4, dtype=dtype))
    large = torch.tensor([-60000., 60000.] * 6, dtype=dtype).reshape(1, 1, 1, 12).expand(2, 4, 1, 12).clone()
    large *= torch.tensor([.25, .5, .75, 1.], dtype=dtype)[None, :, None, None]
    for values in (torch.zeros(2, 4, 1, 12, dtype=dtype), large):
        values.requires_grad_()
        result = layer(values)
        assert result.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
        assert torch.isfinite(result).all()
        result.square().sum().backward()
        assert torch.isfinite(values.grad).all()
        assert torch.isfinite(layer.projection.weight.grad).all()


@pytest.mark.parametrize('attention', ['agfl', 'mha', 'performer', 'linformer', 'nystromformer'])
@pytest.mark.parametrize('modality,axis', [('eeg', 'time'), ('eeg', 'electrode'), ('ecg', 'time')])
def test_every_attention_receives_fused_tokens_and_keeps_checkpoint_hooks(attention, modality, axis):
    model = build(small_options(attention_axis=axis, temporal_statistics='mean_logvar'), attention, modality)
    with torch.no_grad():
        model.temporal_statistics.projection.weight.fill_(.1)
        model.temporal_statistics.projection.weight.diagonal().fill_(.3)
    model.eval()
    x = torch.randn(2, 3, 65)
    captured = []
    hook = model.attn_blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.no_grad():
            model(x)
            weight = model.temporal_statistics.projection.weight.clone()
            model.temporal_statistics.projection.weight.zero_()
            model(x)
            model.temporal_statistics.projection.weight.copy_(weight)
    finally:
        hook.remove()
    logits = model(x)
    assert logits.shape == (2, 4)
    assert model.classifier is model.fc and len(captured) == 2
    assert captured[0].shape[1] == (3 if axis == 'electrode' else 8)
    assert not torch.allclose(captured[0], captured[1])
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert model.temporal_statistics.projection.weight.grad is not None
    assert torch.isfinite(model.temporal_statistics.projection.weight.grad).all()
    replay = build(small_options(attention_axis=axis, temporal_statistics='mean_logvar'), attention, modality).eval()
    replay.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(replay(x), logits, rtol=0, atol=0)


def test_full_recipe_adds_only_one_small_projection_and_keeps_seven_tokens():
    options = {**{k: v for k, v in deepcopy(EEG_DEFAULTS).items()
                 if k not in {'electrode_dim', 'electrode_architecture'}}, 'attention_axis': 'time', 'temp_kernel': 125}
    data = {'modality': 'eeg', 'channels': 22, 'samples': 1000, 'num_classes': 4}
    old = get_model_spec('eegnet').rebuild_saved(options, data, 'agfl', {'heads': 4})
    new = get_model_spec('eegnet').rebuild_saved({**options, 'temporal_statistics': 'mean_logvar'}, data, 'agfl', {'heads': 4})
    assert new.num_tokens == old.num_tokens == 7
    assert new.fc.in_features == old.fc.in_features == 224
    assert new.temporal_statistics.bin_edges == (0, 142, 285, 428, 571, 714, 857, 1000)
    assert sum(parameter.numel() for parameter in new.parameters()) - sum(parameter.numel() for parameter in old.parameters()) == 1024


@pytest.mark.parametrize('kwargs', [dict(features=1), dict(bins=7), dict(samples=0)])
def test_degenerate_statistics_are_rejected(kwargs):
    with pytest.raises(ValueError, match='at least two'):
        TemporalStatisticsFusion(**{'features': 4, 'out_features': 4, 'bins': 3, 'samples': 12, **kwargs})


def test_unknown_statistics_mode_is_rejected():
    with pytest.raises(ValueError, match='temporal_statistics'):
        build(small_options(temporal_statistics='unsupported'))
