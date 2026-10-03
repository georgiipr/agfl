"""AGFL mathematics against its original layer, and every backbone x attention pair.

Development checks for the experiment machine only; they are not a prerequisite
for training. The first group compares the packaged AGFL attention with the
unchanged original layer in ``tests/references/agfl.py`` (state, output and
gradients). The second group builds each backbone with each attention on 19
electrodes and checks that the attention mixes electrodes, receives gradients
and leaves the backbone initialization untouched.

Run from the project folder: ``python -m pytest tests/test_models.py -q``.
"""
from copy import deepcopy

import pytest
import torch

from agfl_speech.attention import available_attentions
from agfl_speech.attention.agfl.config import DEFAULTS
from agfl_speech.attention.agfl.layer import AGFL, normalize_graph, propagate, topk_mask
from agfl_speech.attention.mha.layer import MultiHeadAttention
from agfl_speech.models import available_models, get_model_spec
from agfl_speech.models._shared.layers import ElectrodeReadout
from agfl_speech.models._shared.modality import validate_attention_domain, validate_model_graph
from tests.model_fixtures import small_model_options
from tests.references.agfl import AGFL as HistoricalAGFL


ATTENTIONS = ('agfl', 'mha', 'hcann')
MODELS = ('eegnet', 'eegnet_original', 'signal_transformer')
EEGNET_LAYOUTS = ('pre_spatial', 'spatial_fusion', 'compact')
CHANNELS, SAMPLES, CLASSES, BATCH, HEADS = 19, 64, 8, 4, 2


def metadata():
    return {'modality': 'eeg', 'channels': CHANNELS, 'samples': SAMPLES, 'num_classes': CLASSES}


def small_options(**extra):
    return {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2, 'depth': 2, **extra}


def attention_layers(model):
    return [module for module in model.modules() if getattr(module, 'is_attention', False)]


def routing_maps(layer):
    """[rows, heads, nodes, nodes] routing weights of the last forward pass."""
    return layer.last_adj.transpose(0, 1) if layer.attention_key == 'agfl' else layer.last_attn


@pytest.mark.parametrize('degree', [0, 1, 2, 4])
def test_default_agfl_preserves_initialization_output_and_gradient(degree):
    torch.manual_seed(57)
    old = HistoricalAGFL(8, 2, degree).double()
    torch.manual_seed(57)
    new = AGFL(small_options(K=degree)).double()
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], expected, rtol=0, atol=0)
    # Nonzero signed taps exercise graph/filter gradients rather than merely
    # comparing two zero-initialized layers' projection biases.
    with torch.no_grad():
        for head in old.filters:
            head.alpha_logits.copy_(torch.linspace(-.2, .7, degree + 1))
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 11, 8, dtype=torch.float64)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected, actual = old(a, 1, 3), new(b, 1, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(new.last_adj, old.last_adj, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)


def test_polynomial_and_renormalization_match_independent_formulas():
    a = torch.tensor([[[.8, .2], [.3, .7]]], dtype=torch.float64)
    x = torch.tensor([[[2., 1.], [1., -3.]]], dtype=torch.float64)
    taps = list(propagate(a, x, 3, 'polynomial'))
    for degree, tap in enumerate(taps):
        torch.testing.assert_close(tap, torch.linalg.matrix_power(a, degree) @ x)
    taps = list(propagate(a, x, 3, 'renormalized'))
    previous = x
    for tap in taps[1:]:
        expected = a @ previous
        expected *= x.norm(dim=-1, keepdim=True) / (expected.norm(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(tap, expected)
        previous = expected
    assert not torch.allclose(taps[-1], torch.linalg.matrix_power(a, 3) @ x)


def test_topk_ties_and_softmax_order_are_explicit():
    tied = torch.ones(1, 4, 4)
    assert topk_mask(tied, 2, 'threshold').sum() == 16
    exact = topk_mask(tied, 2, 'exact')
    assert (exact.sum(-1) == 2).all()
    scores = torch.tensor([[[1., 4., 3., 2.]]])
    before = topk_mask(scores, 2)
    after = topk_mask(scores.softmax(-1), 2)
    assert torch.equal(before, after)
    a = normalize_graph(scores, before, 'softmax', 'scores', True)
    b = normalize_graph(scores, after, 'softmax', 'softmax', True)
    torch.testing.assert_close(a, b)
    unnormalized = normalize_graph(scores, after, 'softmax', 'softmax', False)
    assert unnormalized.sum() < 1


def test_none_means_dense_topk_and_zero_coefficients_are_preserved():
    layer = AGFL(small_options(top_k=None))
    x = torch.randn(2, 5, 8)
    output = layer(x)
    assert (layer.last_adj > 0).all()
    for head in layer.filters:
        assert torch.count_nonzero(head.alpha_logits) == 0
    torch.testing.assert_close(output, layer.proj.bias.expand_as(output))


def test_mha_matches_torch_reference():
    layer = MultiHeadAttention(8, 2).double()
    reference = torch.nn.MultiheadAttention(8, 2, batch_first=True, dropout=0).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(layer.qkv.weight)
        reference.in_proj_bias.copy_(layer.qkv.bias)
        reference.out_proj.weight.copy_(layer.proj.weight)
        reference.out_proj.bias.copy_(layer.proj.bias)
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    torch.testing.assert_close(layer(x), reference(x, x, x, need_weights=True)[0])


def test_registries_hold_exactly_the_kept_backbones_and_attentions():
    assert available_models() == ['eegnet', 'eegnet_original', 'signal_transformer']
    assert available_attentions() == ['agfl', 'hcann', 'mha']
    for key in MODELS:
        assert set(get_model_spec(key).variants) == {'eeg'}
        with pytest.raises(ValueError, match='implementation for modality'):
            get_model_spec(key).defaults_for('ecg')
    eegnet = get_model_spec('eegnet').defaults_for('eeg')
    assert eegnet['temp_kernel'] == 250 and eegnet['electrode_architecture'] == 'spatial_fusion'
    assert 'temporal_statistics' not in eegnet
    transformer = get_model_spec('signal_transformer').defaults_for('eeg')
    assert transformer['eeg_kernel_size'] == 31 and 'ecg_patch_size' not in transformer


def test_no_attention_and_removed_keys_are_not_selectable():
    assert 'none' not in available_models()
    assert 'none' not in available_attentions()
    for key in ('none', 'conformer', 'eegencoder', 'dstseegencoder', 'performer', 'linformer', 'nystromformer'):
        with pytest.raises(ValueError, match='Unknown model'):
            get_model_spec(key)
    for key in ATTENTIONS:
        with pytest.raises(ValueError, match='selects an attention mechanism'):
            get_model_spec(key)


def test_spatial_readout_can_distinguish_opposite_electrode_patterns():
    readout = ElectrodeReadout(2, 1)
    with torch.no_grad():
        readout.projection.weight.copy_(torch.tensor([[[1., -1.]]]))
    tokens = torch.tensor([[[2.], [-2.]], [[-2.], [2.]]])
    torch.testing.assert_close(tokens.mean(1), torch.zeros(2, 1))
    torch.testing.assert_close(readout(tokens), torch.tensor([[4.], [-4.]]))


@pytest.mark.parametrize('key', MODELS)
@pytest.mark.parametrize('attention', ATTENTIONS)
def test_each_backbone_hosts_each_attention_across_electrodes(key, attention):
    torch.manual_seed(3)
    spec = get_model_spec(key)
    model = spec.build(small_model_options(key), metadata(), attention, {'heads': HEADS})
    assert model.token_axis == 'electrode' and model.num_tokens == CHANNELS
    validate_model_graph(model, metadata())
    layers = attention_layers(model)
    assert len(layers) == 1
    layer = layers[0]
    assert layer.attention_key == attention
    assert layer.token_axis == 'electrode' and layer.num_tokens == CHANNELS
    x = torch.randn(BATCH, CHANNELS, SAMPLES)
    logits = model(x)
    assert logits.shape == (BATCH, CLASSES)
    assert torch.isfinite(logits).all()
    # eegnet_original presents one electrode set per time step, trial-major.
    rows = BATCH * SAMPLES if key == 'eegnet_original' else BATCH
    maps = routing_maps(layer)
    assert maps.shape == (rows, HEADS, CHANNELS, CHANNELS)
    assert (maps >= 0).all()
    torch.testing.assert_close(maps.sum(-1), torch.ones(rows, HEADS, CHANNELS), rtol=0, atol=1e-5)
    torch.nn.functional.cross_entropy(logits, torch.arange(BATCH)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    reached = [p.grad.abs().sum() for p in layer.parameters() if p.grad is not None]
    assert reached and sum(reached) > 0
    with pytest.raises(ValueError, match='Expected'):
        model(x.transpose(1, 2))


@pytest.mark.parametrize('layout', EEGNET_LAYOUTS)
@pytest.mark.parametrize('attention', ATTENTIONS)
def test_every_eegnet_electrode_layout_hosts_each_attention(layout, attention):
    torch.manual_seed(5)
    options = {**small_model_options('eegnet'), 'electrode_architecture': layout}
    model = get_model_spec('eegnet').build(options, metadata(), attention, {'heads': HEADS})
    validate_model_graph(model, metadata())
    layer, = attention_layers(model)
    logits = model(torch.randn(BATCH, CHANNELS, SAMPLES))
    assert logits.shape == (BATCH, CLASSES)
    rows = BATCH * SAMPLES if layout == 'pre_spatial' else BATCH
    # pre_spatial tokens carry the f1 temporal-filter outputs; the other two
    # layouts use electrode_dim features per electrode.
    width = options['f1'] if layout == 'pre_spatial' else 32
    assert layer.options['dim'] == width
    assert routing_maps(layer).shape == (rows, HEADS, CHANNELS, CHANNELS)
    torch.nn.functional.cross_entropy(logits, torch.arange(BATCH)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.block1[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('key', MODELS)
def test_backbone_initialization_is_independent_of_attention_capacity(key):
    states = []
    for attention in ATTENTIONS:
        torch.manual_seed(41)
        model = get_model_spec(key).build(small_model_options(key), metadata(), attention, {'heads': HEADS})
        prefixes = [name + '.' for name, module in model.named_modules() if getattr(module, 'is_attention', False)]
        assert prefixes
        states.append({k: v for k, v in model.state_dict().items() if not any(k.startswith(p) for p in prefixes)})
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for name in state:
            torch.testing.assert_close(state[name], states[0][name], rtol=0, atol=0)


def test_eegnet_zero_initialized_agfl_does_not_block_feature_gradients():
    model = get_model_spec('eegnet').build(small_model_options('eegnet'), metadata(), 'agfl', {'heads': HEADS})
    loss = torch.nn.functional.cross_entropy(model(torch.randn(BATCH, CHANNELS, SAMPLES)), torch.arange(BATCH))
    loss.backward()
    assert model.block1[0].weight.grad.abs().sum() > 0
    assert model.spatial_readout.projection.weight.grad.abs().sum() > 0


def test_learned_spatial_readouts_and_the_original_spatial_filter():
    for key in ('eegnet', 'signal_transformer'):
        model = get_model_spec(key).build(small_model_options(key), metadata(), 'mha', {'heads': HEADS})
        readouts = [m for m in model.modules() if isinstance(m, ElectrodeReadout)]
        assert readouts and all(m.projection is not None for m in readouts)
    # The authors' EEGNet has no separate readout: its own depthwise spatial
    # convolution over all 19 electrodes follows the attention.
    original = get_model_spec('eegnet_original').build(
        small_model_options('eegnet_original'), metadata(), 'mha', {'heads': HEADS})
    assert not [m for m in original.modules() if isinstance(m, ElectrodeReadout)]
    assert original.spatial_readout is None
    assert original.block2[0].kernel_size == (CHANNELS, 1)


def test_eeg_attention_cannot_leave_the_electrode_axis():
    spec = get_model_spec('eegnet')
    with pytest.raises(ValueError, match='across electrodes'):
        spec.build({**small_model_options('eegnet'), 'attention_axis': 'time'}, metadata(), 'mha', {'heads': HEADS})
    with pytest.raises(ValueError, match='temporal_bias'):
        spec.build(small_model_options('eegnet'), metadata(), 'mha', {'heads': HEADS, 'temporal_bias': True})
    with pytest.raises(ValueError, match='Unknown eegnet model_options'):
        spec.build({'temporal_statistics': 'mean_logvar'}, metadata(), 'mha', {'heads': HEADS})
    with pytest.raises(ValueError, match='Only EEG'):
        validate_attention_domain('ecg', {}, {})
    with pytest.raises(ValueError, match='implementation for modality'):
        spec.build(small_model_options('eegnet'), {**metadata(), 'modality': 'ecg'}, 'mha', {'heads': HEADS})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA check for target environment')
@pytest.mark.parametrize('key', MODELS)
@pytest.mark.parametrize('attention', ATTENTIONS)
def test_cuda_autocast_keeps_gradients_finite(key, attention):
    model = get_model_spec(key).build(small_model_options(key), metadata(), attention, {'heads': HEADS}).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(torch.randn(BATCH, CHANNELS, SAMPLES, device='cuda')),
                                                torch.arange(BATCH, device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
