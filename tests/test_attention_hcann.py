"""The ``hcann`` attention: the dataset authors' bias-free multi-head attention.

Development checks for the experiment machine only; they are not a prerequisite
for training. ``HCANNAttention`` must equal the explicit formula

    softmax(Q K^T / sqrt(d_head)) V  ->  output linear layer  ->  dropout

with one joint Q/K/V projection that has no bias, optional LayerNorm before the
projection (``pre_norm``) and optional dropout after the output layer.

Run from the project folder: ``python -m pytest tests/test_attention_hcann.py -q``.
"""
import math

import pytest
import torch
from torch.nn import functional as F

from agfl_speech.attention import AttentionFactory, available_attentions, get_attention_spec
from agfl_speech.attention.hcann.layer import HCANNAttention
from agfl_speech.attention.mha.layer import MultiHeadAttention
from agfl_speech.config import resolve_config
from agfl_speech.visualization.maps import mixer_map


DIM, HEADS, TOKENS, BATCH = 8, 2, 19, 3


def explicit_attention(layer, x):
    """Independent evaluation from the layer's weights; returns output and routing."""
    batch, tokens, dim = x.shape
    head_dim = dim // layer.heads
    if layer.norm is not None:
        x = F.layer_norm(x, (dim,), layer.norm.weight, layer.norm.bias, layer.norm.eps)
    weight = layer.to_qkv.weight  # rows [0, dim) give Q, [dim, 2 dim) K, [2 dim, 3 dim) V

    def heads(part):
        projected = x @ weight[part * dim:(part + 1) * dim].T
        return projected.reshape(batch, tokens, layer.heads, head_dim).permute(0, 2, 1, 3)
    q, k, v = heads(0), heads(1), heads(2)
    routing = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(head_dim), dim=-1)
    mixed = (routing @ v).permute(0, 2, 1, 3).reshape(batch, tokens, dim)
    return mixed @ layer.to_out[0].weight.T + layer.to_out[0].bias, routing


def test_registry_lists_exactly_the_three_attentions():
    assert available_attentions() == ['agfl', 'hcann', 'mha']
    spec = get_attention_spec('hcann')
    assert spec.key == 'hcann'
    assert spec.defaults == {'heads': 4, 'dropout': 0.0, 'pre_norm': False}
    for removed in ('performer', 'linformer', 'nystromformer'):
        with pytest.raises(ValueError, match='Unknown attention'):
            get_attention_spec(removed)


@pytest.mark.parametrize('pre_norm', [False, True])
def test_layer_equals_the_explicit_formula(pre_norm):
    torch.manual_seed(11)
    layer = HCANNAttention(DIM, HEADS, pre_norm=pre_norm).double()
    if pre_norm:
        # A non-trivial normalization makes the comparison sensitive to it.
        with torch.no_grad():
            layer.norm.weight.copy_(torch.linspace(.5, 1.5, DIM))
            layer.norm.bias.copy_(torch.linspace(-.2, .2, DIM))
    x = torch.randn(BATCH, TOKENS, DIM, dtype=torch.float64)
    expected, routing = explicit_attention(layer, x)
    actual = layer(x)
    assert actual.shape == (BATCH, TOKENS, DIM)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.last_attn, routing)
    q, k, v = layer.project(x)
    assert q.shape == k.shape == v.shape == (BATCH, HEADS, TOKENS, DIM // HEADS)
    torch.testing.assert_close(layer.routing(q, k), routing)


def test_projection_has_no_bias_and_state_keys_follow_the_options():
    plain = HCANNAttention(DIM, HEADS)
    assert plain.to_qkv.bias is None
    assert plain.norm is None
    assert set(plain.state_dict()) == {'to_qkv.weight', 'to_out.0.weight', 'to_out.0.bias'}
    assert plain.to_qkv.weight.shape == (3 * DIM, DIM)
    assert plain.to_out[0].weight.shape == (DIM, DIM)
    assert plain.options == {'dim': DIM, 'heads': HEADS, 'dropout': 0.0, 'pre_norm': False}
    normalized = HCANNAttention(DIM, HEADS, pre_norm=True, dropout=0.25)
    assert normalized.to_qkv.bias is None
    assert set(normalized.state_dict()) == {'norm.weight', 'norm.bias', 'to_qkv.weight',
                                            'to_out.0.weight', 'to_out.0.bias'}
    assert normalized.to_out[1].p == 0.25
    assert normalized.options == {'dim': DIM, 'heads': HEADS, 'dropout': 0.25, 'pre_norm': True}


def test_last_attn_rows_are_detached_probabilities():
    layer = HCANNAttention(DIM, HEADS)
    x = torch.randn(BATCH, TOKENS, DIM, requires_grad=True)
    output = layer(x)
    routing = layer.last_attn
    assert routing.shape == (BATCH, HEADS, TOKENS, TOKENS)
    assert not routing.requires_grad
    assert (routing >= 0).all()
    torch.testing.assert_close(routing.sum(-1), torch.ones(BATCH, HEADS, TOKENS), rtol=0, atol=1e-5)
    output.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert layer.to_qkv.weight.grad.abs().sum() > 0


def test_pre_norm_is_layer_normalization_of_the_input():
    torch.manual_seed(2)
    plain = HCANNAttention(DIM, HEADS).double()
    normalized = HCANNAttention(DIM, HEADS, pre_norm=True).double()
    missing, unexpected = normalized.load_state_dict(plain.state_dict(), strict=False)
    assert sorted(missing) == ['norm.bias', 'norm.weight'] and not unexpected
    x = 3 * torch.randn(BATCH, TOKENS, DIM, dtype=torch.float64) + 1
    torch.testing.assert_close(normalized(x), plain(normalized.norm(x)))
    assert not torch.allclose(normalized(x), plain(x))


def test_dropout_follows_the_output_projection_and_is_inactive_in_evaluation():
    torch.manual_seed(4)
    layer = HCANNAttention(DIM, HEADS, dropout=0.5)
    x = torch.randn(BATCH, TOKENS, DIM)
    expected, _ = explicit_attention(layer, x)
    layer.eval()
    torch.testing.assert_close(layer(x), expected, rtol=1e-5, atol=1e-6)
    layer.train()
    output = layer(x)
    dropped = output == 0
    assert dropped.any() and not dropped.all()
    # Surviving outputs are scaled by 1 / (1 - p); the routing is not dropped.
    torch.testing.assert_close(output[~dropped], 2 * expected[~dropped], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(layer.last_attn.sum(-1), torch.ones(BATCH, HEADS, TOKENS), rtol=0, atol=1e-5)


def test_it_is_mha_without_the_qkv_bias():
    torch.manual_seed(6)
    hcann = HCANNAttention(DIM, HEADS).double()
    mha = MultiHeadAttention(DIM, HEADS).double()
    with torch.no_grad():
        mha.qkv.weight.copy_(hcann.to_qkv.weight)
        mha.qkv.bias.zero_()
        mha.proj.weight.copy_(hcann.to_out[0].weight)
        mha.proj.bias.copy_(hcann.to_out[0].bias)
    x = torch.randn(BATCH, TOKENS, DIM, dtype=torch.float64)
    torch.testing.assert_close(hcann(x), mha(x))
    torch.testing.assert_close(hcann.last_attn, mha.last_attn)


@pytest.mark.parametrize('arguments,message', [
    ((8, 3), 'divisible'), ((8, 0), 'divisible'), ((8, 2.0), 'divisible'), ((8.0, 2), 'divisible'),
])
def test_heads_must_divide_the_dimension(arguments, message):
    with pytest.raises(ValueError, match=message):
        HCANNAttention(*arguments)


@pytest.mark.parametrize('options,message', [
    ({'pre_norm': 'yes'}, 'pre_norm'), ({'pre_norm': 1}, 'pre_norm'),
    ({'dropout': 1.0}, 'dropout'), ({'dropout': -0.1}, 'dropout'),
    ({'dropout': float('nan')}, 'dropout'), ({'dropout': True}, 'dropout'), ({'dropout': '0.1'}, 'dropout'),
])
def test_invalid_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        HCANNAttention(DIM, HEADS, **options)


def test_spec_builds_the_layer_and_rejects_unknown_or_indivisible_options():
    spec = get_attention_spec('hcann')
    layer = spec.build({'heads': 2, 'pre_norm': True, 'dropout': 0.1}, DIM, TOKENS)
    assert isinstance(layer, HCANNAttention)
    assert layer.options == {'dim': DIM, 'heads': 2, 'dropout': 0.1, 'pre_norm': True}
    assert spec.build({}, DIM, TOKENS).options == {'dim': DIM, 'heads': 4, 'dropout': 0.0, 'pre_norm': False}
    with pytest.raises(ValueError, match='Unknown hcann attention_options'):
        spec.build({'landmarks': 4}, DIM, TOKENS)
    with pytest.raises(ValueError, match='must divide'):
        spec.build({'heads': 3}, DIM, TOKENS)


def test_configuration_accepts_hcann_options_and_rejects_those_of_other_attentions():
    config = resolve_config({'attention': 'hcann', 'attention_options': {'pre_norm': True, 'dropout': 0.1}})
    assert config['attention'] == 'hcann'
    assert config['attention_options'] == {'heads': 4, 'dropout': 0.1, 'pre_norm': True}
    assert resolve_config({'attention': 'hcann'})['attention_options'] == {'heads': 4, 'dropout': 0.0, 'pre_norm': False}
    for options in ({'K': 2}, {'temporal_bias': False}, {'output_gate': False}, {'landmarks': 8}):
        with pytest.raises(ValueError, match='Unknown options for attention hcann'):
            resolve_config({'attention': 'hcann', 'attention_options': options})
    with pytest.raises(ValueError, match='heads'):
        resolve_config({'attention': 'hcann', 'attention_options': {'heads': 0}})


def test_factory_marks_an_electrode_attention_without_consuming_the_global_generator():
    options = {'heads': HEADS, 'dropout': 0.0, 'pre_norm': True}
    torch.manual_seed(5)
    untouched = torch.rand(3)
    torch.manual_seed(5)
    factory = AttentionFactory('hcann', options, modality='eeg', channels=TOKENS)
    layer = factory(DIM, TOKENS, token_axis='electrode')
    torch.testing.assert_close(torch.rand(3), untouched, rtol=0, atol=0)
    assert isinstance(layer, HCANNAttention)
    assert layer.attention_key == 'hcann' and layer.is_attention is True
    assert layer.num_tokens == TOKENS and layer.token_axis == 'electrode'
    assert (layer.layer_idx, layer.depth) == (0, 1)
    assert layer.options == {'dim': DIM, 'heads': HEADS, 'dropout': 0.0, 'pre_norm': True}
    # The same seed and position give the same weights.
    torch.manual_seed(5)
    again = AttentionFactory('hcann', options, modality='eeg', channels=TOKENS)(DIM, TOKENS, token_axis='electrode')
    for name, value in layer.state_dict().items():
        torch.testing.assert_close(again.state_dict()[name], value, rtol=0, atol=0)
    with pytest.raises(ValueError, match='one graph node per electrode'):
        factory(DIM, TOKENS, token_axis='time')
    with pytest.raises(ValueError, match='one graph node per electrode'):
        factory(DIM, TOKENS + 1, token_axis='electrode')


@pytest.mark.parametrize('pre_norm', [False, True])
def test_mixer_map_returns_the_routing_of_the_same_input(pre_norm):
    torch.manual_seed(8)
    layer = HCANNAttention(DIM, HEADS, pre_norm=pre_norm, dropout=0.5).eval()
    x = torch.randn(BATCH, TOKENS, DIM)
    with torch.no_grad():
        layer(x)
        maps, kind = mixer_map(layer, x)
    assert 'HCANN' in kind
    assert maps.shape == (BATCH, HEADS, TOKENS, TOKENS)
    torch.testing.assert_close(maps, layer.last_attn)
    torch.testing.assert_close(maps.sum(-1), torch.ones(BATCH, HEADS, TOKENS), rtol=0, atol=1e-5)
    omitted, reason = mixer_map(layer, x, max_tokens=TOKENS - 1)
    assert omitted is None and 'omitted' in reason
