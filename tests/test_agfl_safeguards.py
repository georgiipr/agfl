"""Target-machine checks for unusable filters and reduced-precision graphs."""
from copy import deepcopy
import math

import pytest
import torch

from agfl_speech.attention.agfl.config import DEFAULTS
from agfl_speech.attention.agfl.layer import AGFL, normalize_graph


def options(**updates):
    settings = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2}
    settings.update(updates)
    return settings


def require_graph_device(device, dtype):
    if device == 'cuda':
        if not torch.cuda.is_available():
            pytest.skip('CUDA is unavailable')
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip('This CUDA device does not support bfloat16')


@pytest.mark.parametrize('learnable', [True, False])
def test_zero_initialized_relu_filter_is_rejected(learnable):
    with pytest.raises(ValueError, match='Zero-initialized ReLU coefficients cannot learn'):
        AGFL(options(coefficient_activation='relu', learnable_coefficients=learnable))


def test_frozen_zero_identity_filter_is_rejected():
    with pytest.raises(ValueError, match='Frozen zero coefficients disable graph filtering'):
        AGFL(options(learnable_coefficients=False))


@pytest.mark.parametrize('learnable', [0, 1, None, 'false', 'true'])
def test_coefficient_learning_flag_requires_a_boolean(learnable):
    with pytest.raises(ValueError, match='learnable_coefficients must be a boolean'):
        AGFL(options(learnable_coefficients=learnable))


@pytest.mark.parametrize('activation', ['sigmoid', 'softmax'])
@pytest.mark.parametrize('learnable', [True, False])
def test_zero_logits_with_nonzero_effective_coefficients_remain_supported(activation, learnable):
    model = AGFL(options(coefficient_activation=activation, learnable_coefficients=learnable))
    graph_filter = model.filters[0]
    # No projection makes the expected scale observable independently of its
    # random weights. Identity adjacency preserves every hop of constant values.
    graph_filter.W = torch.nn.Identity()
    graph_filter.projection = 'none'
    graph = torch.eye(3).unsqueeze(0)
    values = torch.ones(1, 3, 4)
    expected = 1.5 if activation == 'sigmoid' else 1.
    torch.testing.assert_close(graph_filter(graph, values), values * expected)
    assert graph_filter.alpha_logits.requires_grad is learnable


@pytest.mark.parametrize('activation', ['identity', 'relu'])
@pytest.mark.parametrize('initialization', ['uniform', 'lower_order'])
def test_nonzero_frozen_coefficients_remain_supported(activation, initialization):
    model = AGFL(options(coefficient_activation=activation, coefficient_init=initialization,
                         learnable_coefficients=False))
    for graph_filter in model.filters:
        assert not graph_filter.alpha_logits.requires_grad
        assert torch.all(graph_filter.alpha_logits > 0)


def test_historical_learnable_zero_identity_filter_remains_supported():
    model = AGFL(options())
    assert all(head.alpha_logits.requires_grad for head in model.filters)
    assert all(torch.count_nonzero(head.alpha_logits) == 0 for head in model.filters)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('method', ['row', 'symmetric'])
@pytest.mark.parametrize('fill', [0., -1.])
def test_empty_reduced_precision_graph_rows_have_finite_zero_gradients(device, dtype, method, fill):
    require_graph_device(device, dtype)
    scores = torch.full((1, 3, 3), fill, dtype=dtype, device=device, requires_grad=True)
    graph = normalize_graph(scores, torch.ones_like(scores, dtype=torch.bool), method, 'scores', True)
    assert graph.dtype == dtype
    torch.testing.assert_close(graph, torch.zeros_like(graph), rtol=0, atol=0)
    graph.float().sum().backward()
    assert scores.grad is not None
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores), rtol=0, atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@pytest.mark.parametrize('method', ['row', 'symmetric'])
def test_reduced_precision_graph_matches_normalized_positive_weights(device, dtype, method):
    require_graph_device(device, dtype)
    # The third node is isolated, while the first two exercise nonzero degrees.
    scores = torch.tensor([[[2., 1., -1.], [1., 2., -1.], [-1., -1., 0.]]],
                          dtype=dtype, device=device, requires_grad=True)
    graph = normalize_graph(scores, torch.ones_like(scores, dtype=torch.bool), method, 'scores', True)
    expected = torch.tensor([[[2 / 3, 1 / 3, 0.], [1 / 3, 2 / 3, 0.], [0., 0., 0.]]],
                            dtype=dtype, device=device)
    torch.testing.assert_close(graph, expected, rtol=0, atol=0)
    (graph.float() * torch.arange(9, device=device).reshape(1, 3, 3)).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
def test_post_softmax_normalizes_before_small_weights_are_rounded_away(device, dtype):
    require_graph_device(device, dtype)
    scores = torch.tensor([[[0., -20., -22.]]], dtype=dtype, device=device, requires_grad=True)
    mask = torch.tensor([[[False, True, True]]], device=device)
    graph = normalize_graph(scores, mask, 'softmax', 'softmax', True)
    expected = torch.tensor([[[0., 1 / (1 + math.exp(-2.)),
                               1 / (1 + math.exp(2.))]]], dtype=dtype, device=device)
    torch.testing.assert_close(graph, expected, rtol=0, atol=0)
    graph[..., 1].float().sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) > 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('method', ['row', 'symmetric'])
def test_half_precision_positive_degree_does_not_overflow(device, method):
    require_graph_device(device, torch.float16)
    scores = torch.full((1, 2, 2), 60000., dtype=torch.float16, device=device, requires_grad=True)
    graph = normalize_graph(scores, torch.ones_like(scores, dtype=torch.bool), method, 'scores', True)
    torch.testing.assert_close(graph, torch.full_like(graph, .5), rtol=0, atol=0)
    (graph.float() * torch.tensor([[[0., 1.], [2., 3.]]], device=device)).sum().backward()
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad) > 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('method', ['row', 'symmetric', 'softmax'])
def test_full_precision_normalization_preserves_existing_formula(dtype, method):
    scores = torch.tensor([[[2., -1.], [.5, 1.]]], dtype=dtype)
    mask = torch.tensor([[[True, False], [True, True]]])
    if method == 'softmax':
        expected = scores.softmax(-1) * mask
        expected = expected / expected.sum(-1, keepdim=True).clamp_min(1e-12)
    else:
        expected = scores.relu() * mask
        if method == 'row':
            expected = expected / expected.sum(-1, keepdim=True).clamp_min(1e-12)
        else:
            expected = (expected + expected.transpose(-1, -2)) / 2
            inverse = expected.sum(-1).clamp_min(1e-12).rsqrt()
            expected = inverse.unsqueeze(-1) * expected * inverse.unsqueeze(-2)
    actual = normalize_graph(scores, mask, method, 'softmax' if method == 'softmax' else 'scores', True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
