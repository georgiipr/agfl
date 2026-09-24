"""Cluster checks for opt-in trial-conditioned signed AGFL hop coefficients."""
from copy import deepcopy

import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL, GraphFilter
from agfl.models import get_model_spec
from tests.model_fixtures import small_model_options
from tests.references.agfl import AGFL as HistoricalAGFL


def options(**updates):
    settings = deepcopy(DEFAULTS)
    settings.update(dim=32, heads=4, K=2, projection='qkv', qkv_bias=True,
                    filter_projection='none', score_scaling='sqrt_dim',
                    coefficient_init='one_hop', top_k=5)
    settings.update(updates)
    return settings


def conditioned_filter(dim=4, **updates):
    return GraphFilter(dim, options(coefficient_conditioning='trial_power', **updates))


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
@pytest.mark.parametrize('projection', ['none', 'shared', 'separate'])
def test_zero_gate_preserves_initialization_outputs_and_shared_gradients(variant, projection):
    settings = options(agfl_variant=variant, filter_projection=projection)
    torch.manual_seed(101)
    static = AGFL(settings).double()
    static_rng = torch.get_rng_state().clone()
    torch.manual_seed(101)
    conditioned = AGFL({**settings, 'coefficient_conditioning': 'trial_power'}).double()
    assert torch.equal(torch.get_rng_state(), static_rng)
    extra_keys = set(conditioned.state_dict()) - set(static.state_dict())
    assert extra_keys == {f'filters.{head}.coefficient_gate.weight' for head in range(4)}
    assert sum(parameter.numel() for name, parameter in conditioned.named_parameters()
               if '.coefficient_gate.' in name) == 96
    for name, parameter in static.state_dict().items():
        torch.testing.assert_close(conditioned.state_dict()[name], parameter, rtol=0, atol=0)

    inputs = torch.randn(3, 7, 32, dtype=torch.float64)
    static_inputs = inputs.clone().requires_grad_()
    conditioned_inputs = inputs.clone().requires_grad_()
    expected = static(static_inputs)
    actual = conditioned(conditioned_inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(conditioned.last_adj, static.last_adj, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(conditioned_inputs.grad, static_inputs.grad, rtol=1e-10, atol=1e-11)
    for name, parameter in static.named_parameters():
        other = dict(conditioned.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(other.grad, parameter.grad, rtol=1e-10, atol=1e-11)
    for graph_filter in conditioned.filters:
        gradient = graph_filter.coefficient_gate.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_static_defaults_strictly_load_historical_checkpoint_without_new_keys():
    torch.manual_seed(103)
    historical = HistoricalAGFL(8, 2, 2)
    historical_rng = torch.get_rng_state().clone()
    # Old saved dictionaries do not contain either conditioning option.
    settings = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2}
    settings.pop('coefficient_conditioning')
    settings.pop('coefficient_conditioning_scale')
    torch.manual_seed(103)
    current = AGFL(settings)
    assert torch.equal(torch.get_rng_state(), historical_rng)
    assert current.state_dict().keys() == historical.state_dict().keys()
    current.load_state_dict(historical.state_dict(), strict=True)
    assert all(graph_filter.coefficient_gate is None for graph_filter in current.filters)


@pytest.mark.parametrize('activation', ['identity', 'relu', 'sigmoid', 'softmax'])
def test_static_coefficient_api_returns_effective_values_for_each_example(activation):
    graph_filter = GraphFilter(4, options(coefficient_activation=activation, coefficient_init='uniform'))
    values = torch.randn(3, 7, 4)
    expected = graph_filter.alpha_logits
    if activation == 'relu':
        expected = expected.relu()
    elif activation == 'sigmoid':
        expected = expected.sigmoid()
    elif activation == 'softmax':
        expected = expected.softmax(-1)
    torch.testing.assert_close(graph_filter.coefficient_values(values), expected.expand(3, -1))


def test_learned_coefficients_vary_by_trial_with_bounded_signed_corrections():
    graph_filter = conditioned_filter().double()
    with torch.no_grad():
        graph_filter.coefficient_gate.weight.copy_(torch.tensor(
            [[.5, -.2, .1, -.4], [-.2, .3, -.1, .2], [.1, -.4, .2, .3]], dtype=torch.float64))
    first = torch.tensor([[[1., 2., 4., 8.], [2., 1., 8., 4.], [1., 3., 5., 7.]]], dtype=torch.float64)
    second = first.flip(-1)
    together = torch.cat([first, second])
    coefficients = graph_filter.coefficient_values(together)
    assert not torch.allclose(coefficients[0], coefficients[1])
    correction = coefficients - graph_filter.alpha_logits
    assert torch.all(correction.abs() <= graph_filter.conditioning_scale)
    assert torch.any(correction < 0) and torch.any(correction > 0)
    # Changing a different example cannot change the first example's gate.
    torch.testing.assert_close(coefficients[:1], graph_filter.coefficient_values(first), rtol=1e-12, atol=1e-12)
    permuted = graph_filter.coefficient_values(together[:, [2, 0, 1]])
    torch.testing.assert_close(permuted, coefficients, rtol=1e-12, atol=1e-12)
    # Power pooling is insensitive to sign flips that change the signed mean.
    signs = torch.tensor([[[1., -1., 1., -1.], [-1., 1., -1., 1.], [1., 1., -1., -1.]]],
                         dtype=torch.float64)
    torch.testing.assert_close(graph_filter.coefficient_values(together * signs), coefficients, rtol=0, atol=0)
    # The nonzero, example-specific coefficients must actually drive the filter.
    graph = torch.tensor([[[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]]], dtype=torch.float64).expand(2, -1, -1)
    expected = sum(coefficients[:, hop, None, None] * (torch.linalg.matrix_power(graph, hop) @ together)
                   for hop in range(3))
    torch.testing.assert_close(graph_filter(graph, together), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize('zero_input', [False, True])
def test_descriptor_is_finite_for_zero_and_large_reduced_precision_values(device, dtype, zero_input):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA bfloat16 is unavailable')
    graph_filter = conditioned_filter().to(device=device, dtype=dtype)
    with torch.no_grad():
        graph_filter.coefficient_gate.weight.copy_(torch.tensor(
            [[.1, -.2, .3, -.1], [.2, .1, -.1, -.2], [-.1, .3, -.2, .1]], device=device, dtype=dtype))
    pattern = [0., 0., 0., 0.] if zero_input else [60000., -30000., 10000., -100.]
    values = torch.tensor([[pattern, pattern, pattern]], device=device, dtype=dtype, requires_grad=True)
    coefficients = graph_filter.coefficient_values(values)
    assert coefficients.dtype == dtype
    assert torch.isfinite(coefficients).all()
    coefficients.float().square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert torch.isfinite(graph_filter.coefficient_gate.weight.grad).all()
    if zero_input:
        torch.testing.assert_close(coefficients, graph_filter.alpha_logits.unsqueeze(0), rtol=0, atol=0)
    else:
        assert torch.count_nonzero(graph_filter.coefficient_gate.weight.grad) > 0


@pytest.mark.parametrize('updates, message', [
    ({'coefficient_conditioning': 'unknown'}, 'coefficient_conditioning must'),
    ({'coefficient_activation': 'softmax', 'coefficient_init': 'uniform'}, 'requires identity activation'),
    ({'coefficient_activation': 'sigmoid', 'coefficient_init': 'uniform'}, 'requires identity activation'),
    ({'coefficient_activation': 'relu', 'coefficient_init': 'uniform'}, 'requires identity activation'),
    ({'learnable_coefficients': False}, 'requires identity activation'),
    ({'dim': 4, 'heads': 4}, 'at least two features per head'),
])
def test_incompatible_conditioning_options_are_rejected(updates, message):
    settings = options(coefficient_conditioning='trial_power')
    settings.update(updates)
    with pytest.raises(ValueError, match=message):
        AGFL(settings)


@pytest.mark.parametrize('scale', [0., -1., float('nan'), float('inf'), True, None, '0.5'])
def test_conditioning_scale_must_be_finite_positive_number(scale):
    with pytest.raises(ValueError, match='coefficient_conditioning_scale must'):
        conditioned_filter(coefficient_conditioning_scale=scale)


@pytest.mark.parametrize('model_key', ['eegnet', 'eegencoder', 'dstseegencoder', 'conformer', 'signal_transformer'])
@pytest.mark.parametrize('modality, channels', [('eeg', 6), ('ecg', 1)])
def test_trial_conditioning_integrates_with_every_backbone(model_key, modality, channels):
    torch.manual_seed(107)
    model = get_model_spec(model_key).build(
        small_model_options(model_key, modality),
        {'modality': modality, 'channels': channels, 'samples': 65, 'num_classes': 3},
        'agfl', {'heads': 2, 'K': 2, 'projection': 'qkv', 'qkv_bias': True,
                 'filter_projection': 'none', 'score_scaling': 'sqrt_dim',
                 'coefficient_init': 'one_hop', 'coefficient_conditioning': 'trial_power'})
    logits = model(torch.randn(3, channels, 65))
    assert logits.shape == (3, 3) and torch.isfinite(logits).all()
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    gates = [module.coefficient_gate for module in model.modules() if isinstance(module, GraphFilter)]
    assert gates and all(gate is not None for gate in gates)
    assert all(gate.weight.grad is not None and torch.isfinite(gate.weight.grad).all() for gate in gates)
    assert any(torch.count_nonzero(gate.weight.grad) > 0 for gate in gates)
