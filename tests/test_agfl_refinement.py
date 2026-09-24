"""Target-machine checks for opt-in AGFL graph/filter initialization.

These checks use synthetic inputs and do not establish trained accuracy gains.
Do not execute them in the local workspace.
"""
from copy import deepcopy
import math

import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL, GraphConstructor, GraphFilter
from agfl.models import get_model_spec
from tests.references.agfl import AGFL as HistoricalAGFL


def options(**updates):
    settings = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2,
                'projection': 'qkv', 'qkv_bias': True,
                'filter_projection': 'none', 'score_scaling': 'sqrt_dim',
                'coefficient_init': 'one_hop', 'top_k': 3}
    settings.update(updates)
    return settings


def test_default_and_omitted_temperature_preserve_historical_state_and_rng():
    old_options = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2}
    old_options.pop('temperature_init')
    torch.manual_seed(71)
    historical = HistoricalAGFL(dim=8, heads=2, K=2).double()
    expected_rng = torch.get_rng_state().clone()

    models = []
    for settings in (old_options, {**old_options, 'temperature_init': 1.0}):
        torch.manual_seed(71)
        current = AGFL(settings).double()
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert current.state_dict().keys() == historical.state_dict().keys()
        for name, expected in historical.state_dict().items():
            torch.testing.assert_close(current.state_dict()[name], expected, rtol=0, atol=0)
        models.append(current)

    # Exercise a learned nonzero filter and nondefault saved temperatures,
    # rather than checking only the trivial zero-initialized branch.
    with torch.no_grad():
        for head, (builder, graph_filter) in enumerate(zip(historical.builders, historical.filters)):
            builder.temperature.fill_(0.6 + head * 0.2)
            graph_filter.alpha_logits.copy_(torch.tensor([0.2, 0.6, -0.1], dtype=torch.float64))
    inputs = torch.randn(2, 7, 8, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(73))
    expected = historical(inputs, 0, 1)
    for current in models:
        current.load_state_dict(historical.state_dict(), strict=True)
        torch.testing.assert_close(current(inputs), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('value', [True, False, None, '0.5', float('nan'),
                                  float('inf'), -float('inf'), 0, 0.09, 5.01,
                                  pytest.param(10**400, id='out_of_float_range_integer')])
def test_temperature_initialization_rejects_invalid_values(value):
    with pytest.raises(ValueError, match='temperature_init must be a finite number'):
        AGFL(options(score_scaling='temperature', temperature_init=value))


@pytest.mark.parametrize('scaling', ['sqrt_dim', 'raw'])
def test_nondefault_temperature_requires_active_temperature_scaling(scaling):
    with pytest.raises(ValueError, match='Nondefault temperature_init requires'):
        AGFL(options(score_scaling=scaling, temperature_init=0.5))


@pytest.mark.parametrize('value', [0.1, 0.5, 1, 1.0, 5])
def test_valid_temperature_initialization_is_a_floating_scalar(value):
    builder = GraphConstructor(4, 'temperature', value)
    assert builder.temperature.shape == torch.Size([])
    assert builder.temperature.is_floating_point()
    assert builder.temperature.requires_grad
    assert builder.temperature.item() == pytest.approx(value)


def test_sharper_temperature_preserves_support_and_matches_independent_formula_gradient():
    q = torch.tensor([[[0.2, 0.4], [0.8, -0.3], [-0.4, 0.6], [0.1, -0.9]]],
                     dtype=torch.float64)
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-0.8, 0.2], [0.3, -0.7]]],
                     dtype=torch.float64)
    cold = AGFL(options(dim=4, score_scaling='temperature', temperature_init=0.5)).double()
    warm = AGFL(options(dim=4, score_scaling='temperature', temperature_init=1.0)).double()
    cold_graph = cold.construct_graph(0, q, k, 0, 1)
    warm_graph = warm.construct_graph(0, q, k, 0, 1)
    assert torch.equal(cold_graph > 0, warm_graph > 0)
    assert torch.all((cold_graph > 0).sum(-1) == 3)

    raw = q @ k.transpose(-1, -2)
    support = torch.zeros_like(raw, dtype=torch.bool)
    support.scatter_(-1, raw.argsort(dim=-1, descending=True)[..., :3], True)
    reference_temperature = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    expected = (raw / (math.sqrt(2) * reference_temperature)).masked_fill(
        ~support, float('-inf')).softmax(-1)
    torch.testing.assert_close(cold_graph, expected, rtol=1e-12, atol=1e-12)

    def entropy(graph):
        return -(graph * graph.clamp_min(1e-30).log()).sum(-1)

    assert torch.all(entropy(cold_graph) < entropy(warm_graph))
    weights = torch.arange(16, dtype=torch.float64).reshape(1, 4, 4)
    (cold_graph * weights).sum().backward()
    (expected * weights).sum().backward()
    actual_gradient = cold.builders[0].temperature.grad
    assert actual_gradient is not None and torch.isfinite(actual_gradient)
    assert actual_gradient.abs() > 0
    torch.testing.assert_close(actual_gradient, reference_temperature.grad, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('degree', [1, 2, 4])
def test_identity_one_hop_formula_preserves_local_values_and_all_taps_can_learn(degree):
    graph_filter = GraphFilter(2, options(K=degree, coefficient_init='identity_one_hop')).double()
    graph = torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]],
                         dtype=torch.float64, requires_grad=True)
    values = torch.arange(1, 7, dtype=torch.float64).reshape(1, 3, 2).requires_grad_()
    expected_coefficients = torch.zeros(degree + 1, dtype=torch.float64)
    expected_coefficients[:2] = 0.5
    torch.testing.assert_close(graph_filter.alpha_logits, expected_coefficients, rtol=0, atol=0)
    actual = graph_filter(graph, values)
    torch.testing.assert_close(actual, 0.5 * values + 0.5 * (graph @ values), rtol=0, atol=0)
    actual.sum().backward()
    for gradient in (graph_filter.alpha_logits.grad, graph.grad, values.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.all(gradient != 0)


@pytest.mark.parametrize('updates', [{'K': 0}, {'coefficient_activation': 'relu'},
                                     {'coefficient_activation': 'sigmoid'},
                                     {'coefficient_activation': 'softmax'}])
def test_identity_one_hop_requires_first_hop_and_identity_activation(updates):
    with pytest.raises(ValueError, match='identity_one_hop initialization requires'):
        AGFL(options(coefficient_init='identity_one_hop', **updates))


def test_refinement_keeps_eegnet_and_shared_projection_initialization_fixed():
    spec = get_model_spec('eegnet')
    model_options = {'attention_axis': 'time', 'f1': 4, 'd': 2, 'f2': 8,
                     'temp_kernel': 7, 'pk1': 4, 'pk2': 4}
    metadata = {'modality': 'eeg', 'channels': 6, 'samples': 112, 'num_classes': 4}
    control_options = options(heads=4, top_k='scheduled')
    control_options.pop('dim')
    refined_options = {**control_options, 'score_scaling': 'temperature',
                       'temperature_init': 0.5, 'coefficient_init': 'identity_one_hop'}
    torch.manual_seed(79)
    control = spec.build(model_options, metadata, 'agfl', control_options).double().eval()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(79)
    refined = spec.build(model_options, metadata, 'agfl', refined_options).double().eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert control.num_tokens == refined.num_tokens == 7
    initial_control, initial_refined = control.state_dict(), refined.state_dict()
    assert initial_control.keys() == initial_refined.keys()
    for name, expected in initial_control.items():
        if name.endswith('.temperature') or name.endswith('.alpha_logits'):
            continue
        torch.testing.assert_close(initial_refined[name], expected, rtol=0, atol=0)

    inputs = torch.randn(2, 6, 112, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(83), requires_grad=True)
    logits = refined(inputs)
    assert logits.shape == (2, 4) and torch.isfinite(logits).all()
    upstream = torch.tensor([[1., -0.5, 0.3, -0.2], [-0.4, 0.7, -0.8, 0.6]], dtype=torch.float64)
    (logits * upstream).sum().backward()
    layer = refined.attn_blocks[0]
    temperatures = torch.stack([builder.temperature.grad for builder in layer.builders])
    coefficients = torch.stack([graph_filter.alpha_logits.grad for graph_filter in layer.filters])
    assert torch.isfinite(temperatures).all() and temperatures.abs().sum() > 0
    assert torch.isfinite(coefficients).all() and coefficients[:, 2:].abs().sum() > 0
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_saved_refinement_weights_override_constructor_initialization():
    old = AGFL(options(score_scaling='temperature', temperature_init=0.5,
                       coefficient_init='identity_one_hop')).double()
    with torch.no_grad():
        old.builders[0].temperature.fill_(0.7)
        old.filters[0].alpha_logits.copy_(torch.tensor([0.4, 0.8, -0.2], dtype=torch.float64))
    restored = AGFL(options(score_scaling='temperature', temperature_init=1.0,
                            coefficient_init='one_hop')).double()
    restored.load_state_dict(old.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 8, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(89))
    torch.testing.assert_close(restored(inputs), old(inputs), rtol=0, atol=0)
