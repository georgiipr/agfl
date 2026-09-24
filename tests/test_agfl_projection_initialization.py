"""Target-machine checks for AGFL's optional identity hop projections.

Do not run these tests in the local workspace. Initial equivalence and gradient
checks do not establish an improvement in trained accuracy.
"""
from copy import deepcopy

import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL, GraphFilter
from agfl.models import get_model_spec
from tests.references.agfl import AGFL as HistoricalAGFL


def options(**updates):
    settings = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2, 'K': 2,
                'projection': 'qkv', 'qkv_bias': True,
                'score_scaling': 'sqrt_dim', 'coefficient_init': 'one_hop',
                'filter_projection': 'none', 'top_k': 5}
    settings.update(updates)
    return settings


@pytest.mark.parametrize('projection', ['shared', 'separate'])
@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
def test_identity_projections_match_value_only_initial_outputs_rng_and_shared_gradients(projection, variant):
    torch.manual_seed(101)
    baseline = AGFL(options(agfl_variant=variant)).double()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(101)
    projected = AGFL(options(agfl_variant=variant, filter_projection=projection,
                             filter_projection_init='identity')).double()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    baseline_parameters = dict(baseline.named_parameters())
    projected_parameters = dict(projected.named_parameters())
    assert baseline_parameters.keys() <= projected_parameters.keys()
    for name, expected in baseline_parameters.items():
        torch.testing.assert_close(projected_parameters[name], expected, rtol=0, atol=0)
    for graph_filter in projected.filters:
        transforms = list(graph_filter.W) if projection == 'separate' else [graph_filter.W]
        for transform in transforms:
            assert isinstance(transform, torch.nn.Linear)
            assert transform.bias is None
            torch.testing.assert_close(transform.weight, torch.eye(4, dtype=torch.float64), rtol=0, atol=0)

    generator = torch.Generator().manual_seed(103)
    inputs = torch.randn(2, 7, 8, dtype=torch.float64, generator=generator, requires_grad=True)
    projected_inputs = inputs.detach().clone().requires_grad_()
    expected, actual = baseline(inputs), projected(projected_inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(projected_inputs.grad, inputs.grad, rtol=1e-10, atol=1e-12)
    for name, parameter in baseline_parameters.items():
        actual_gradient = projected_parameters[name].grad
        if parameter.grad is None:
            assert actual_gradient is None
        else:
            assert actual_gradient is not None and torch.isfinite(actual_gradient).all()
            torch.testing.assert_close(actual_gradient, parameter.grad, rtol=1e-10, atol=1e-12)
    for graph_filter in projected.filters:
        active_transform = graph_filter.W[1] if projection == 'separate' else graph_filter.W
        assert active_transform.weight.grad is not None
        assert torch.isfinite(active_transform.weight.grad).all()
        assert active_transform.weight.grad.abs().sum() > 0


def test_separate_hop_transforms_learn_after_their_zero_coefficients_become_nonzero():
    graph_filter = GraphFilter(2, options(filter_projection='separate',
                                          filter_projection_init='identity')).double()
    graph = torch.tensor([[[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]]],
                         dtype=torch.float64)
    values = torch.arange(1, 7, dtype=torch.float64).reshape(1, 3, 2)
    optimizer = torch.optim.SGD(graph_filter.parameters(), lr=0.01)
    graph_filter(graph, values).sum().backward()
    assert torch.all(graph_filter.alpha_logits.grad != 0)
    assert torch.all(graph_filter.W[1].weight.grad != 0)
    for hop in (0, 2):
        # alpha=0 intentionally blocks W's initial gradient, but not alpha's.
        torch.testing.assert_close(graph_filter.W[hop].weight.grad,
                                   torch.zeros_like(graph_filter.W[hop].weight), rtol=0, atol=0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    graph_filter(graph, values).sum().backward()
    for transform in graph_filter.W:
        assert transform.weight.grad is not None and torch.isfinite(transform.weight.grad).all()
        assert torch.all(transform.weight.grad != 0)


@pytest.mark.parametrize('projection', ['shared', 'separate'])
@pytest.mark.parametrize('explicit_default', [False, True])
def test_pytorch_projection_initialization_retains_historical_state_and_rng(projection, explicit_default):
    settings = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2, 'filter_projection': projection}
    settings.pop('filter_projection_init')
    settings.pop('temperature_init')
    if explicit_default:
        settings['filter_projection_init'] = 'pytorch'
    torch.manual_seed(107)
    historical = HistoricalAGFL(dim=8, heads=2, K=2, separate_W=projection == 'separate').double()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(107)
    restored = AGFL(settings).double()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert restored.state_dict().keys() == historical.state_dict().keys()
    for name, expected in historical.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected, rtol=0, atol=0)
    with torch.no_grad():
        for graph_filter in historical.filters:
            graph_filter.alpha_logits.copy_(torch.tensor([0.2, 0.6, -0.1], dtype=torch.float64))
    restored.load_state_dict(historical.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 8, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(109))
    torch.testing.assert_close(restored(inputs), historical(inputs, 0, 1), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('projection', ['shared', 'separate'])
def test_saved_projection_weights_override_identity_initialization(projection):
    source = AGFL(options(filter_projection=projection)).double()
    destination = AGFL(options(filter_projection=projection, filter_projection_init='identity')).double()
    assert source.state_dict().keys() == destination.state_dict().keys()
    destination.load_state_dict(source.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 8, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(113))
    torch.testing.assert_close(destination(inputs), source(inputs), rtol=0, atol=0)


@pytest.mark.parametrize('value', [None, True, 0, 'orthogonal', [], {}])
def test_invalid_projection_initialization_is_rejected(value):
    with pytest.raises(ValueError, match='filter_projection_init must be pytorch or identity'):
        AGFL(options(filter_projection='separate', filter_projection_init=value))


def test_identity_projection_initialization_requires_a_projection():
    with pytest.raises(ValueError, match='Identity filter_projection_init requires'):
        AGFL(options(filter_projection='none', filter_projection_init='identity'))


def test_identity_hop_transforms_preserve_initial_eegnet_computation_at_fifteen_tokens():
    spec = get_model_spec('eegnet')
    model_options = {'attention_axis': 'time', 'f1': 4, 'd': 2, 'f2': 8,
                     'temp_kernel': 7, 'pk1': 2, 'pk2': 4}
    metadata = {'modality': 'eeg', 'channels': 6, 'samples': 120, 'num_classes': 4}
    baseline_options = options(heads=4)
    baseline_options.pop('dim')
    projected_options = {**baseline_options, 'filter_projection': 'separate',
                         'filter_projection_init': 'identity'}
    torch.manual_seed(127)
    baseline = spec.build(model_options, metadata, 'agfl', baseline_options).double().eval()
    expected_rng = torch.get_rng_state().clone()
    torch.manual_seed(127)
    projected = spec.build(model_options, metadata, 'agfl', projected_options).double().eval()
    assert torch.equal(torch.get_rng_state(), expected_rng)
    assert baseline.num_tokens == projected.num_tokens == 15
    baseline_state, projected_state = baseline.state_dict(), projected.state_dict()
    assert baseline_state.keys() <= projected_state.keys()
    for name, expected in baseline_state.items():
        torch.testing.assert_close(projected_state[name], expected, rtol=0, atol=0)
    inputs = torch.randn(2, 6, 120, dtype=torch.float64,
                         generator=torch.Generator().manual_seed(131))
    torch.testing.assert_close(projected(inputs), baseline(inputs), rtol=1e-10, atol=1e-12)
