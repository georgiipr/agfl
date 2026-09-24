"""Target-machine invariants for opt-in token-local AGFL hop routing."""
from copy import deepcopy

import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL, GraphFilter


def settings(**updates):
    options = deepcopy(DEFAULTS)
    options.update(dim=32, heads=4, K=2, projection='qkv', qkv_bias=True,
                   filter_projection='none', score_scaling='sqrt_dim',
                   coefficient_init='one_hop', top_k=5)
    options.update(updates)
    return options


def routed_filter(dim=4, **updates):
    return GraphFilter(dim, settings(coefficient_conditioning='token_contrast', **updates))


def nonzero_gate(module):
    with torch.no_grad():
        # Distinct, modest hop/feature weights avoid saturation and a gate whose
        # rows differ only by a common offset removed by zero-sum redistribution.
        weight = torch.arange(module.coefficient_gate.weight.numel(),
                              device=module.coefficient_gate.weight.device,
                              dtype=module.coefficient_gate.weight.dtype)
        weight = torch.sin(weight * .73).reshape_as(module.coefficient_gate.weight) * .3
        module.coefficient_gate.weight.copy_(weight)


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
@pytest.mark.parametrize('projection', ['none', 'shared', 'separate'])
def test_zero_route_preserves_rng_initial_function_and_shared_gradients(variant, projection):
    options = settings(agfl_variant=variant, filter_projection=projection)
    torch.manual_seed(211)
    static = AGFL(options).double()
    static_rng = torch.get_rng_state().clone()
    torch.manual_seed(211)
    routed = AGFL({**options, 'coefficient_conditioning': 'token_contrast'}).double()
    assert torch.equal(torch.get_rng_state(), static_rng)
    assert set(routed.state_dict()) - set(static.state_dict()) == {
        f'filters.{head}.coefficient_gate.weight' for head in range(4)}
    assert sum(parameter.numel() for name, parameter in routed.named_parameters()
               if '.coefficient_gate.' in name) == 192
    for name, parameter in static.state_dict().items():
        torch.testing.assert_close(routed.state_dict()[name], parameter, rtol=0, atol=0)

    inputs = torch.randn(3, 7, 32, dtype=torch.float64)
    static_input, routed_input = inputs.clone().requires_grad_(), inputs.clone().requires_grad_()
    expected, actual = static(static_input), routed(routed_input)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(routed.last_adj, static.last_adj, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(routed_input.grad, static_input.grad, rtol=1e-10, atol=1e-11)
    for name, parameter in static.named_parameters():
        actual_gradient = dict(routed.named_parameters())[name].grad
        if parameter.grad is None:
            assert actual_gradient is None
        else:
            torch.testing.assert_close(actual_gradient, parameter.grad, rtol=1e-10, atol=1e-11)
    for graph_filter in routed.filters:
        gradient = graph_filter.coefficient_gate.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_routing_matches_independent_formula_and_does_not_add_uniform_gain():
    module = routed_filter(dim=3).double()
    nonzero_gate(module)
    values = torch.tensor([[[1., -2., .5], [4., .2, -1.], [-.1, 3., 2.]]], dtype=torch.float64)
    graph = torch.tensor([[[.6, .3, .1], [.2, .7, .1], [.1, .2, .7]]], dtype=torch.float64)
    first, second = graph @ values, graph @ (graph @ values)

    def normalize(features):
        centered = features - features.mean(dim=-1, keepdim=True)
        return centered / (centered.square().mean(dim=-1, keepdim=True) + 1e-5).sqrt()

    descriptor = torch.cat((normalize(values), normalize(first - values)), dim=-1)
    raw = (descriptor @ module.coefficient_gate.weight.T).tanh()
    delta = module.conditioning_scale * .5 * (raw - raw.mean(dim=-1, keepdim=True))
    coefficients = module.alpha_logits + delta
    expected = sum(coefficients[..., hop, None] * propagated
                   for hop, propagated in enumerate((values, first, second)))
    actual_coefficients = module.coefficient_values(values, graph)
    torch.testing.assert_close(actual_coefficients, coefficients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module.coefficient_values(values, first_hop=first), coefficients,
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module(graph, values), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(delta.sum(-1), torch.zeros_like(delta[..., 0]), rtol=0, atol=1e-15)
    assert torch.all(delta.abs() <= module.conditioning_scale)
    assert delta.min() < 0 < delta.max()
    assert not torch.allclose(actual_coefficients[:, 0], actual_coefficients[:, 1])


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
def test_routing_is_sample_independent_and_token_permutation_equivariant(variant):
    torch.manual_seed(213)
    module = routed_filter(agfl_variant=variant).double()
    nonzero_gate(module)
    values = torch.randn(3, 5, 4, dtype=torch.float64)
    graphs = torch.randn(3, 5, 5, dtype=torch.float64).softmax(-1)
    output = module(graphs, values)
    coefficients = module.coefficient_values(values, graphs)
    assert coefficients.shape == (3, 5, 3)
    torch.testing.assert_close(module(graphs[:1], values[:1]), output[:1], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module.coefficient_values(values[:1], graphs[:1]), coefficients[:1],
                               rtol=1e-12, atol=1e-12)
    changed = values.clone()
    changed[1:] *= 100
    torch.testing.assert_close(module(graphs, changed)[:1], output[:1], rtol=1e-12, atol=1e-12)
    order = torch.tensor([3, 0, 4, 1, 2])
    reordered_graphs = graphs[:, order][:, :, order]
    torch.testing.assert_close(module(reordered_graphs, values[:, order]), output[:, order],
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module.coefficient_values(values[:, order], reordered_graphs),
                               coefficients[:, order], rtol=1e-12, atol=1e-12)


def test_identity_graph_cannot_turn_redistribution_into_token_amplitude_scaling():
    torch.manual_seed(215)
    module = routed_filter().double()
    nonzero_gate(module)
    values = torch.randn(2, 7, 4, dtype=torch.float64)
    graph = torch.eye(7, dtype=torch.float64).expand(2, -1, -1)
    coefficients = module.coefficient_values(values, graph)
    assert not torch.allclose(coefficients[:, 0], coefficients[:, 1])
    torch.testing.assert_close(coefficients.sum(-1), module.alpha_logits.sum().expand(2, 7),
                               rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module(graph, values), module.alpha_logits.sum() * values,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('degree', [1, 2, 4])
def test_centered_hop_corrections_remain_bounded_at_gate_saturation(degree):
    torch.manual_seed(216)
    module = routed_filter(K=degree).double()
    nonzero_gate(module)
    with torch.no_grad():
        module.coefficient_gate.weight.mul_(1000)
    values = torch.randn(2, 7, 4, dtype=torch.float64)
    graph = torch.randn(2, 7, 7, dtype=torch.float64).softmax(-1)
    coefficients = module.coefficient_values(values, graph)
    assert coefficients.shape == (2, 7, degree + 1)
    deltas = coefficients - module.alpha_logits
    assert torch.isfinite(coefficients).all()
    assert torch.all(deltas.abs() <= module.conditioning_scale + 1e-15)
    torch.testing.assert_close(deltas.sum(-1), torch.zeros(2, 7, dtype=torch.float64),
                               rtol=0, atol=1e-15)
    assert torch.isfinite(module(graph, values)).all()


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float64])
@pytest.mark.parametrize('zero_input', [False, True])
def test_large_half_precision_contrast_is_promoted_before_subtraction(device, dtype, zero_input):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    if device == 'cuda' and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA bfloat16 is unavailable')
    module = routed_filter().to(device=device, dtype=dtype)
    nonzero_gate(module)
    pattern = [0., 0., 0., 0.] if zero_input else [60000., -60000., 20000., -20000.]
    values = torch.tensor([[pattern] * 3], device=device, dtype=dtype, requires_grad=True)
    first_hop = (-values.detach()).requires_grad_()
    coefficients = module.coefficient_values(values, first_hop=first_hop)
    assert coefficients.shape == (1, 3, 3) and coefficients.dtype == dtype
    assert torch.isfinite(coefficients).all()
    coefficients.float().square().sum().backward()
    for gradient in (values.grad, first_hop.grad, module.coefficient_gate.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()
    if zero_input:
        torch.testing.assert_close(coefficients, module.alpha_logits.expand(1, 3, -1), rtol=0, atol=0)
    else:
        assert torch.count_nonzero(module.coefficient_gate.weight.grad) > 0


def test_cuda_autocast_forward_backward_and_gate_gradients_are_finite():
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable')
    torch.manual_seed(217)
    module = AGFL(settings(coefficient_conditioning='token_contrast')).cuda()
    for graph_filter in module.filters:
        nonzero_gate(graph_filter)
    values = torch.randn(3, 7, 32, device='cuda', requires_grad=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = module(values)
        loss = output.float().square().mean()
    assert torch.isfinite(output).all()
    loss.backward()
    assert torch.isfinite(values.grad).all()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in module.parameters())
    assert all(graph_filter.coefficient_gate.weight.grad is not None for graph_filter in module.filters)


def test_old_static_and_trial_state_dicts_keep_their_shapes_and_load_strictly():
    for mode in ('static', 'trial_power'):
        previous = AGFL(settings(coefficient_conditioning=mode)).double()
        saved_options = settings(coefficient_conditioning=mode)
        if mode == 'static':
            saved_options.pop('coefficient_conditioning')
            saved_options.pop('coefficient_conditioning_scale')
        loaded = AGFL(saved_options).double()
        loaded.load_state_dict(previous.state_dict(), strict=True)
        assert loaded.state_dict().keys() == previous.state_dict().keys()
        values = torch.randn(2, 7, 32, dtype=torch.float64)
        torch.testing.assert_close(loaded(values), previous(values), rtol=0, atol=0)
        coefficients = loaded.filters[0].coefficient_values(values[..., :8])
        assert coefficients.shape == (2, 3)
    routed = AGFL(settings(coefficient_conditioning='token_contrast'))
    with pytest.raises(RuntimeError):
        routed.load_state_dict(AGFL(settings()).state_dict(), strict=True)


def test_saved_routed_checkpoint_replays_exactly(tmp_path):
    torch.manual_seed(219)
    options = settings(coefficient_conditioning='token_contrast')
    original = AGFL(options).double().eval()
    for graph_filter in original.filters:
        nonzero_gate(graph_filter)
    values = torch.randn(2, 7, 32, dtype=torch.float64)
    with torch.no_grad():
        expected = original(values)
    path = tmp_path / 'routed.pt'
    torch.save({'options': options, 'model': original.state_dict()}, path)
    saved = torch.load(path, map_location='cpu', weights_only=True)
    restored = AGFL(saved['options']).double().eval()
    restored.load_state_dict(saved['model'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize('updates', [
    {'K': 0, 'coefficient_init': 'uniform'},
    {'dim': 4, 'heads': 4},
    {'coefficient_activation': 'softmax', 'coefficient_init': 'uniform'},
    {'coefficient_activation': 'sigmoid', 'coefficient_init': 'uniform'},
    {'coefficient_activation': 'relu', 'coefficient_init': 'uniform'},
    {'learnable_coefficients': False},
    {'coefficient_conditioning_scale': 0},
    {'coefficient_conditioning_scale': float('nan')},
    {'coefficient_conditioning_scale': True},
])
def test_incompatible_token_routing_configuration_is_rejected(updates):
    with pytest.raises(ValueError):
        AGFL(settings(coefficient_conditioning='token_contrast', **updates))


def test_token_gate_requires_the_actual_one_hop_features_or_graph():
    module = routed_filter()
    with pytest.raises(ValueError):
        module.coefficient_values(torch.randn(2, 7, 4))


def test_token_preset_restores_the_reference_backbone_training_and_matched_splits():
    from agfl.config import comparison_identity, experiment_identity, merge, resolve_experiments
    from agfl.presets import load_preset

    reference = load_preset('eegnet-a03-conditioned-agfl')
    current = load_preset('eegnet-a03-token-agfl')
    assert [row['attention'] for row in current['experiments']] == ['agfl', 'agfl', 'mha', 'linformer']
    configs = [resolve_experiments(merge(current['base'], row))[0] for row in current['experiments']]
    old = resolve_experiments(merge(reference['base'], reference['experiments'][0]))[0]
    assert len({comparison_identity(config) for config in configs}) == 1
    assert len({experiment_identity(config) for config in configs}) == 4
    assert sum(len(config['seeds']) for config in configs) == 20
    for config in configs:
        assert config['model'] == 'eegnet' and config['subject_id'] == 'A03'
        for key in ('data', 'split', 'model_options', 'training', 'seeds'):
            assert config[key] == old[key], key
        assert config['model_options']['temporal_statistics'] == 'mean'
        assert config['model_options']['temp_kernel'] == 32
        assert config['training']['ema_decay'] == 0
        assert config['training']['checkpoint_tiebreaker'] == 'none'
    assert configs[0]['attention_options'] == old['attention_options']
    assert configs[1]['attention_options'] == {**old['attention_options'],
                                             'coefficient_conditioning': 'token_contrast'}
    assert configs[3]['attention_options']['projection_rank'] == 4
