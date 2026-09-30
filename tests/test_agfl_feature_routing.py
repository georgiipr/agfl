"""Target-machine invariants for feature-wise routing; no EEG training needed."""
import pytest
import torch

from agfl.attention.agfl.layer import AGFL, GraphFilter
from tests.test_agfl_token_routing import nonzero_gate, settings


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
@pytest.mark.parametrize('projection', ['none', 'shared', 'separate'])
def test_feature_route_starts_at_static_function_rng_and_shared_gradients(variant, projection):
    options = settings(agfl_variant=variant, filter_projection=projection)
    torch.manual_seed(321)
    static = AGFL(options).double()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(321)
    feature = AGFL({**options, 'coefficient_conditioning': 'feature_contrast'}).double()
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(parameter.numel() for name, parameter in feature.named_parameters()
               if '.coefficient_gate.' in name) == 1536
    for name, value in static.state_dict().items():
        torch.testing.assert_close(feature.state_dict()[name], value, rtol=0, atol=0)
    values = torch.randn(3, 7, 32, dtype=torch.float64)
    first, second = values.clone().requires_grad_(), values.clone().requires_grad_()
    expected, actual = static(first), feature(second)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    torch.testing.assert_close(second.grad, first.grad, rtol=1e-10, atol=1e-11)
    for name, parameter in static.named_parameters():
        gradient = dict(feature.named_parameters())[name].grad
        if parameter.grad is None:
            assert gradient is None
        else:
            torch.testing.assert_close(gradient, parameter.grad, rtol=1e-10, atol=1e-11)
    for item in feature.filters:
        assert torch.isfinite(item.coefficient_gate.weight.grad).all()
        assert torch.count_nonzero(item.coefficient_gate.weight.grad) > 0


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
def test_feature_router_contains_token_router_and_preserves_old_checkpoint_shapes(variant):
    options = settings(agfl_variant=variant, coefficient_conditioning='token_contrast')
    old = AGFL(options).double()
    for item in old.filters:
        nonzero_gate(item)
    feature = AGFL({**options, 'coefficient_conditioning': 'feature_contrast'}).double()
    expanded = {name: value.repeat(8, 1) if '.coefficient_gate.' in name else value
                for name, value in old.state_dict().items()}
    feature.load_state_dict(expanded, strict=True)
    values = torch.randn(2, 7, 32, dtype=torch.float64)
    torch.testing.assert_close(feature(values), old(values), rtol=1e-12, atol=1e-12)
    restored = AGFL(options).double()
    restored.load_state_dict(old.state_dict(), strict=True)
    torch.testing.assert_close(restored(values), old(values), rtol=0, atol=0)
    # Saved modes are never silently reinterpreted as the wider router.
    with pytest.raises(RuntimeError):
        feature.load_state_dict(old.state_dict(), strict=True)


@pytest.mark.parametrize('degree', [1, 2, 4])
def test_feature_formula_zero_sum_bounds_independence_and_token_permutation(degree):
    torch.manual_seed(323)
    module = GraphFilter(4, settings(K=degree, coefficient_conditioning='feature_contrast')).double()
    nonzero_gate(module)
    values = torch.randn(3, 5, 4, dtype=torch.float64)
    graph = torch.randn(3, 5, 5, dtype=torch.float64).softmax(-1)
    hops = [values]
    for _ in range(degree):
        hops.append(graph @ hops[-1])

    def normalize(array):
        centered = array - array.mean(-1, keepdim=True)
        return centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()

    descriptor = torch.cat((normalize(values), normalize(hops[1] - values)), dim=-1)
    raw = (descriptor @ module.coefficient_gate.weight.T).tanh().reshape(3, 5, 4, degree + 1)
    delta = .25 * (raw - raw.mean(-1, keepdim=True))
    expected_coefficients = module.alpha_logits + delta
    coefficients = module.coefficient_values(values, graph)
    assert coefficients.shape == (3, 5, 4, degree + 1)
    torch.testing.assert_close(coefficients, expected_coefficients, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(coefficients.sum(-1), module.alpha_logits.sum().expand(3, 5, 4),
                               rtol=0, atol=1e-15)
    assert delta.abs().max() <= .5 * degree / (degree + 1)
    assert not torch.allclose(coefficients[:, :, 0], coefficients[:, :, 1])
    expected = sum(expected_coefficients[..., hop] * value for hop, value in enumerate(hops))
    torch.testing.assert_close(module(graph, values), expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(module(graph[:1], values[:1]), expected[:1], rtol=1e-12, atol=1e-12)
    order = torch.tensor([3, 0, 4, 1, 2])
    torch.testing.assert_close(module(graph[:, order][:, :, order], values[:, order]), expected[:, order],
                               rtol=1e-12, atol=1e-12)
    identity = torch.eye(5, dtype=torch.float64).expand(3, -1, -1)
    torch.testing.assert_close(module(identity, values), module.alpha_logits.sum() * values,
                               rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_feature_descriptor_promotes_large_contrasts_before_subtraction(dtype):
    module = GraphFilter(4, settings(coefficient_conditioning='feature_contrast')).to(dtype=dtype)
    nonzero_gate(module)
    values = torch.tensor([[[60000., -60000., 20000., -20000.]] * 3], dtype=dtype, requires_grad=True)
    first_hop = (-values.detach()).requires_grad_()
    coefficients = module.coefficient_values(values, first_hop=first_hop)
    assert coefficients.shape == (1, 3, 4, 3) and coefficients.dtype == dtype
    assert torch.isfinite(coefficients).all()
    coefficients.float().square().sum().backward()
    for gradient in (values.grad, first_hop.grad, module.coefficient_gate.weight.grad):
        assert gradient is not None and torch.isfinite(gradient).all()


@pytest.mark.parametrize('updates', [
    {'K': 0, 'coefficient_init': 'uniform'}, {'dim': 4, 'heads': 4},
    {'coefficient_activation': 'softmax', 'coefficient_init': 'uniform'},
    {'learnable_coefficients': False}, {'coefficient_conditioning_scale': 0},
])
def test_invalid_feature_routing_configuration_is_rejected(updates):
    with pytest.raises(ValueError):
        AGFL(settings(coefficient_conditioning='feature_contrast', **updates))


def test_feature_checkpoint_replay_and_cuda_autocast(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    options = settings(coefficient_conditioning='feature_contrast')
    original = AGFL(options).cuda()
    for item in original.filters:
        nonzero_gate(item)
    values = torch.randn(3, 7, 32, device='cuda', requires_grad=True)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        output = original(values)
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert torch.isfinite(values.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in original.parameters())
    saved_path = tmp_path / 'feature.pt'
    torch.save({'options': options, 'model': original.state_dict()}, saved_path)
    saved = torch.load(saved_path, map_location='cuda', weights_only=True)
    restored = AGFL(saved['options']).cuda()
    restored.load_state_dict(saved['model'], strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(values), original(values), rtol=0, atol=0)
