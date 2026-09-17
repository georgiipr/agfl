"""Target-machine checks: learned graph filters can start from the MHA formula."""
from copy import deepcopy

import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL
from agfl.attention.mha.layer import MultiHeadAttention


def qkv_options(**updates):
    options = deepcopy(DEFAULTS)
    options.update(dim=32, heads=4, K=2, projection='qkv', qkv_bias=True,
                   filter_projection='none', score_scaling='sqrt_dim',
                   coefficient_init='one_hop', coefficient_activation='identity',
                   top_k=None, agfl_variant='polynomial')
    options.update(updates)
    return options


def test_dense_one_hop_matches_mha_outputs_and_shared_gradients():
    # Initialization must consume the same RNG draws for the shared projections,
    # so EEGNet's subsequent classifier initialization is also unchanged.
    torch.manual_seed(23)
    baseline = MultiHeadAttention(32, 4).double()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(23)
    graph_filter = AGFL(qkv_options()).double()
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(dict(graph_filter.named_parameters())[name], parameter,
                                   rtol=0, atol=0)

    torch.manual_seed(29)
    baseline_input = torch.randn(3, 7, 32, dtype=torch.float64, requires_grad=True)
    graph_input = baseline_input.detach().clone().requires_grad_(True)
    baseline_output = baseline(baseline_input)
    graph_output = graph_filter(graph_input)
    torch.testing.assert_close(graph_output, baseline_output, rtol=1e-10, atol=1e-12)

    upstream = torch.randn_like(baseline_output)
    (baseline_output * upstream).sum().backward()
    (graph_output * upstream).sum().backward()
    torch.testing.assert_close(graph_input.grad, baseline_input.grad, rtol=1e-9, atol=1e-11)
    for name, parameter in baseline.named_parameters():
        actual = dict(graph_filter.named_parameters())[name]
        torch.testing.assert_close(actual.grad, parameter.grad, rtol=1e-9, atol=1e-11)
    for tap in graph_filter.filters:
        torch.testing.assert_close(tap.alpha_logits.detach(),
                                   torch.tensor([0., 1., 0.], dtype=torch.float64))
        assert torch.isfinite(tap.alpha_logits.grad).all()
        # Higher-order taps start at zero but can learn immediately.
        assert torch.count_nonzero(tap.alpha_logits.grad) == 3
    assert torch.count_nonzero(graph_filter.qkv.weight.grad) > 0


@pytest.mark.parametrize('updates', [
    {'K': 0}, {'coefficient_activation': 'relu'},
    {'coefficient_activation': 'sigmoid'}, {'coefficient_activation': 'softmax'},
])
def test_one_hop_rejects_missing_hop_or_incompatible_activation(updates):
    with pytest.raises(ValueError, match='one_hop initialization requires'):
        AGFL(qkv_options(**updates))


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
def test_sparse_one_hop_has_finite_forward_and_backward(variant):
    torch.manual_seed(31)
    model = AGFL(qkv_options(top_k=5, agfl_variant=variant))
    inputs = torch.randn(3, 7, 32, requires_grad=True)
    output = model(inputs)
    assert output.shape == inputs.shape
    assert torch.isfinite(output).all()
    assert torch.all(torch.count_nonzero(model.last_adj, dim=-1) == 5)
    output.square().mean().backward()
    for parameter in (inputs, model.qkv.weight, model.proj.weight,
                      *(tap.alpha_logits for tap in model.filters)):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
