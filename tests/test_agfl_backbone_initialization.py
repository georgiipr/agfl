"""Target-machine checks of MHA-equivalent AGFL initialization in real backbones.

These small synthetic cases check the initial computation and gradient paths,
not trained accuracy. Run on the experiment machine, not the local workspace.
"""
import pytest
import torch

from agfl_speech.models import get_model_spec
from tests.model_fixtures import small_model_options


MODELS = ('eegnet', 'eegnet_original', 'signal_transformer')

# Leave heads and K unspecified: the model's own attention defaults must survive.
DENSE_ONE_HOP = {
    'projection': 'qkv', 'qkv_bias': True, 'filter_projection': 'none',
    'score_scaling': 'sqrt_dim', 'coefficient_init': 'one_hop',
    'coefficient_activation': 'identity', 'learnable_coefficients': True,
    'top_k': None, 'top_k_stage': 'scores', 'graph_normalization': 'softmax',
    'agfl_variant': 'polynomial',
}


def assert_finite_close(actual, expected, *, rtol=1e-8, atol=1e-10):
    assert actual is not None and expected is not None
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize('key', MODELS)
def test_dense_one_hop_matches_mha_in_each_backbone(key):
    spec = get_model_spec(key)
    options = small_model_options(key)
    # Exercise more than one insertion in the stacked architecture.
    if key == 'signal_transformer':
        options['depth'] = 2
    channels = 6
    metadata = {'modality': 'eeg', 'channels': channels, 'samples': 65,
                'num_classes': 3}

    construct = spec.build
    torch.manual_seed(43)
    baseline = construct(options, metadata, 'mha').double().eval()
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(43)
    graph_model = construct(options, metadata, 'agfl', DENSE_ONE_HOP).double().eval()
    assert torch.equal(torch.get_rng_state(), baseline_rng)

    # This includes the stem, classifier, positional/BatchNorm buffers and all
    # Q/K/V/output projections, not only weights outside attention modules.
    baseline_state, graph_state = baseline.state_dict(), graph_model.state_dict()
    assert baseline_state.keys() <= graph_state.keys()
    for name, expected in baseline_state.items():
        torch.testing.assert_close(graph_state[name], expected, rtol=0, atol=0)

    baseline_attention = {name: layer for name, layer in baseline.named_modules()
                          if getattr(layer, 'is_attention', False)}
    graph_attention = {name: layer for name, layer in graph_model.named_modules()
                       if getattr(layer, 'is_attention', False)}
    assert baseline_attention.keys() == graph_attention.keys()
    assert graph_attention
    depth = options.get('depth', 1)
    assert [layer.layer_idx for layer in graph_attention.values()] == list(range(depth))
    agfl_defaults = spec.attention_defaults_for('agfl')
    for name, layer in graph_attention.items():
        other = baseline_attention[name]
        assert layer.depth == other.depth == depth
        assert layer.token_axis == other.token_axis == graph_model.token_axis == 'electrode'
        assert layer.num_tokens == other.num_tokens == channels
        assert layer.heads == other.heads == agfl_defaults['heads']
        assert layer.options['K'] == agfl_defaults['K']
        for graph_filter in layer.filters:
            expected = torch.zeros(agfl_defaults['K'] + 1, dtype=torch.float64)
            expected[1] = 1.
            torch.testing.assert_close(graph_filter.alpha_logits, expected, rtol=0, atol=0)

    generator = torch.Generator().manual_seed(47)
    baseline_input = torch.randn(3, channels, 65, dtype=torch.float64,
                                 generator=generator, requires_grad=True)
    graph_input = baseline_input.detach().clone().requires_grad_(True)
    expected = baseline(baseline_input)
    actual = graph_model(graph_input)
    assert actual.shape == expected.shape == (3, metadata['num_classes'])
    assert_finite_close(actual, expected)

    # An arbitrary common upstream gradient avoids depending on a particular
    # class loss and exercises all logits through the complete feature path.
    upstream = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()
    assert_finite_close(graph_input.grad, baseline_input.grad)
    assert graph_input.grad.abs().sum() > 0
    graph_parameters = dict(graph_model.named_parameters())
    for name, parameter in baseline.named_parameters():
        actual_grad = graph_parameters[name].grad
        if parameter.grad is None:
            assert actual_grad is None, name
        else:
            assert_finite_close(actual_grad, parameter.grad)

    for layer in graph_attention.values():
        # Each projection can learn immediately; individual entries may
        # legitimately have zero gradients, especially the key bias.
        for gradient in layer.qkv.weight.grad.chunk(3, dim=0):
            assert torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
        coefficient_gradients = torch.stack([head.alpha_logits.grad for head in layer.filters])
        assert torch.isfinite(coefficient_gradients).all()
        assert coefficient_gradients[:, 0].abs().sum() > 0
        if layer.options['K'] > 1:
            # Check higher-order taps jointly, without requiring every head
            # or every tap to have a nonzero gradient for this synthetic batch.
            assert coefficient_gradients[:, 2:].abs().sum() > 0
