"""Target-machine checks for token routing artifacts; no training or datasets."""
from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL
from agfl.visualization.diagnostics import (
    TrialCoefficientCapture, coefficient_conditioning_figures, filter_figures,
)


class TokenWriter:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.plt = Mock()
        self.figures, self.axes = [], []

        def subplots(*args, **kwargs):
            if kwargs.get('squeeze') is False:
                axes = np.empty((args[0], args[1]), dtype=object)
                for index in np.ndindex(axes.shape):
                    axes[index] = Mock()
                self.axes.append(axes)
            else:
                axes = Mock()
            return Mock(), axes

        self.plt.subplots.side_effect = subplots

    def save(self, figure, name, caption):
        self.figures.append({'name': name, 'caption': caption})


def token_model(*, degree=2, variant='polynomial'):
    options = {**deepcopy(DEFAULTS), 'dim': 4, 'heads': 2, 'K': degree,
               'projection': 'qkv', 'qkv_bias': False, 'filter_projection': 'none',
               'coefficient_init': 'one_hop', 'coefficient_activation': 'identity',
               'coefficient_conditioning': 'token_contrast', 'coefficient_conditioning_scale': .5,
               'score_scaling': 'sqrt_dim', 'agfl_variant': variant, 'hop_normalization': 'feature'}
    model = torch.nn.ModuleDict({'attention': AGFL(options)}).double().eval()
    with torch.no_grad():
        model['attention'].qkv.weight.zero_()
        model['attention'].qkv.weight[8:].copy_(torch.eye(4))
        for head, item in enumerate(model['attention'].filters):
            weights = torch.arange((degree + 1) * 4, dtype=torch.float64).reshape(degree + 1, 4)
            # Nonparallel rows are needed after centering across hops.
            weights = (weights.remainder(7) - 3.) / 5.
            item.coefficient_gate.weight.copy_(weights * (head + 1))
    return model


def token_values():
    return torch.tensor([
        [[1., 2., 3., 4.], [2., 1., 4., 3.], [.5, 3., .1, 2.]],
        [[4., .5, 1., .25], [5., .2, .8, .2], [1., 2., 4., .4]],
        [[.1, 3., .5, 4.], [.2, 4., .6, 3.], [2., 1., 2., 1.]],
        [[2., 2., 1., 1.], [3., 3., 2., 2.], [.5, 1., 2., 3.]],
        [[5., .1, .1, 5.], [6., .2, .2, 6.], [.1, 2., 5., 1.]],
    ], dtype=torch.float64)


def capture_tokens(model, values, ids):
    capture = TrialCoefficientCapture(model)
    try:
        with torch.no_grad():
            for start, end in ((0, 2), (2, 3), (3, 5)):
                capture.start_batch(ids[start:end])
                model['attention'](values[start:end])
    finally:
        for handle in capture.hooks:
            handle.remove()
    return capture


@pytest.mark.parametrize('variant', ['polynomial', 'renormalized'])
def test_token_capture_uses_actual_graph_and_values_for_every_trial_and_token(tmp_path, variant):
    model, values = token_model(variant=variant), token_values()
    ids, labels = ['trial-z', 'trial-a', 'trial-q', 'trial-b', 'trial-m'], [3, 0, 1, 2, 0]
    expected, expected_contributions = [], []
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for head, item in enumerate(model['attention'].filters):
            v = values[:, :, head * 2:(head + 1) * 2]
            # Zero Q/K and threshold ties produce a uniform actual graph.
            first_hop = v.mean(dim=1, keepdim=True).expand_as(v)
            if variant == 'renormalized':
                first_hop = first_hop * (torch.linalg.vector_norm(v, dim=-1, keepdim=True) /
                                         (torch.linalg.vector_norm(first_hop, dim=-1, keepdim=True) + 1e-6))
            descriptor = torch.cat([F.layer_norm(v, (2,), eps=1e-5),
                                    F.layer_norm(first_hop - v, (2,), eps=1e-5)], dim=-1)
            gate = F.linear(descriptor, item.coefficient_gate.weight).tanh()
            coefficients = item.alpha_logits + .25 * (gate - gate.mean(dim=-1, keepdim=True))
            expected.append(coefficients.numpy())
            second_hop = first_hop.mean(dim=1, keepdim=True).expand_as(v)
            if variant == 'renormalized':
                second_hop = second_hop * (torch.linalg.vector_norm(v, dim=-1, keepdim=True) /
                                           (torch.linalg.vector_norm(second_hop, dim=-1, keepdim=True) + 1e-6))
            terms = torch.stack([v, first_hop, second_hop], dim=-2) * coefficients.unsqueeze(-1)
            expected_contributions.append(torch.linalg.vector_norm(terms, dim=-1).numpy())
        expected_logits = model['attention'](values)
    expected = np.stack(expected, axis=1)
    rng_before = torch.random.get_rng_state().clone()

    capture = capture_tokens(model, values, ids)
    writer = TokenWriter(tmp_path)
    summary = coefficient_conditioning_figures(writer, capture, ids, labels, 'validation')

    assert summary['schema_version'] == 2
    assert summary == json.loads((tmp_path / 'coefficient_conditioning.json').read_text())
    layer = summary['layers']['attention']
    assert layer['coefficient_array_axes'] == ['trial', 'head', 'token', 'hop']
    assert layer['samples'] == 5 and layer['tokens'] == 3
    assert layer['summary_reduction_axes'] == ['trial', 'token']
    assert layer['attainable_delta_bound'] == pytest.approx(1 / 3)
    with np.load(tmp_path / layer['array_file'], allow_pickle=False) as saved:
        assert saved['effective_coefficients'].shape == (5, 2, 3, 3)
        np.testing.assert_allclose(saved['effective_coefficients'], expected, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(saved['sample_ids'], ids)
        np.testing.assert_array_equal(saved['labels'], labels)
        np.testing.assert_array_equal(saved['token_indices'], [0, 1, 2])
        np.testing.assert_allclose(saved['coefficient_deltas'], expected - saved['base_coefficients'][None, :, None])
        np.testing.assert_allclose(saved['normalized_deltas'], saved['coefficient_deltas'] / .5)
        np.testing.assert_allclose(saved['bound_normalized_deltas'], saved['coefficient_deltas'] * 3)
        np.testing.assert_allclose(saved['within_trial_token_std'], expected.std(axis=2))
        np.testing.assert_allclose(saved['delta_sum_over_hops'], 0., atol=1e-12)
        assert saved['propagated_hop_cosines'].shape == (5, 2, 3, 3, 3)
        assert saved['propagated_hop_cosine_valid'].all()
        np.testing.assert_allclose(np.diagonal(saved['propagated_hop_cosines'], axis1=-2, axis2=-1), 1., atol=1e-12)
        np.testing.assert_allclose(saved['hop_contribution_norms'], np.stack(expected_contributions, axis=1),
                                   rtol=1e-10, atol=1e-12)
    assert np.max(expected.std(axis=2)) > .01
    for head in range(2):
        statistics = layer['heads'][head]
        np.testing.assert_allclose(statistics['effective_mean'], expected[:, head].mean(axis=(0, 1)))
        np.testing.assert_allclose(statistics['effective_mean_per_token'], expected[:, head].mean(axis=0))
        np.testing.assert_allclose(statistics['within_trial_token_std_mean'], expected[:, head].std(axis=1).mean(axis=0))
        assert statistics['delta_sum_over_hops_max_absolute'] < 1e-12
        # Each hop panel has a separate trial-by-token matrix, never a pooled
        # token mean or an accidental plot of only the last batch.
        for hop in range(3):
            displayed = writer.axes[head][0, hop].imshow.call_args.args[0]
            np.testing.assert_allclose(displayed, expected[:, head, :, hop], rtol=1e-10, atol=1e-12)
    assert len(writer.figures) == 2
    assert all('Columns retain every token' in item['caption'] for item in writer.figures)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        torch.testing.assert_close(model['attention'](values), expected_logits, rtol=0, atol=0)


@pytest.mark.parametrize('degree', [1, 2, 3])
def test_token_saturation_uses_centered_tanh_attainable_bound(tmp_path, degree):
    model = token_model(degree=degree)
    capture = TrialCoefficientCapture(model)
    for handle in capture.hooks:
        handle.remove()
    taps = degree + 1
    bound = .5 * (taps - 1) / taps
    relative = np.array([1.] + [-1 / (taps - 1)] * (taps - 1))
    base = np.zeros(taps)
    base[1] = 1.
    coefficients = np.broadcast_to(base + bound * relative, (2, 3, taps)).copy()
    for head in range(2):
        capture.layers['attention']['coefficients'][head] = [coefficients]
        capture.layers['attention']['sample_ids'][head] = [np.array(['first', 'second'])]

    summary = coefficient_conditioning_figures(
        TokenWriter(tmp_path), capture, ['first', 'second'], [0, 1], 'validation')

    layer = summary['layers']['attention']
    assert layer['attainable_delta_bound'] == pytest.approx(bound)
    assert layer['heads'][0]['saturation_fraction'] == ([1., 1.] if degree == 1 else [1.] + [0.] * degree)
    with np.load(tmp_path / layer['array_file'], allow_pickle=False) as saved:
        assert np.abs(saved['normalized_deltas']).max() < .95
        assert np.abs(saved['bound_normalized_deltas']).max() == pytest.approx(1.)


def test_rejects_token_adjustments_that_violate_zero_sum_even_within_individual_bounds(tmp_path):
    ids = ['a', 'b', 'c', 'd', 'e']
    model = token_model()
    with torch.no_grad():
        for item in model['attention'].filters:
            item.coefficient_gate.weight.zero_()
    capture = capture_tokens(model, token_values(), ids)
    capture.layers['attention']['coefficients'][0][0][0, 0, 0] += .01

    with pytest.raises(ValueError, match='sum to zero'):
        coefficient_conditioning_figures(TokenWriter(tmp_path), capture, ids, np.zeros(5), 'validation')

    assert not (tmp_path / 'coefficient_conditioning.json').exists()


def test_unmoved_token_gate_has_no_false_variation_and_base_plot_is_labeled(tmp_path):
    model = token_model()
    with torch.no_grad():
        for item in model['attention'].filters:
            item.coefficient_gate.weight.zero_()
    ids = ['a', 'b', 'c', 'd', 'e']
    writer = TokenWriter(tmp_path)
    capture = capture_tokens(model, token_values(), ids)

    summary = coefficient_conditioning_figures(writer, capture, ids, [0, 1, 2, 3, 0], 'validation')
    parameters = filter_figures(writer, model)['attention']

    assert parameters['coefficient_scope'] == 'base_before_token_adjustment'
    assert all(head['coefficients_effective'] is None for head in parameters['heads'])
    assert 'before destination-token adjustments' in writer.figures[-1]['caption']
    for head in summary['layers']['attention']['heads']:
        assert head['effective_mean'] == [0., 1., 0.]
        assert head['within_trial_token_std_mean'] == [0., 0., 0.]
        assert head['within_trial_token_std_max'] == [0., 0., 0.]
        assert head['saturation_fraction'] == [0., 0., 0.]
        assert head['delta_sum_over_hops_max_absolute'] == 0.


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_token_summary_serializes_the_training_dtype_tolerance(tmp_path, dtype):
    # The other formula checks use float64, which JSON happens to accept as a
    # scalar. Real checkpoints use float32; its NumPy scalar must not leak into
    # the saved summary even when it wins the tolerance's max comparison.
    model = token_model().to(dtype=dtype)
    ids = ['a', 'b', 'c', 'd', 'e']
    capture = capture_tokens(model, token_values().to(dtype=dtype), ids)
    summary = coefficient_conditioning_figures(
        TokenWriter(tmp_path), capture, ids, [0, 1, 2, 3, 0], 'validation')
    tolerance = summary['layers']['attention']['zero_sum_check_tolerance']
    assert type(tolerance) is float
    assert tolerance == pytest.approx(max(1e-7, float(torch.finfo(dtype).eps) * 3 * 8))
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert json.loads((tmp_path / 'coefficient_conditioning.json').read_text()) == summary


def test_token_capture_rejects_reordered_head_ids(tmp_path):
    ids = ['a', 'b', 'c', 'd', 'e']
    capture = capture_tokens(token_model(), token_values(), ids)
    capture.layers['attention']['sample_ids'][1][0] = np.array(['b', 'a'])

    with pytest.raises(ValueError, match='order/count'):
        coefficient_conditioning_figures(TokenWriter(tmp_path), capture, ids, np.zeros(5), 'validation')
