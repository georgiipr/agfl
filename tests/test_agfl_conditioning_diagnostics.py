"""Target-machine checks for trial-conditioned diagnostics; never train models."""
from copy import deepcopy
import json
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL
from agfl.visualization.diagnostics import (
    TrialCoefficientCapture, coefficient_conditioning_figures, filter_figures,
)


class DiagnosticWriter:
    """Exercise actual arrays/JSON without requiring a plotting backend."""

    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.plt = Mock()
        self.figures = []

        def subplots(*args, **kwargs):
            if kwargs.get('squeeze') is False:
                axes = np.empty((args[0], args[1]), dtype=object)
                for index in np.ndindex(axes.shape):
                    axes[index] = Mock()
            else:
                axes = Mock()
            return Mock(), axes

        self.plt.subplots.side_effect = subplots

    def save(self, figure, name, caption):
        self.figures.append({'name': name, 'caption': caption})


def model_with_conditioning(conditioning='trial_power', scale=.5):
    options = {**deepcopy(DEFAULTS), 'dim': 4, 'heads': 2, 'K': 2,
               'projection': 'qkv', 'qkv_bias': False, 'filter_projection': 'none',
               'coefficient_activation': 'identity', 'coefficient_init': 'one_hop',
               'coefficient_conditioning': conditioning, 'coefficient_conditioning_scale': scale,
               'score_scaling': 'sqrt_dim'}
    model = torch.nn.ModuleDict({'attention': AGFL(options)}).double().eval()
    with torch.no_grad():
        # Deliberately make Q/K graphs uninformative about V: coefficients must
        # come from the actual value features, not adjacency snapshots.
        model['attention'].qkv.weight.zero_()
        model['attention'].qkv.weight[8:].copy_(torch.eye(4))
        if conditioning != 'static':
            for head, item in enumerate(model['attention'].filters):
                item.coefficient_gate.weight.copy_(
                    torch.tensor([[.1, -.2], [.3, .15], [-.25, .1]]) * (head + 1))
    return model


def value_inputs():
    return torch.tensor([
        [[1., 2., 3., 4.], [2., 1., 4., 3.]],
        [[4., .5, 1., .25], [5., .2, .8, .2]],
        [[.1, 3., .5, 4.], [.2, 4., .6, 3.]],
        [[2., 2., 1., 1.], [3., 3., 2., 2.]],
        [[5., .1, .1, 5.], [6., .2, .2, 6.]],
    ], dtype=torch.float64)


def captured_batches(model, inputs, sample_ids):
    capture = TrialCoefficientCapture(model)
    try:
        with torch.no_grad():
            for start, end in ((0, 2), (2, 3), (3, 5)):
                capture.start_batch(sample_ids[start:end])
                model['attention'](inputs[start:end])
    finally:
        for handle in capture.hooks:
            handle.remove()
    return capture


def test_all_batches_save_actual_value_coefficients_with_trial_ids_and_unchanged_state(tmp_path):
    model, inputs = model_with_conditioning(), value_inputs()
    sample_ids = ['trial-z', 'trial-b', 'trial-x', 'trial-a', 'trial-q']
    labels = np.array([3, 0, 2, 1, 3])
    before = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        expected = torch.stack([
            item.coefficient_values(inputs[:, :, head * 2:(head + 1) * 2])
            for head, item in enumerate(model['attention'].filters)
        ], dim=1).numpy()
        expected_logits = model['attention'](inputs)
    rng_before = torch.random.get_rng_state().clone()

    capture = captured_batches(model, inputs, sample_ids)
    writer = DiagnosticWriter(tmp_path)
    summary = coefficient_conditioning_figures(writer, capture, sample_ids, labels, 'validation')

    assert summary == json.loads((tmp_path / 'coefficient_conditioning.json').read_text())
    assert summary['sample_ids'] == sample_ids
    assert summary['labels'] == labels.tolist()
    assert summary['partition'] == 'validation'
    layer = summary['layers']['attention']
    assert layer['samples'] == 5
    assert layer['coefficient_array_axes'] == ['trial', 'head', 'hop']
    with np.load(tmp_path / layer['array_file'], allow_pickle=False) as saved:
        assert saved['effective_coefficients'].shape == (5, 2, 3)
        np.testing.assert_allclose(saved['effective_coefficients'], expected, atol=1e-12)
        np.testing.assert_array_equal(saved['sample_ids'], sample_ids)
        np.testing.assert_array_equal(saved['labels'], labels)
        np.testing.assert_allclose(saved['coefficient_deltas'], expected - saved['base_coefficients'][None])
        np.testing.assert_allclose(saved['normalized_deltas'], saved['coefficient_deltas'] / .5)
        np.testing.assert_array_equal(saved['gate_weights'], np.stack([
            item.coefficient_gate.weight.detach().numpy() for item in model['attention'].filters]))
    assert np.max(expected.std(axis=0)) > .01
    assert layer['heads'][0]['effective_mean'] == pytest.approx(expected[:, 0].mean(axis=0))
    assert layer['heads'][0]['effective_std'] == pytest.approx(expected[:, 0].std(axis=0))
    assert len(writer.figures) == 1
    assert 'selected validation trials' in writer.figures[0]['caption']
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.parameters())
    with torch.no_grad():
        torch.testing.assert_close(model['attention'](inputs), expected_logits, rtol=0, atol=0)


@pytest.mark.parametrize('problem', ['reordered_ids', 'missing_batch', 'duplicate_batch'])
def test_rejects_incomplete_or_misaligned_trial_capture(tmp_path, problem):
    ids = ['a', 'b', 'c', 'd', 'e']
    capture = captured_batches(model_with_conditioning(), value_inputs(), ids)
    state = capture.layers['attention']
    if problem == 'reordered_ids':
        state['sample_ids'][1][0] = np.array(['b', 'a'])
    elif problem == 'missing_batch':
        state['coefficients'][0].pop()
        state['sample_ids'][0].pop()
    else:
        state['coefficients'][1].append(state['coefficients'][1][-1])
        state['sample_ids'][1].append(state['sample_ids'][1][-1])

    with pytest.raises(ValueError, match='order/count'):
        coefficient_conditioning_figures(DiagnosticWriter(tmp_path), capture, ids, np.zeros(5), 'validation')

    assert not (tmp_path / 'coefficient_conditioning.json').exists()


@pytest.mark.parametrize('ids', [None, ['one-only']])
def test_capture_requires_current_batch_ids(ids):
    model = model_with_conditioning()
    capture = TrialCoefficientCapture(model)
    try:
        if ids is not None:
            capture.start_batch(ids)
        with torch.no_grad(), pytest.raises(ValueError, match='sample IDs'):
            model['attention'](value_inputs()[:2])
    finally:
        for handle in capture.hooks:
            handle.remove()


def test_summary_uses_configured_delta_bound_for_saturation(tmp_path):
    model = model_with_conditioning(scale=.25)
    capture = TrialCoefficientCapture(model)
    for handle in capture.hooks:
        handle.remove()
    normalized = np.array([[.96, .94, 0.], [-.98, -1., .2]])
    base = np.array([0., 1., 0.])
    for head in range(2):
        capture.layers['attention']['coefficients'][head] = [base + .25 * normalized]
        capture.layers['attention']['sample_ids'][head] = [np.array(['first', 'second'])]

    summary = coefficient_conditioning_figures(
        DiagnosticWriter(tmp_path), capture, ['first', 'second'], [0, 1], 'validation')

    layer = summary['layers']['attention']
    assert layer['coefficient_conditioning_scale'] == .25
    assert layer['saturation_normalized_threshold'] == .95
    for head in layer['heads']:
        assert head['saturation_fraction'] == [1., .5, 0.]
        assert head['delta_mean_absolute'] == pytest.approx(np.abs(.25 * normalized).mean(axis=0))
        assert head['delta_std'] == pytest.approx((.25 * normalized).std(axis=0))


def test_older_static_options_have_no_dynamic_capture_and_keep_effective_coefficients(tmp_path):
    model = model_with_conditioning('static')
    model['attention'].options.pop('coefficient_conditioning')
    model['attention'].options.pop('coefficient_conditioning_scale')
    capture = TrialCoefficientCapture(model)
    writer = DiagnosticWriter(tmp_path)

    assert capture.layers == {} and capture.hooks == []
    assert coefficient_conditioning_figures(writer, capture, ['a'], [0], 'validation') is None
    layer = filter_figures(writer, model)['attention']

    assert layer['coefficient_conditioning'] == 'static'
    assert layer['coefficient_scope'] == 'sample_independent'
    assert all(head['coefficients_effective'] == [0., 1., 0.] for head in layer['heads'])
    assert not (tmp_path / 'coefficient_conditioning.json').exists()
    assert writer.figures[0]['caption'] == 'Learned AGFL coefficients after the configured coefficient activation.'


def test_conditioned_parameter_plot_labels_base_instead_of_actual_trial_coefficients(tmp_path):
    model = model_with_conditioning()
    writer = DiagnosticWriter(tmp_path)

    layer = filter_figures(writer, model)['attention']

    assert layer['coefficient_scope'] == 'base_before_trial_adjustment'
    assert layer['coefficient_conditioning'] == 'trial_power'
    assert all(head['coefficients_effective'] is None for head in layer['heads'])
    assert all(head['coefficients_base'] == [0., 1., 0.] for head in layer['heads'])
    assert 'base coefficients before trial-dependent adjustments' in writer.figures[0]['caption']


def test_unmoved_zero_gate_reports_no_trial_variation_or_saturation(tmp_path):
    model = model_with_conditioning()
    with torch.no_grad():
        for item in model['attention'].filters:
            item.coefficient_gate.weight.zero_()
    ids = ['a', 'b', 'c', 'd', 'e']
    capture = captured_batches(model, value_inputs(), ids)

    summary = coefficient_conditioning_figures(
        DiagnosticWriter(tmp_path), capture, ids, [0, 1, 2, 3, 0], 'validation')

    for head in summary['layers']['attention']['heads']:
        assert head['effective_mean'] == [0., 1., 0.]
        assert head['effective_std'] == [0., 0., 0.]
        assert head['delta_max_absolute'] == [0., 0., 0.]
        assert head['saturation_fraction'] == [0., 0., 0.]
        assert head['gate_frobenius_norm'] == 0.


@pytest.mark.parametrize('bad_delta', [float('nan'), 2.])
def test_rejects_nonfinite_or_out_of_bound_saved_adjustments(tmp_path, bad_delta):
    ids = ['a', 'b', 'c', 'd', 'e']
    capture = captured_batches(model_with_conditioning(), value_inputs(), ids)
    capture.layers['attention']['coefficients'][0][0][0, 0] = bad_delta

    with pytest.raises(ValueError, match='conditioning bound'):
        coefficient_conditioning_figures(DiagnosticWriter(tmp_path), capture, ids, np.zeros(5), 'validation')

    assert not (tmp_path / 'coefficient_conditioning.json').exists()
