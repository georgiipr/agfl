"""Target-machine checks of saved AGFL parameters; no plotting dependency needed."""
from copy import deepcopy
import json
import math
from unittest.mock import Mock

import pytest
import torch

from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL
from agfl.visualization.diagnostics import filter_figures


class ParameterFigureWriter:
    """Capture figure registration while exercising the real JSON writer."""

    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.plt = Mock()
        def subplots(*args, **kwargs):
            axes = [[Mock() for _ in range(args[1])]] if kwargs.get('squeeze') is False else Mock()
            return Mock(), axes
        self.plt.subplots.side_effect = subplots
        self.figures = []

    def save(self, figure, name, caption):
        self.figures.append({'name': name, 'caption': caption})


def parameter_model(**updates):
    options = {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2, 'K': 2,
               'projection': 'qkv', 'filter_projection': 'none',
               'score_scaling': 'temperature', 'temperature_init': .5,
               'coefficient_init': 'identity_one_hop',
               'coefficient_activation': 'identity'}
    options.update(updates)
    return torch.nn.ModuleDict({'attention': AGFL(options)}).double()


def test_statistics_use_loaded_parameters_and_preserve_checkpoint_state(tmp_path):
    original = parameter_model()
    with torch.no_grad():
        original['attention'].filters[0].alpha_logits.copy_(torch.tensor([-.2, .8, .4]))
        original['attention'].filters[1].alpha_logits.copy_(torch.tensor([1.1, .3, -.1]))
        original['attention'].builders[0].temperature.fill_(.04)
        original['attention'].builders[1].temperature.fill_(6.)
    checkpoint_path = tmp_path / 'checkpoint.pt'
    torch.save({'model': original.state_dict()}, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    loaded = parameter_model()
    loaded.load_state_dict(torch.load(checkpoint_path, weights_only=True)['model'], strict=True)
    before = {key: value.clone() for key, value in loaded.state_dict().items()}
    options_before = deepcopy(loaded['attention'].options)
    writer = ParameterFigureWriter(tmp_path / 'diagnostics')

    statistics = filter_figures(writer, loaded)

    assert json.loads((writer.root / 'filter_statistics.json').read_text()) == statistics
    layer = statistics['attention']
    assert layer['coefficient_init'] == 'identity_one_hop'
    assert layer['temperature_init'] == .5
    assert layer['temperature_active'] is True
    assert layer['temperature_clamp_bounds'] == [.1, 5.]
    for head, expected_alpha, raw, effective, status in zip(
            layer['heads'], [[-.2, .8, .4], [1.1, .3, -.1]], [.04, 6.], [.1, 5.],
            ['below_minimum', 'above_maximum']):
        assert head['coefficients_raw'] == pytest.approx(expected_alpha)
        assert head['coefficients_effective'] == pytest.approx(expected_alpha)
        assert head['coefficients_learnable'] is True
        assert head['temperature_raw'] == pytest.approx(raw)
        assert head['temperature_effective'] == pytest.approx(effective)
        assert head['temperature_active'] is True
        assert head['temperature_learnable'] is True
        assert head['temperature_clamped'] is True
        assert head['temperature_clamp_status'] == status
    assert [figure['name'] for figure in writer.figures] == [
        'filters/attention_coefficients', 'filters/attention_temperature']
    assert checkpoint_path.read_bytes() == checkpoint_bytes
    assert loaded['attention'].options == options_before
    for key, expected in before.items():
        torch.testing.assert_close(loaded.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize('activation,expected', [
    ('identity', [-2., 0., 2.]),
    ('relu', [0., 0., 2.]),
    ('sigmoid', [1 / (1 + math.exp(2)), .5, 1 / (1 + math.exp(-2))]),
    ('softmax', [math.exp(-2) / (math.exp(-2) + 1 + math.exp(2)),
                 1 / (math.exp(-2) + 1 + math.exp(2)),
                 math.exp(2) / (math.exp(-2) + 1 + math.exp(2))]),
])
def test_statistics_report_activated_coefficients_and_inactive_temperature(tmp_path, activation, expected):
    model = parameter_model(coefficient_init='uniform', coefficient_activation=activation,
                            learnable_coefficients=False, score_scaling='sqrt_dim', temperature_init=1.)
    with torch.no_grad():
        for graph_filter, builder in zip(model['attention'].filters, model['attention'].builders):
            graph_filter.alpha_logits.copy_(torch.tensor([-2., 0., 2.]))
            builder.temperature.fill_(6.)
    writer = ParameterFigureWriter(tmp_path)

    layer = filter_figures(writer, model)['attention']

    assert layer['temperature_active'] is False
    assert layer['temperature_clamp_bounds'] is None
    for head in layer['heads']:
        assert head['coefficients_raw'] == [-2., 0., 2.]
        assert head['coefficients_effective'] == pytest.approx(expected)
        assert head['coefficients_learnable'] is False
        assert head['temperature_raw'] == 6.
        assert head['temperature_active'] is False
        assert head['temperature_effective'] is None
        assert head['temperature_learnable'] is False
        assert head['temperature_clamped'] is False
        assert head['temperature_clamp_status'] == 'inactive'
    assert [figure['name'] for figure in writer.figures] == ['filters/attention_coefficients']


def test_clamp_boundaries_are_distinct_from_outside_values(tmp_path):
    model = parameter_model()
    with torch.no_grad():
        model['attention'].builders[0].temperature.fill_(.1)
        model['attention'].builders[1].temperature.fill_(5.)
    writer = ParameterFigureWriter(tmp_path)

    heads = filter_figures(writer, model)['attention']['heads']

    assert [head['temperature_clamp_status'] for head in heads] == ['lower_boundary', 'upper_boundary']
    assert all(head['temperature_clamped'] is False for head in heads)
    assert [head['temperature_effective'] for head in heads] == [.1, 5.]


def test_older_options_without_initial_temperature_use_historical_default(tmp_path):
    model = parameter_model(coefficient_init='one_hop', temperature_init=1.)
    model['attention'].options.pop('temperature_init')
    with torch.no_grad():
        model['attention'].builders[0].temperature.fill_(.7)
    before = deepcopy(model['attention'].options)
    writer = ParameterFigureWriter(tmp_path)

    layer = filter_figures(writer, model)['attention']

    assert layer['temperature_init'] == 1.
    assert layer['heads'][0]['temperature_raw'] == pytest.approx(.7)
    assert layer['heads'][0]['temperature_clamp_status'] == 'interior'
    assert model['attention'].options == before


def test_baseline_model_gets_no_agfl_statistics_or_temperature_figure(tmp_path):
    writer = ParameterFigureWriter(tmp_path)

    assert filter_figures(writer, torch.nn.Linear(8, 8)) == {}

    assert not (tmp_path / 'filter_statistics.json').exists()
    assert writer.figures == []


def test_projection_statistics_measure_checkpoint_weights_after_identity_initialization(tmp_path):
    model = parameter_model(filter_projection='separate', filter_projection_init='identity')
    with torch.no_grad():
        # A learned off-diagonal entry must be visible even though the declared
        # initialization is identity. Every other projection remains identity.
        model['attention'].filters[0].W[2].weight[0, 1] = .25
    writer = ParameterFigureWriter(tmp_path)

    layer = filter_figures(writer, model)['attention']

    assert layer['filter_projection'] == 'separate'
    assert layer['filter_projection_init'] == 'identity'
    for head_index, head in enumerate(layer['heads']):
        assert len(head['projections']) == 3
        for hop, projection in enumerate(head['projections']):
            changed = head_index == 0 and hop == 2
            assert projection['projection_index'] == projection['hop_order'] == hop
            assert projection['shape'] == [4, 4]
            assert projection['learnable'] is True
            assert projection['frobenius_norm'] == pytest.approx(math.sqrt(4 + .25 ** 2) if changed else 2.)
            assert projection['distance_from_identity'] == pytest.approx(.25 if changed else 0.)
    names = [figure['name'] for figure in writer.figures]
    assert 'filters/attention_head_0' in names and 'filters/attention_head_1' in names


def test_old_projection_options_report_historical_initialization_without_mutation(tmp_path):
    model = parameter_model(filter_projection='shared')
    model['attention'].options.pop('filter_projection_init', None)
    with torch.no_grad():
        for graph_filter in model['attention'].filters:
            graph_filter.W.weight.zero_()
    before = deepcopy(model['attention'].options)
    writer = ParameterFigureWriter(tmp_path)

    layer = filter_figures(writer, model)['attention']

    assert layer['filter_projection_init'] == 'pytorch'
    assert model['attention'].options == before
    for head in layer['heads']:
        assert len(head['projections']) == 1
        projection = head['projections'][0]
        assert projection['hop_order'] is None
        assert projection['frobenius_norm'] == 0.
        assert projection['distance_from_identity'] == 2.
