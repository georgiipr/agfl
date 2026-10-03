"""The ``eegnet_original`` backbone: the dataset authors' EEGNet hosting attention.

Development checks for the experiment machine only; they are not a prerequisite
for training. The backbone keeps the authors' layer sizes (temporal kernel 32,
F1=16, D=2, F2=32, pooling 8 then 16), applies their max-norm constraint once
at construction, and lets the selected attention mix the 19 electrodes between
the temporal filters (block1) and the authors' spatial filter (block2).

Run from the project folder: ``python -m pytest tests/test_eegnet_original.py -q``.
"""
from copy import deepcopy

import pytest
import torch

from agfl_speech.config import resolve_config
from agfl_speech.models import get_model_spec
from agfl_speech.models.eegnet_original.model import OriginalEEGNet
from tests.model_fixtures import small_model_options


ATTENTIONS = ('agfl', 'mha', 'hcann')
RELEASE = {'modality': 'eeg', 'channels': 19, 'samples': 500, 'num_classes': 8}
SMALL = {'modality': 'eeg', 'channels': 19, 'samples': 64, 'num_classes': 8}


def build(options=None, attention='mha', heads=2, metadata=None):
    settings = {**small_model_options('eegnet_original'), **(options or {})}
    return get_model_spec('eegnet_original').build(settings, dict(metadata or SMALL), attention, {'heads': heads})


def row_norms(weight):
    return weight.detach().flatten(1).norm(dim=1)


def test_defaults_are_the_authors_settings():
    spec = get_model_spec('eegnet_original')
    assert spec.comparison_family == 'eegnet_original_v1'
    assert set(spec.variants) == {'eeg'}
    assert spec.defaults_for('eeg') == {
        'temp_kernel': 32, 'f1': 16, 'd': 2, 'f2': 32, 'pk1': 8, 'pk2': 16,
        'dropout_rate': .5, 'max_norm1': 1., 'max_norm2': .25, 'max_norm_schedule': 'init',
        'electrode_architecture': 'pre_spatial', 'attention_residual': True, 'attention_dropout': 0.0,
        'batch_norm_momentum': .1, 'batch_norm_eps': 1e-5}
    config = resolve_config({'model': 'eegnet_original'})
    assert config['model_options'] == spec.defaults_for('eeg')
    assert config['comparison_family'] == 'eegnet_original_v1'
    assert config['training']['loss'] == 'cross_entropy'
    # The project EEGNet differs in its settings, starting with the temporal kernel.
    assert get_model_spec('eegnet').defaults_for('eeg')['temp_kernel'] == 250


@pytest.mark.parametrize('attention', ATTENTIONS)
def test_layer_shapes_match_the_authors_eegnet_at_the_released_trial_size(attention):
    model = get_model_spec('eegnet_original').build({}, dict(RELEASE), attention, {})
    assert isinstance(model, OriginalEEGNet)
    statistics = {'weight': (16,), 'bias': (16,), 'running_mean': (16,), 'running_var': (16,),
                  'num_batches_tracked': ()}
    wide = {name: (32,) if shape else () for name, shape in statistics.items()}
    expected = {
        'block1.0.weight': (16, 1, 1, 32),    # temporal filters
        'block2.0.weight': (32, 1, 19, 1),    # depthwise spatial filter over all electrodes
        'block3.0.weight': (32, 1, 1, 16),    # separable convolution: depthwise ...
        'block3.1.weight': (32, 32, 1, 1),    # ... then pointwise
        'fc.weight': (8, 96), 'fc.bias': (8,),  # 32 features x (500 // 8 // 16 = 3) steps
        **{f'block1.1.{name}': shape for name, shape in statistics.items()},
        **{f'block2.1.{name}': shape for name, shape in wide.items()},
        **{f'block3.2.{name}': shape for name, shape in wide.items()},
    }
    state = model.state_dict()
    authors = {name: tuple(value.shape) for name, value in state.items()
               if name.split('.')[0] in {'block1', 'block2', 'block3', 'fc'}}
    assert authors == expected
    # Everything else belongs to the hosted attention and its electrode identity.
    others = set(state) - set(authors)
    assert 'electrode_position.embedding' in others
    assert tuple(state['electrode_position.embedding'].shape) == (1, 19, 16)
    assert all(name == 'electrode_position.embedding' or name.startswith('attn_blocks.0.') for name in others)
    assert model.token_axis == 'electrode' and model.num_tokens == 19
    assert model.attn_blocks[0].attention_key == attention
    assert model.attn_blocks[0].options['dim'] == 16 and model.attn_blocks[0].options['heads'] == 4
    model.eval()
    with torch.no_grad():
        logits = model(torch.randn(2, 19, 500))
    assert logits.shape == (2, 8) and torch.isfinite(logits).all()


def test_max_norm_is_applied_once_at_construction():
    torch.manual_seed(0)
    model = build()
    assert model.max_norm_schedule == 'init'
    # PyTorch's default classifier rows are longer than 0.25, so the constraint acts.
    torch.testing.assert_close(row_norms(model.fc.weight), torch.full((8,), .25), rtol=0, atol=1e-5)
    assert row_norms(model.block2[0].weight).max() <= 1 + 1e-6
    with torch.no_grad():
        model.fc.weight.fill_(1.)
        model.block2[0].weight.fill_(1.)
    classifier, spatial = model.fc.weight.clone(), model.block2[0].weight.clone()
    for _ in range(2):
        model.clip_weights()
    torch.testing.assert_close(model.fc.weight, classifier, rtol=0, atol=0)
    torch.testing.assert_close(model.block2[0].weight, spatial, rtol=0, atol=0)
    # A copy (as made for EMA weights) keeps the once-only state.
    copied = deepcopy(model)
    copied.clip_weights()
    torch.testing.assert_close(copied.fc.weight, classifier, rtol=0, atol=0)


def test_every_step_schedule_reapplies_max_norm():
    torch.manual_seed(0)
    model = build({'max_norm_schedule': 'every_step'})
    assert model.max_norm_schedule == 'every_step'
    torch.testing.assert_close(row_norms(model.fc.weight), torch.full((8,), .25), rtol=0, atol=1e-5)
    with torch.no_grad():
        model.fc.weight.fill_(1.)
        model.block2[0].weight.fill_(1.)
    model.clip_weights()
    torch.testing.assert_close(row_norms(model.fc.weight), torch.full((8,), .25), rtol=0, atol=1e-5)
    torch.testing.assert_close(row_norms(model.block2[0].weight), torch.ones(8), rtol=0, atol=1e-5)
    for schedule in ('sometimes', None):
        with pytest.raises(ValueError, match='max_norm_schedule'):
            build({'max_norm_schedule': schedule})


def test_separable_convolution_requires_f2_equal_to_f1_times_d():
    with pytest.raises(ValueError, match='f2 == f1'):
        build({'f2': 12})
    with pytest.raises(ValueError, match='f2 == f1'):
        build({'d': 1})
    assert build({'f1': 6, 'd': 2, 'f2': 12}).fc.in_features == 12 * (64 // 2 // 2)


@pytest.mark.parametrize('layout', ['spatial_fusion', 'compact', 'temporal'])
def test_attention_position_is_fixed_before_the_spatial_filter(layout):
    with pytest.raises(ValueError, match='pre_spatial'):
        build({'electrode_architecture': layout})


@pytest.mark.parametrize('options,message', [
    ({'f2': 16}, 'f2 == f1'), ({'d': 4}, 'f2 == f1'), ({'f1': 8}, 'f2 == f1'),
    ({'electrode_architecture': 'spatial_fusion'}, 'pre_spatial'),
    ({'electrode_architecture': 'compact'}, 'pre_spatial'),
    ({'max_norm_schedule': 'sometimes'}, 'max_norm_schedule'),
])
def test_invalid_settings_fail_when_the_configuration_is_resolved(options, message):
    with pytest.raises(ValueError, match=message):
        resolve_config({'model': 'eegnet_original', 'model_options': options})


def test_saved_checkpoint_reconstruction_keeps_the_constructor_checks():
    spec = get_model_spec('eegnet_original')
    options = small_model_options('eegnet_original')
    rebuilt = spec.rebuild_saved(options, dict(SMALL), 'mha', {'heads': 2})
    assert rebuilt.state_dict().keys() == build().state_dict().keys()
    for changed, message in (({'f2': 12}, 'f2 == f1'), ({'electrode_architecture': 'compact'}, 'pre_spatial'),
                             ({'max_norm_schedule': 'sometimes'}, 'max_norm_schedule')):
        with pytest.raises(ValueError, match=message):
            spec.rebuild_saved({**options, **changed}, dict(SMALL), 'mha', {'heads': 2})


def test_options_of_the_project_eegnet_are_not_accepted():
    for option in ('electrode_dim', 'attention_axis', 'spatial_readout', 'initialization', 'temporal_statistics'):
        with pytest.raises(ValueError, match='Unknown eegnet_original model_options'):
            build({option: None})
        with pytest.raises(ValueError, match='Unknown options for eegnet_original'):
            resolve_config({'model': 'eegnet_original', 'model_options': {option: None}})


@pytest.mark.parametrize('attention', ATTENTIONS)
def test_attention_sees_one_electrode_node_per_time_step(attention):
    torch.manual_seed(1)
    model = build(attention=attention).eval()
    batch, samples, f1 = 3, SMALL['samples'], 4
    captured = []
    handle = model.attn_blocks[0].register_forward_hook(
        lambda module, arguments, output: captured.append((arguments[0].detach(), output.detach())))
    x = torch.randn(batch, 19, samples)
    with torch.no_grad():
        logits = model(x)
        temporal = model.block1(x.unsqueeze(1))  # [B, F1, electrodes, T]
    handle.remove()
    assert logits.shape == (batch, 8)
    assert len(captured) == 1
    tokens, mixed = captured[0]
    assert tokens.shape == (batch * samples, 19, f1)
    assert mixed.shape == tokens.shape
    # Trial-major rows: row b * T + t holds the 19 electrodes of trial b at time t,
    # each with its own F1 temporal-filter outputs plus the electrode identity.
    identity = model.electrode_position.embedding[0]
    for trial in range(batch):
        for step in (0, samples // 2, samples - 1):
            torch.testing.assert_close(tokens[trial * samples + step],
                                       temporal[trial, :, :, step].T + identity, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize('attention', ATTENTIONS)
def test_it_is_the_project_eegnet_pre_spatial_layout_with_other_settings(attention):
    torch.manual_seed(13)
    original = build(attention=attention)
    torch.manual_seed(13)
    project = get_model_spec('eegnet').build(
        {**small_model_options('eegnet'), 'electrode_architecture': 'pre_spatial'},
        dict(SMALL), attention, {'heads': 2})
    assert original.state_dict().keys() == project.state_dict().keys()
    for name, value in project.state_dict().items():
        torch.testing.assert_close(original.state_dict()[name], value, rtol=0, atol=0)
    x = torch.randn(3, 19, SMALL['samples'])
    with torch.no_grad():
        torch.testing.assert_close(original.eval()(x), project.eval()(x), rtol=0, atol=0)


@pytest.mark.parametrize('attention', ATTENTIONS)
def test_heads_must_divide_f1_when_the_configuration_is_resolved(attention):
    config = {'model': 'eegnet_original', 'attention': attention}
    assert resolve_config({**config, 'attention_options': {'heads': 8}})['attention_options']['heads'] == 8
    with pytest.raises(ValueError, match='f1 must be divisible'):
        resolve_config({**config, 'attention_options': {'heads': 3}})
    with pytest.raises(ValueError, match='f1 must be divisible'):
        resolve_config({**config, 'model_options': {'f1': 6, 'f2': 12}})


def test_residual_and_gradients_pass_through_the_authors_spatial_filter():
    torch.manual_seed(9)
    model = build(attention='agfl')
    logits = model(torch.randn(4, 19, SMALL['samples']))
    torch.nn.functional.cross_entropy(logits, torch.arange(4)).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.block1[0].weight.grad.abs().sum() > 0
    assert model.block2[0].weight.grad.abs().sum() > 0
    assert model.attn_blocks[0].proj.bias.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='attention_residual'):
        build({'attention_residual': 'yes'})
