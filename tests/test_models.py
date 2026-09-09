"""Numerical checks to execute on the target machine; not run locally."""
from copy import deepcopy
import pytest
import torch
from agfl.models import available_models, get_model_spec
from agfl.attention.agfl.config import DEFAULTS
from agfl.attention.agfl.layer import AGFL, propagate, topk_mask, normalize_graph
from agfl.attention.mha.layer import MultiHeadAttention
from agfl.attention.performer.layer import Performer
from agfl.attention.linformer.layer import Linformer
from agfl.attention.nystromformer.layer import Nystromformer
from tests.references.agfl import AGFL as HistoricalAGFL

from agfl.attention import available_attentions
from tests.model_fixtures import small_model_options

ATTENTIONS = ['agfl', 'mha', 'performer', 'linformer', 'nystromformer']
MODELS = ['eegnet', 'eegencoder', 'dstseegencoder', 'conformer', 'signal_transformer']


def small_options(**extra):
    return {**deepcopy(DEFAULTS), 'dim': 8, 'heads': 2, 'depth': 2, **extra}


@pytest.mark.parametrize('degree', [0, 1, 2, 4])
def test_default_agfl_preserves_initialization_output_and_gradient(degree):
    torch.manual_seed(57)
    old = HistoricalAGFL(8, 2, degree).double()
    torch.manual_seed(57)
    new = AGFL(small_options(K=degree)).double()
    assert old.state_dict().keys() == new.state_dict().keys()
    for key, expected in old.state_dict().items():
        torch.testing.assert_close(new.state_dict()[key], expected, rtol=0, atol=0)
    # Nonzero signed taps exercise graph/filter gradients rather than merely
    # comparing two zero-initialized layers' projection biases.
    with torch.no_grad():
        for head in old.filters:
            head.alpha_logits.copy_(torch.linspace(-.2, .7, degree + 1))
    new.load_state_dict(old.state_dict())
    x = torch.randn(2, 11, 8, dtype=torch.float64)
    a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
    expected, actual = old(a, 1, 3), new(b, 1, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(new.last_adj, old.last_adj, rtol=0, atol=0)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(a.grad, b.grad, rtol=1e-12, atol=1e-12)
    for (name, p), (_, q) in zip(old.named_parameters(), new.named_parameters()):
        if p.grad is None:
            assert q.grad is None, name
        else:
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)


def test_polynomial_and_renormalization_match_independent_formulas():
    a = torch.tensor([[[.8, .2], [.3, .7]]], dtype=torch.float64)
    x = torch.tensor([[[2., 1.], [1., -3.]]], dtype=torch.float64)
    taps = list(propagate(a, x, 3, 'polynomial'))
    for degree, tap in enumerate(taps):
        torch.testing.assert_close(tap, torch.linalg.matrix_power(a, degree) @ x)
    taps = list(propagate(a, x, 3, 'renormalized'))
    previous = x
    for tap in taps[1:]:
        expected = a @ previous
        expected *= x.norm(dim=-1, keepdim=True) / (expected.norm(dim=-1, keepdim=True) + 1e-6)
        torch.testing.assert_close(tap, expected)
        previous = expected
    assert not torch.allclose(taps[-1], torch.linalg.matrix_power(a, 3) @ x)


def test_topk_ties_and_softmax_order_are_explicit():
    tied = torch.ones(1, 4, 4)
    assert topk_mask(tied, 2, 'threshold').sum() == 16
    exact = topk_mask(tied, 2, 'exact')
    assert (exact.sum(-1) == 2).all()
    scores = torch.tensor([[[1., 4., 3., 2.]]])
    before = topk_mask(scores, 2)
    after = topk_mask(scores.softmax(-1), 2)
    assert torch.equal(before, after)
    a = normalize_graph(scores, before, 'softmax', 'scores', True)
    b = normalize_graph(scores, after, 'softmax', 'softmax', True)
    torch.testing.assert_close(a, b)
    unnormalized = normalize_graph(scores, after, 'softmax', 'softmax', False)
    assert unnormalized.sum() < 1


def test_none_means_dense_topk_and_zero_coefficients_are_preserved():
    layer = AGFL(small_options(top_k=None))
    x = torch.randn(2, 5, 8)
    output = layer(x)
    assert (layer.last_adj > 0).all()
    for head in layer.filters:
        assert torch.count_nonzero(head.alpha_logits) == 0
    torch.testing.assert_close(output, layer.proj.bias.expand_as(output))


@pytest.mark.parametrize('key', MODELS)
@pytest.mark.parametrize('attention', ATTENTIONS)
@pytest.mark.parametrize('modality,channels', [('eeg', 6), ('ecg', 1)])
def test_each_model_injects_selected_attention_and_backpropagates(key, attention, modality, channels):
    spec = get_model_spec(key)
    assert set(spec.variants) == {'eeg', 'ecg'}
    metadata = {'modality': modality, 'channels': channels, 'samples': 65, 'num_classes': 3}
    model = spec.build(small_model_options(key, modality), metadata, attention, {'heads': 2})
    x = torch.randn(3, channels, 65)
    logits = model(x)
    assert logits.shape == (3, 3)
    assert model.token_axis == ('electrode' if modality == 'eeg' else 'time_patch' if key == 'signal_transformer' else 'time')
    layers = [m for m in model.modules() if getattr(m, 'is_attention', False)]
    assert layers and all(m.attention_key == attention for m in layers)
    assert all(m.token_axis == model.token_axis for m in layers)
    if modality == 'eeg':
        assert all(m.num_tokens == channels for m in layers)
    else:
        assert all(m.num_tokens > 1 for m in layers)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    with pytest.raises(ValueError, match='Expected'):
        model(x.transpose(1, 2))


@pytest.mark.parametrize('key', MODELS)
@pytest.mark.parametrize('modality', ['eeg', 'ecg'])
def test_backbone_initialization_is_independent_of_attention_capacity(key, modality):
    states = []
    for attention in ATTENTIONS:
        torch.manual_seed(41)
        spec = get_model_spec(key)
        model = spec.build(small_model_options(key, modality),
                           {'modality': modality, 'channels': 6, 'samples': 65, 'num_classes': 4},
                           attention, {'heads': 2})
        prefixes = [name + '.' for name, module in model.named_modules() if getattr(module, 'is_attention', False)]
        states.append({k: v for k, v in model.state_dict().items() if not any(k.startswith(p) for p in prefixes)})
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for name in state:
            torch.testing.assert_close(state[name], states[0][name], rtol=0, atol=0)


def test_mha_matches_torch_reference():
    layer = MultiHeadAttention(8, 2).double()
    reference = torch.nn.MultiheadAttention(8, 2, batch_first=True, dropout=0).double()
    with torch.no_grad():
        reference.in_proj_weight.copy_(layer.qkv.weight)
        reference.in_proj_bias.copy_(layer.qkv.bias)
        reference.out_proj.weight.copy_(layer.proj.weight)
        reference.out_proj.bias.copy_(layer.proj.bias)
    x = torch.randn(2, 7, 8, dtype=torch.float64)
    torch.testing.assert_close(layer(x), reference(x, x, x, need_weights=True)[0])


def test_linformer_identity_and_nystrom_full_landmarks_recover_attention():
    options = {'dim': 8, 'heads': 2, 'projection_rank': 5, 'landmarks': 5}
    mha = MultiHeadAttention(8, 2)
    linformer = Linformer(options, 5)
    nystrom = Nystromformer(options)
    for layer in (linformer, nystrom):
        layer.qkv.load_state_dict(mha.qkv.state_dict())
        layer.proj.load_state_dict(mha.proj.state_dict())
    with torch.no_grad():
        linformer.E.copy_(torch.eye(5).expand(2, 5, 5))
        linformer.F.copy_(torch.eye(5).expand(2, 5, 5))
    x = torch.randn(2, 5, 8)
    torch.testing.assert_close(linformer(x), mha(x), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(nystrom(x), mha(x), rtol=1e-5, atol=1e-6)


def test_performer_contraction_matches_explicit_feature_attention():
    layer = Performer({'dim': 8, 'heads': 2, 'random_features': 16})
    x = torch.randn(2, 6, 8)
    q, k, v = layer.project(x)
    q, k = layer.feature_map(q, True), layer.feature_map(k, False)
    weights = q @ k.transpose(-1, -2)
    weights /= weights.sum(-1, keepdim=True)
    torch.testing.assert_close(layer(x), layer.combine(weights @ v), rtol=1e-5, atol=1e-6)
    assert 'features' in layer.state_dict()


def test_no_attention_is_not_a_selectable_model():
    assert 'none' not in available_models()
    assert 'none' not in available_attentions()
    with pytest.raises(ValueError, match='Unknown model'):
        get_model_spec('none')


def test_spatial_readout_can_distinguish_opposite_electrode_patterns():
    from agfl.models._shared.layers import ElectrodeReadout
    readout = ElectrodeReadout(2, 1)
    with torch.no_grad():
        readout.projection.weight.copy_(torch.tensor([[[1., -1.]]]))
    tokens = torch.tensor([[[2.], [-2.]], [[-2.], [2.]]])
    torch.testing.assert_close(tokens.mean(1), torch.zeros(2, 1))
    torch.testing.assert_close(readout(tokens), torch.tensor([[4.], [-4.]]))


def test_eegnet_zero_initialized_agfl_does_not_block_feature_gradients():
    model = get_model_spec('eegnet').build(small_model_options('eegnet', 'eeg'),
        {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 3}, 'agfl', {'heads': 2})
    loss = torch.nn.functional.cross_entropy(model(torch.randn(3, 4, 65)), torch.tensor([0, 1, 2]))
    loss.backward()
    assert model.block1[0].weight.grad.abs().sum() > 0
    assert model.spatial_readout.projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('key', MODELS)
def test_all_eeg_models_have_learned_spatial_readouts(key):
    from agfl.models._shared.layers import ElectrodeReadout
    model = get_model_spec(key).build(small_model_options(key, 'eeg'),
        {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 3}, 'mha', {'heads': 2})
    readouts = [m for m in model.modules() if isinstance(m, ElectrodeReadout)]
    assert readouts and all(m.projection is not None for m in readouts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA check for target environment')
@pytest.mark.parametrize('key', MODELS)
@pytest.mark.parametrize('attention', ATTENTIONS)
def test_cuda_autocast_keeps_gradients_finite(key, attention):
    spec = get_model_spec(key)
    model = spec.build(small_model_options(key, 'ecg'),
                       {'modality': 'ecg', 'channels': 1, 'samples': 256, 'num_classes': 2},
                       attention, {'heads': 2}).cuda()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 1, 256, device='cuda')),
                                                torch.tensor([0, 1], device='cuda'))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
