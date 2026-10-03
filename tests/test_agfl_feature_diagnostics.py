"""Target-machine checks for feature routing plots and JSON/NPZ axes."""
import json

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from agfl_speech.attention.agfl.layer import AGFL
from agfl_speech.visualization.diagnostics import coefficient_conditioning_figures, filter_figures
from tests.test_agfl_token_diagnostics import TokenWriter, capture_tokens, token_values
from tests.test_agfl_token_routing import nonzero_gate, settings


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('projection', ['none', 'separate'])
def test_feature_capture_preserves_every_coordinate_and_weighted_hop_norm(tmp_path, dtype, projection):
    options = settings(dim=4, heads=2, top_k=None, filter_projection=projection,
                       coefficient_conditioning='feature_contrast')
    model = torch.nn.ModuleDict({'attention': AGFL(options)}).to(dtype=dtype).eval()
    with torch.no_grad():
        model['attention'].qkv.weight.zero_()
        model['attention'].qkv.bias.zero_()
        model['attention'].qkv.weight[8:].copy_(torch.eye(4, dtype=dtype))
        for item in model['attention'].filters:
            nonzero_gate(item)
    values, ids = token_values().to(dtype=dtype), ['z', 'a', 'q', 'b', 'm']
    expected, contributions = [], []
    with torch.no_grad():
        for head, item in enumerate(model['attention'].filters):
            v = values[..., head * 2:(head + 1) * 2]
            one = v.mean(1, keepdim=True).expand_as(v)
            two = one.mean(1, keepdim=True).expand_as(v)
            descriptor = torch.cat((F.layer_norm(v, (2,), eps=1e-5),
                                    F.layer_norm(one - v, (2,), eps=1e-5)), dim=-1)
            raw = F.linear(descriptor, item.coefficient_gate.weight).tanh().reshape(5, 3, 2, 3)
            alpha = item.alpha_logits + .25 * (raw - raw.mean(-1, keepdim=True))
            expected.append(alpha.numpy())
            terms = [(item.W[k](h) if projection == 'separate' else h) * alpha[..., k]
                     for k, h in enumerate((v, one, two))]
            contributions.append(torch.stack([torch.linalg.vector_norm(t, dim=-1) for t in terms], -1).numpy())
        before = model['attention'](values).clone()
    capture = capture_tokens(model, values, ids)
    writer = TokenWriter(tmp_path)
    summary = coefficient_conditioning_figures(writer, capture, ids, [0, 1, 2, 3, 0], 'validation')
    expected = np.stack(expected, axis=1)
    layer = summary['layers']['attention']
    assert summary['schema_version'] == 3
    assert type(layer['zero_sum_check_tolerance']) is float
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    assert json.loads((tmp_path / 'coefficient_conditioning.json').read_text()) == summary
    assert layer['coefficient_array_axes'] == ['trial', 'head', 'token', 'feature', 'hop']
    assert layer['gate_array_axes'] == ['head', 'feature', 'hop', 'descriptor_feature']
    assert layer['within_trial_token_std_axes'] == ['trial', 'head', 'feature', 'hop']
    assert layer['delta_sum_over_hops_axes'] == ['trial', 'head', 'token', 'feature']
    with np.load(tmp_path / layer['array_file'], allow_pickle=False) as saved:
        assert saved['effective_coefficients'].shape == (5, 2, 3, 2, 3)
        assert saved['gate_weights'].shape == (2, 2, 3, 4)
        np.testing.assert_array_equal(saved['sample_ids'], ids)
        np.testing.assert_allclose(saved['effective_coefficients'], expected, atol=2e-6)
        np.testing.assert_allclose(saved['within_trial_token_std'], expected.std(axis=2), atol=2e-6)
        np.testing.assert_allclose(saved['within_token_feature_std'], expected.std(axis=3), atol=2e-6)
        np.testing.assert_allclose(saved['delta_sum_over_hops'], 0, atol=2e-6)
        np.testing.assert_allclose(saved['hop_contribution_norms'], np.stack(contributions, 1), atol=2e-6)
    for head in range(2):
        np.testing.assert_allclose(layer['heads'][head]['effective_mean'],
                                   expected[:, head].mean(axis=(0, 1, 2)), atol=2e-6)
        for hop in range(3):
            displayed = writer.axes[head][0, hop].imshow.call_args.args[0]
            np.testing.assert_allclose(displayed, expected[:, head, :, :, hop].reshape(5, 6), atol=2e-6)
    parameters = filter_figures(writer, model)['attention']
    assert parameters['coefficient_scope'] == 'base_before_feature_adjustment'
    assert all(h['coefficients_effective'] is None for h in parameters['heads'])
    with torch.no_grad():
        torch.testing.assert_close(model['attention'](values), before, rtol=0, atol=0)
