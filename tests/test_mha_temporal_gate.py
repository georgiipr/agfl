"""Optional target-machine checks; no added dependencies or training prerequisite."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from agfl.attention.mha.layer import MultiHeadAttention
from agfl.visualization.maps import mixer_map
from agfl.visualization.common import FigureWriter
from agfl.visualization.temporal_gate import TemporalGateCapture, temporal_gate_figures, validation_ablation_figures


class MatchedMHAControls(unittest.TestCase):
    def test_neutral_additions_preserve_rng_checkpoint_weights_and_function(self):
        torch.manual_seed(91)
        plain = MultiHeadAttention(32, 4).double()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(91)
        augmented = MultiHeadAttention(32, 4, temporal_bias=True, output_gate=True).double()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(set(plain.state_dict()), {'qkv.weight', 'qkv.bias', 'proj.weight', 'proj.bias'})
        self.assertEqual(sum(p.numel() for p in augmented.parameters()) - sum(p.numel() for p in plain.parameters()), 80)
        for key, value in plain.state_dict().items():
            torch.testing.assert_close(augmented.state_dict()[key], value, rtol=0, atol=0)
        x = torch.randn(3, 7, 32, dtype=torch.float64)
        a, b = plain(x), augmented(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward()
        b.square().sum().backward()
        for key, value in plain.named_parameters():
            torch.testing.assert_close(dict(augmented.named_parameters())[key].grad, value.grad, rtol=1e-10, atol=1e-10)
        for key, value in augmented.named_parameters():
            if 'temporal_bias' in key or 'output_gates' in key:
                self.assertTrue(torch.isfinite(value.grad).all())
                self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_learned_additions_match_explicit_formula_and_map(self):
        layer = MultiHeadAttention(8, 2, temporal_bias=True, output_gate=True).double()
        with torch.no_grad():
            layer.temporal_bias.weight.copy_(torch.tensor([[-.3, .2, .7], [-.1, -.4, .2]]))
            for head, gate in enumerate(layer.output_gates):
                gate.weight.copy_(torch.arange(8).reshape(1, 8) * .01 * (head + 1))
                gate.bias.fill_(.1 * head)
        x = torch.randn(3, 5, 8, dtype=torch.float64)
        qkv = F.linear(x, layer.qkv.weight, layer.qkv.bias).reshape(3, 5, 3, 2, 4).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        lag = (torch.arange(5)[None, :] - torch.arange(5)[:, None]).double() / 4
        w = layer.temporal_bias.weight
        bias = w[:, 0, None, None] * lag.abs() + w[:, 1, None, None] * lag + w[:, 2, None, None] * torch.eye(5)
        attention = (q @ k.transpose(-1, -2) / 2 + bias).softmax(-1)
        message = attention @ v
        heads = []
        for h, gate in enumerate(layer.output_gates):
            descriptor = torch.cat((F.layer_norm(v[:, h], (4,)), F.layer_norm(message[:, h] - v[:, h], (4,))), dim=-1)
            heads.append(message[:, h] * (2 * F.linear(descriptor, gate.weight, gate.bias).sigmoid()))
        expected = F.linear(torch.cat(heads, dim=-1), layer.proj.weight, layer.proj.bias)
        torch.testing.assert_close(layer(x), expected)
        graph, description = mixer_map(layer, x)
        torch.testing.assert_close(graph, attention)
        self.assertIn('before output gating', description)
        layer.token_axis = 'electrode'
        with self.assertRaisesRegex(ValueError, 'time tokens'):
            layer(x)

    def test_diagnostics_capture_gates_and_restore_parameters_after_removal(self):
        layer = MultiHeadAttention(8, 2, temporal_bias=True, output_gate=True)
        layer.attention_key = 'mha'
        model = nn.Sequential(layer, nn.Flatten(), nn.Linear(40, 4)).eval()
        with torch.no_grad():
            layer.temporal_bias.weight[:, 2] = .4
            for gate in layer.output_gates:
                gate.bias.fill_(.3)
        x = np.random.default_rng(13).normal(size=(5, 5, 8)).astype(np.float32)
        ids, labels = [f'trial-{i}' for i in range(5)], np.array([0, 1, 2, 3, 0])
        capture = TemporalGateCapture(model)
        try:
            capture.start_batch(ids)
            with torch.no_grad():
                probabilities = model(torch.from_numpy(x)).softmax(-1).numpy()
        finally:
            for hook in capture.hooks:
                hook.remove()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            writer = FigureWriter(directory)
            info = temporal_gate_figures(writer, capture, ids, labels, 'validation')['0']
            self.assertEqual(info['attention'], 'mha')
            with np.load(Path(directory) / info['arrays']) as saved:
                self.assertEqual(saved['sample_ids'].tolist(), ids)
                self.assertEqual(saved['graphs'].shape, (5, 2, 5, 5))
                np.testing.assert_allclose(saved['graphs'].sum(-1), 1, atol=1e-6)
                np.testing.assert_allclose(saved['gates'], 2 / (1 + np.exp(-.3)), rtol=1e-6)
            result = validation_ablation_figures(writer, model, x, ids, labels, probabilities, 'cpu', 'validation')
            self.assertEqual(len(result['ablations']), 3)
            json.loads((Path(directory) / 'temporal_gate/validation_ablation.json').read_text())
            self.assertIsNone(validation_ablation_figures(writer, model, x, ids, labels, probabilities, 'cpu', 'test'))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], atol=0, rtol=0)

    def test_flags_are_explicit_booleans(self):
        for flags in ({'temporal_bias': 1}, {'output_gate': 'yes'}):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                MultiHeadAttention(8, 2, **flags)


if __name__ == '__main__':
    unittest.main()
