"""Target-machine checks only; unittest adds no dependency or launch prerequisite."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from agfl_speech.attention.agfl.config import DEFAULTS
from agfl_speech.attention.agfl.layer import AGFL
from agfl_speech.visualization.common import FigureWriter
from agfl_speech.visualization.temporal_gate import TemporalGateCapture, temporal_gate_figures, validation_ablation_figures

# Figures need the optional 'plots' extra; the layer checks above them do not.
HAS_MATPLOTLIB = importlib.util.find_spec('matplotlib') is not None


def options(**changes):
    settings = deepcopy(DEFAULTS)
    settings.update(dim=32, heads=4, projection='qkv', qkv_bias=True, filter_projection='none',
                    score_scaling='sqrt_dim', coefficient_init='one_hop', top_k=5,
                    coefficient_conditioning='token_contrast')
    return {**settings, **changes}


class TemporalGateLayerTests(unittest.TestCase):
    def test_neutral_initialization_keeps_rng_shared_weights_outputs_and_gradients(self):
        for bias, gate, added in [(True, False, 12), (False, True, 68), (True, True, 80)]:
            with self.subTest(bias=bias, gate=gate):
                torch.manual_seed(19)
                control = AGFL(options()).double()
                rng = torch.get_rng_state().clone()
                torch.manual_seed(19)
                candidate = AGFL(options(temporal_bias=bias, output_gate=gate)).double()
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(sum(p.numel() for p in candidate.parameters()) -
                                 sum(p.numel() for p in control.parameters()), added)
                for name, value in control.state_dict().items():
                    torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
                x = torch.randn(3, 7, 32, dtype=torch.float64)
                a, b = control(x), candidate(x)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                a.square().sum().backward()
                b.square().sum().backward()
                candidate_parameters = dict(candidate.named_parameters())
                for name, p in control.named_parameters():
                    if p.requires_grad:
                        torch.testing.assert_close(p.grad, candidate_parameters[name].grad, rtol=1e-10, atol=1e-10)
                for name, p in candidate_parameters.items():
                    if 'temporal_bias' in name or 'output_gate' in name:
                        self.assertTrue(torch.isfinite(p.grad).all())
                        self.assertGreater(float(p.grad.abs().sum()), 0)

    def test_omitted_flags_keep_old_checkpoint_keys(self):
        previous = options()
        previous.pop('temporal_bias')
        previous.pop('output_gate')
        old = AGFL(previous)
        current = AGFL(options())
        current.load_state_dict(old.state_dict(), strict=True)
        self.assertFalse(any('temporal_bias' in k or 'output_gate' in k for k in current.state_dict()))
        x = torch.randn(2, 7, 32)
        torch.testing.assert_close(old(x), current(x), rtol=0, atol=0)

    def test_bias_precedes_mask_selection(self):
        model = AGFL(options(temporal_bias=True, top_k=1))
        with torch.no_grad():
            model.temporal_bias.weight[:, 2] = 10
        q = torch.zeros(2, 7, 8)
        graph = model.construct_graph(0, q, q, 0, 1)
        torch.testing.assert_close(graph, torch.eye(7).expand(2, -1, -1), rtol=0, atol=0)
        model.token_axis = 'electrode'
        with self.assertRaisesRegex(ValueError, 'time tokens'):
            model(torch.randn(2, 7, 32))

    def test_output_gate_scales_message_before_output_projection(self):
        control = AGFL(options())
        candidate = AGFL(options(output_gate=True))
        candidate.load_state_dict(control.state_dict(), strict=False)
        with torch.no_grad():
            for f in candidate.filters:
                f.output_gate.bias.fill_(float(np.log(3)))  # g = 1.5
        x = torch.randn(2, 7, 32)
        expected = (control(x) - control.proj.bias) * 1.5 + control.proj.bias
        torch.testing.assert_close(candidate(x), expected)

    def test_invalid_modes_are_rejected(self):
        for changes in ({'output_gate': 'yes'}, {'temporal_bias': 1},
                        {'output_gate': True, 'coefficient_conditioning': 'static'},
                        {'temporal_bias': True, 'top_k_stage': 'raw_scores'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                AGFL(options(**changes))

    @unittest.skipUnless(HAS_MATPLOTLIB, "plotting requires the 'plots' extra")
    def test_diagnostics_keep_ids_native_json_and_restore_ablation_parameters(self):
        attention = AGFL(options(temporal_bias=True, output_gate=True))
        # A model sets both attributes through its attention factory; the
        # temporal bias exists only for time tokens.
        attention.attention_key, attention.token_axis = 'agfl', 'time'
        model = nn.Sequential(attention, nn.Flatten(), nn.Linear(224, 4)).eval()
        with torch.no_grad():
            attention.temporal_bias.weight[:, 0] = -.2
            for f in attention.filters:
                f.output_gate.bias.fill_(.3)
        inputs = np.random.default_rng(42).normal(size=(5, 7, 32)).astype(np.float32)
        ids = [f'trial-{i}' for i in range(5)]
        labels = np.array([0, 1, 2, 3, 0])
        capture = TemporalGateCapture(model)
        probabilities = []
        try:
            with torch.no_grad():
                for start in (0, 3):
                    capture.start_batch(ids[start:start+3])
                    probabilities.append(model(torch.from_numpy(inputs[start:start+3])).softmax(-1).numpy())
        finally:
            for hook in capture.hooks:
                hook.remove()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            writer = FigureWriter(directory)
            summary = temporal_gate_figures(writer, capture, ids, labels, 'validation')
            self.assertEqual(summary['0']['samples'], 5)
            with np.load(Path(directory) / 'temporal_gate/0.npz') as saved:
                self.assertEqual(saved['sample_ids'].tolist(), ids)
                self.assertEqual(saved['graphs'].shape, (5, 4, 7, 7))
                np.testing.assert_allclose(saved['gates'], 2 / (1 + np.exp(-.3)), rtol=1e-6)
            result = validation_ablation_figures(writer, model, inputs, ids, labels,
                                                np.concatenate(probabilities), 'cpu', 'validation')
            self.assertEqual(len(result['ablations']), 3)
            json.loads((Path(directory) / 'temporal_gate/validation_ablation.json').read_text())
            self.assertIsNone(validation_ablation_figures(writer, model, inputs, ids, labels,
                                                         np.concatenate(probabilities), 'cpu', 'test'))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    @unittest.skipUnless(HAS_MATPLOTLIB, "plotting requires the 'plots' extra")
    def test_electrode_output_gate_diagnostics_use_electrode_labels_and_folder(self):
        # The only gate layout a SI_Hom run can produce: electrode nodes, an
        # output gate and no temporal score bias.
        attention = AGFL(options(output_gate=True))
        attention.attention_key, attention.token_axis = 'agfl', 'electrode'
        model = nn.Sequential(attention, nn.Flatten(), nn.Linear(224, 4)).eval()
        model.token_axis = 'electrode'
        with torch.no_grad():
            for f in attention.filters:
                f.output_gate.bias.fill_(.3)
        inputs = np.random.default_rng(7).normal(size=(5, 7, 32)).astype(np.float32)
        ids = [f'trial-{i}' for i in range(5)]
        labels = np.array([0, 1, 2, 3, 0])
        names = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'Cz']
        capture = TemporalGateCapture(model)
        try:
            capture.start_batch(ids)
            with torch.no_grad():
                probabilities = model(torch.from_numpy(inputs)).softmax(-1).numpy()
        finally:
            for hook in capture.hooks:
                hook.remove()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            writer = FigureWriter(directory)
            summary = temporal_gate_figures(writer, capture, ids, labels, 'validation',
                                            metadata={'channel_names': names})['0']
            self.assertEqual((summary['token_axis'], summary['node_labels']), ('electrode', names))
            self.assertFalse(summary['temporal_bias_enabled'])
            self.assertTrue(summary['output_gate_enabled'])
            self.assertEqual(summary['arrays'], 'electrode_gate/0.npz')
            with np.load(Path(directory) / summary['arrays']) as saved:
                self.assertEqual(saved['sample_ids'].tolist(), ids)
                self.assertEqual(saved['node_labels'].tolist(), names)
                self.assertEqual(saved['gates'].shape, (5, 4, 7))
                np.testing.assert_allclose(saved['gates'], 2 / (1 + np.exp(-.3)), rtol=1e-6)
                self.assertFalse(saved['temporal_bias'].any())
            self.assertTrue((Path(directory) / 'electrode_gate.json').is_file())
            self.assertFalse((Path(directory) / 'temporal_gate.json').exists())
            result = validation_ablation_figures(writer, model, inputs, ids, labels,
                                                probabilities, 'cpu', 'validation')
            self.assertEqual([row['removed'] for row in result['ablations']], ['output_gate'])
            self.assertEqual(result['report_file'], 'electrode_gate/validation_ablation.json')
            json.loads((Path(directory) / result['report_file']).read_text())
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
