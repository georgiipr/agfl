"""Optional cluster-only regression checks; standard unittest, no new library.

Not a prerequisite for experiment launches. Do not execute in the local
workspace: the project permits only static local verification.
"""
from copy import deepcopy
import unittest

import torch
from torch import nn

from agfl_speech.config import comparison_identity, resolve_experiments
from agfl_speech.models import get_model_spec
from agfl_speech.presets import load_preset
from agfl_speech.visualization.maps import mixer_map


class EEGNetSpatialFusionTests(unittest.TestCase):
    metadata = {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 3}
    options = {'f1': 4, 'd': 2, 'f2': 8, 'pk1': 2, 'pk2': 2,
               'temp_kernel': 7, 'electrode_dim': 8, 'dropout_rate': 0.0}

    def build(self, attention='agfl', **changes):
        return get_model_spec('eegnet').build({**self.options, **changes},
            self.metadata, attention, {'heads': 2})

    def test_both_paths_and_zero_initialized_agfl_receive_gradients(self):
        torch.manual_seed(42)
        model = self.build()
        self.assertTrue(model.spatial_fusion)
        self.assertEqual(model.spatial_block2[0].kernel_size, (4, 1))
        self.assertEqual(model.spatial_block2[0].groups, 4)
        # 16 ordered time bins of eight spatial features, plus attention.
        self.assertEqual(model.classifier.in_features, 16 * 8 + 8)
        values = torch.randn(3, 4, 65, requires_grad=True)
        loss = nn.functional.cross_entropy(model(values), torch.tensor([0, 1, 2]))
        loss.backward()
        for layer in (model.block1[0], model.block2[0], model.electrode_projection,
                      model.spatial_readout.projection, model.spatial_block2[0],
                      model.spatial_block3[0], model.fc):
            with self.subTest(layer=type(layer).__name__):
                self.assertIsNotNone(layer.weight.grad)
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        taps = torch.stack([f.alpha_logits.grad for f in model.attn_blocks[0].filters])
        self.assertTrue(torch.isfinite(taps).all())
        self.assertGreater(taps.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(values.grad).all())
        # The common stem is evaluated once, not independently for both paths.
        self.assertEqual(model.block1[1].num_batches_tracked.item(), 1)
        self.assertEqual(model.spatial_block2[1].num_batches_tracked.item(), 1)

    def test_attention_diagnostics_keep_one_graph_per_trial_for_all_mechanisms(self):
        metadata = {**self.metadata, 'channels': 19}
        for attention in ('agfl', 'mha', 'hcann'):
            with self.subTest(attention=attention):
                torch.manual_seed(42)
                model = get_model_spec('eegnet').build(self.options, metadata,
                    attention, {'heads': 2}).eval()
                inputs = []
                hook = model.attn_blocks[0].register_forward_pre_hook(
                    lambda module, args: inputs.append(args[0].detach().clone()))
                try:
                    with torch.no_grad():
                        logits = model(torch.randn(3, 19, 65))
                finally:
                    hook.remove()
                graph, _ = mixer_map(model.attn_blocks[0], inputs[0])
                self.assertEqual(logits.shape, (3, 3))
                self.assertEqual(graph.shape, (3, 2, 19, 19))
                self.assertTrue(torch.isfinite(logits).all())
                self.assertTrue(torch.isfinite(graph).all())

    def test_attention_choice_does_not_change_common_initial_weights_or_rng(self):
        torch.manual_seed(42)
        agfl = self.build('agfl')
        rng = torch.get_rng_state().clone()
        common = lambda model: {k: v for k, v in model.state_dict().items()
                                if not k.startswith('attn_blocks.')}
        left = common(agfl)
        for attention in ('mha', 'hcann'):
            with self.subTest(attention=attention):
                torch.manual_seed(42)
                other = self.build(attention)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                right = common(other)
                self.assertEqual(left.keys(), right.keys())
                for key in left:
                    torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                for model in (agfl, other):
                    self.assertTrue(all(layer.num_batches_tracked.item() == 0
                        for layer in model.modules() if isinstance(layer, nn.BatchNorm2d)))

    def test_attention_contributes_without_compressing_spatial_features(self):
        torch.manual_seed(42)
        model = self.build('mha').eval()
        features = []
        readout_hook = model.fc.register_forward_pre_hook(
            lambda module, args: features.append(args[0].detach().clone()))
        values = torch.randn(3, 4, 65)
        try:
            with torch.no_grad():
                original = model(values)
            # A test-only perturbation checks that the selected attention
            # actually contributes to predictions; no new run mode is added.
            perturb = model.attn_blocks[0].register_forward_hook(
                lambda module, args, output: output + 1)
            try:
                with torch.no_grad():
                    changed = model(values)
            finally:
                perturb.remove()
        finally:
            readout_hook.remove()
        spatial_width = 8 * 16
        torch.testing.assert_close(features[0][:, :spatial_width],
                                   features[1][:, :spatial_width], rtol=0, atol=0)
        self.assertFalse(torch.equal(features[0][:, spatial_width:], features[1][:, spatial_width:]))
        self.assertFalse(torch.equal(original, changed))

    def test_max_norm_covers_restored_spatial_filters_and_full_classifier(self):
        model = self.build()
        layers = [(model.spatial_block2[0], model.max_norm1),
                  (model.block2[0], model.max_norm1),
                  (model.spatial_readout.projection, model.max_norm1),
                  (model.fc, model.max_norm2)]
        with torch.no_grad():
            for layer, _ in layers:
                layer.weight.fill_(100)
        model.clip_weights()
        for layer, maximum in layers:
            norms = layer.weight.detach().flatten(1).norm(dim=1)
            self.assertLessEqual(norms.max().item(), maximum + 1e-6)

    def test_compact_checkpoint_retains_shapes_and_equation(self):
        spec = get_model_spec('eegnet')
        torch.manual_seed(42)
        original = self.build('mha', electrode_architecture='compact').eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        # A saved configuration names its layout; nothing is substituted on replay.
        restored = spec.rebuild_saved({**self.options, 'electrode_architecture': 'compact'},
                                      self.metadata, 'mha', {'heads': 2}).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertFalse(restored.spatial_fusion)
        self.assertEqual(original.state_dict().keys(), restored.state_dict().keys())
        self.assertEqual(restored.fc.in_features, 8)
        restored.load_state_dict(original.state_dict(), strict=True)
        values = torch.randn(3, 4, 65)
        with torch.no_grad():
            local = original.convolve(values.reshape(12, 1, 1, 65)).reshape(3, 4, -1)
            local = original.electrode_projection(local)
            message = original.attn_blocks[0](original.electrode_position(local))
            expected = original.fc(original.spatial_readout(local + message))
            torch.testing.assert_close(restored(values), expected, rtol=0, atol=0)

    def test_fusion_checkpoint_replays_with_explicit_architecture(self):
        original = self.build().eval()
        options = {**self.options, 'electrode_architecture': 'spatial_fusion'}
        restored = get_model_spec('eegnet').rebuild_saved(
            options, self.metadata, 'agfl', {'heads': 2}).eval()
        restored.load_state_dict(original.state_dict(), strict=True)
        values = torch.randn(3, 4, 65)
        with torch.no_grad():
            torch.testing.assert_close(original(values), restored(values), rtol=0, atol=0)

    def test_default_preset_uses_spatial_fusion_and_the_layout_versions_identity(self):
        configs = resolve_experiments(load_preset('si-hom'))
        self.assertEqual(len(configs), 1)
        config = configs[0]
        self.assertEqual((config['model'], config['attention'], config['subject_id']), ('eegnet', 'agfl', None))
        self.assertEqual(config['model_options']['electrode_architecture'], 'spatial_fusion')
        self.assertEqual(config['model_options']['electrode_dim'], 32)
        self.assertFalse(config['attention_options'].get('temporal_bias', False))
        self.assertFalse(config['attention_options'].get('output_gate', False))
        for layout in ('compact', 'pre_spatial'):
            other = deepcopy(config)
            other['model_options']['electrode_architecture'] = layout
            self.assertNotEqual(comparison_identity(other), comparison_identity(config))

    def test_bad_selector_is_rejected_and_only_the_eeg_variant_exists(self):
        with self.assertRaisesRegex(ValueError, 'electrode_architecture'):
            self.build(electrode_architecture='misspelled')
        with self.assertRaisesRegex(ValueError, 'No eegnet implementation'):
            get_model_spec('eegnet').defaults_for('ecg')


if __name__ == '__main__':
    unittest.main()
