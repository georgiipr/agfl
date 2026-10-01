"""Optional cluster-only checks for EEGNet ``pre_spatial``; do not run locally.

Not a prerequisite for experiment launches. The local workspace permits only
static verification of project code.
"""
from copy import deepcopy
import unittest

import numpy as np
import torch
from torch import nn

from agfl.config import comparison_identity, merge, resolve_config, resolve_experiments
from agfl.datasets import get_dataset_spec
from agfl.models import get_model_spec
from agfl.presets import load_preset
from agfl.visualization.maps import mixer_map, per_trial_maps


class EEGNetPreSpatialTests(unittest.TestCase):
    metadata = {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 3}
    options = {'f1': 4, 'd': 2, 'f2': 8, 'pk1': 2, 'pk2': 2, 'temp_kernel': 7,
               'dropout_rate': 0.0, 'electrode_architecture': 'pre_spatial'}

    def build(self, attention='agfl', **changes):
        return get_model_spec('eegnet').build({**self.options, **changes},
            self.metadata, attention, {'heads': 2})

    def test_attention_mixes_electrodes_at_every_time_step_before_the_spatial_filter(self):
        torch.manual_seed(42)
        model = self.build().eval()
        self.assertTrue(model.pre_spatial)
        self.assertFalse(model.spatial_fusion or model.compact_electrodes)
        self.assertIsNone(model.spatial_readout)
        self.assertFalse(hasattr(model, 'electrode_projection'))
        self.assertFalse(hasattr(model, 'spatial_block2'))
        # EEGNet's original full-channel depthwise spatial filter and classifier.
        self.assertEqual(model.block2[0].kernel_size, (4, 1))
        self.assertEqual(model.block2[0].groups, 4)
        self.assertEqual(model.classifier.in_features, 8 * 16)
        self.assertEqual((model.token_axis, model.num_tokens), ('electrode', 4))
        layer = model.attn_blocks[0]
        self.assertEqual((layer.token_axis, layer.num_tokens), ('electrode', 4))
        shapes = []
        hook = layer.register_forward_hook(
            lambda module, args, output: shapes.append((tuple(args[0].shape), tuple(output.shape))))
        try:
            with torch.no_grad():
                logits = model(torch.randn(3, 4, 65))
        finally:
            hook.remove()
        # One electrode set per time step, f1 features per node, trial-major rows.
        self.assertEqual(shapes, [((3 * 65, 4, 4), (3 * 65, 4, 4))])
        self.assertEqual(tuple(layer.last_adj.shape), (2, 3 * 65, 4, 4))
        self.assertEqual(logits.shape, (3, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_token_rows_are_trial_major_and_electrodes_stay_separate_until_attention(self):
        torch.manual_seed(0)
        model = self.build('mha').eval()
        captured = []
        hook = model.attn_blocks[0].register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        x = torch.randn(2, 4, 65)
        changed = x.clone()
        changed[1, 2] += 3.
        try:
            with torch.no_grad():
                model(x)
                model(changed)
        finally:
            hook.remove()
        before, after = (c.reshape(2, 65, 4, -1) for c in captured)
        torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
        torch.testing.assert_close(before[1][:, [0, 1, 3]], after[1][:, [0, 1, 3]], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[1][:, 2], after[1][:, 2]))

    def test_every_stage_receives_gradients_including_zero_initialized_agfl_taps(self):
        torch.manual_seed(42)
        model = self.build()
        values = torch.randn(3, 4, 65, requires_grad=True)
        loss = nn.functional.cross_entropy(model(values), torch.tensor([0, 1, 2]))
        loss.backward()
        for layer in (model.block1[0], model.block2[0], model.block3[0], model.block3[1], model.fc):
            with self.subTest(layer=type(layer).__name__):
                self.assertIsNotNone(layer.weight.grad)
                self.assertTrue(torch.isfinite(layer.weight.grad).all())
                self.assertGreater(layer.weight.grad.abs().sum().item(), 0)
        taps = torch.stack([f.alpha_logits.grad for f in model.attn_blocks[0].filters])
        self.assertTrue(torch.isfinite(taps).all())
        self.assertGreater(taps.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(values.grad).all())
        torch.manual_seed(42)
        mha = self.build('mha')
        nn.functional.cross_entropy(mha(torch.randn(3, 4, 65)), torch.tensor([0, 1, 2])).backward()
        self.assertGreater(mha.electrode_position.embedding.grad.abs().sum().item(), 0)
        self.assertGreater(mha.attn_blocks[0].qkv.weight.grad.abs().sum().item(), 0)

    def test_attention_choice_does_not_change_common_initial_weights_or_rng(self):
        torch.manual_seed(42)
        agfl = self.build('agfl')
        rng = torch.get_rng_state().clone()
        torch.manual_seed(42)
        mha = self.build('mha')
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        common = lambda model: {k: v for k, v in model.state_dict().items()
                                if not k.startswith('attn_blocks.')}
        left, right = common(agfl), common(mha)
        self.assertEqual(left.keys(), right.keys())
        self.assertTrue(any(k.startswith('block2.') for k in left))
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)

    def test_max_norm_covers_the_spatial_filter_and_classifier(self):
        model = self.build()
        layers = [(model.block2[0], model.max_norm1), (model.fc, model.max_norm2)]
        with torch.no_grad():
            for layer, _ in layers:
                layer.weight.fill_(100)
        model.clip_weights()
        for layer, maximum in layers:
            norms = layer.weight.detach().flatten(1).norm(dim=1)
            self.assertLessEqual(norms.max().item(), maximum + 1e-6)

    def test_diagnostic_maps_average_time_steps_to_one_graph_per_trial(self):
        metadata = {**self.metadata, 'channels': 22}
        for attention in ('agfl', 'mha', 'performer', 'linformer', 'nystromformer'):
            with self.subTest(attention=attention):
                torch.manual_seed(42)
                model = get_model_spec('eegnet').build(self.options, metadata,
                    attention, {'heads': 2}).eval()
                inputs = []
                hook = model.attn_blocks[0].register_forward_pre_hook(
                    lambda module, args: inputs.append(args[0].detach().clone()))
                try:
                    with torch.no_grad():
                        logits = model(torch.randn(3, 22, 65))
                finally:
                    hook.remove()
                graph, _ = mixer_map(model.attn_blocks[0], inputs[0])
                self.assertEqual(tuple(graph.shape), (3 * 65, 2, 22, 22))
                reduced = per_trial_maps(graph.detach().cpu().numpy(), 3)
                self.assertEqual(reduced.shape, (3, 2, 22, 22))
                self.assertTrue(np.isfinite(reduced).all())
                self.assertEqual(logits.shape, (3, 3))
        maps = np.arange(6 * 1 * 2 * 2, dtype=float).reshape(6, 1, 2, 2)
        np.testing.assert_allclose(per_trial_maps(maps, 2), maps.reshape(2, 3, 1, 2, 2).mean(1))
        self.assertIs(per_trial_maps(maps, 6), maps)
        self.assertIs(per_trial_maps(maps, 4), maps)

    def test_heads_must_divide_the_temporal_filter_count(self):
        with self.assertRaisesRegex(ValueError, 'divisible'):
            resolve_config({'dataset': 'eeg', 'model': 'eegnet', 'attention_options': {'heads': 4},
                            'model_options': {'electrode_architecture': 'pre_spatial', 'f1': 6}})
        with self.assertRaisesRegex(ValueError, 'divisible'):
            get_model_spec('eegnet').build({**self.options, 'f1': 6}, self.metadata, 'mha', {'heads': 4})

    def test_checkpoint_replays_with_explicit_architecture_and_old_saves_stay_compact(self):
        torch.manual_seed(1)
        original = self.build().eval()
        restored = get_model_spec('eegnet').rebuild_saved(
            self.options, self.metadata, 'agfl', {'heads': 2}).eval()
        self.assertTrue(restored.pre_spatial)
        restored.load_state_dict(original.state_dict(), strict=True)
        values = torch.randn(3, 4, 65)
        with torch.no_grad():
            torch.testing.assert_close(original(values), restored(values), rtol=0, atol=0)
        without = {k: v for k, v in self.options.items() if k != 'electrode_architecture'}
        old = get_model_spec('eegnet').rebuild_saved({**without, 'electrode_dim': 8},
                                                     self.metadata, 'agfl', {'heads': 2})
        self.assertFalse(old.pre_spatial)
        self.assertTrue(old.compact_electrodes)

    def test_recipe_defaults_no_longer_cap_every_subject(self):
        data = get_dataset_spec('eeg').defaults
        self.assertEqual((data['filter_scope'], data['artifact_policy']), ('run', 'include'))
        eeg = resolve_config({'dataset': 'eeg', 'model': 'eegnet'})
        training = eeg['training']
        self.assertEqual((training['epochs'], training['learning_rate'], training['loss'],
                          training['checkpoint_criterion']), (500, 1e-3, 'cross_entropy', 'loss'))
        self.assertEqual(eeg['model_options']['temp_kernel'], 125)
        # The new layout is an explicit selection; the default layout is unchanged.
        self.assertEqual(eeg['model_options']['electrode_architecture'], 'spatial_fusion')
        ecg = resolve_config({'dataset': 'ecg', 'model': 'eegnet'})
        self.assertEqual(ecg['model_options']['temp_kernel'], 32)
        self.assertEqual((ecg['training']['epochs'], ecg['training']['loss']), (50, 'focal'))
        self.assertNotIn('electrode_architecture', get_model_spec('eegnet').defaults_for('ecg'))
        with self.assertRaisesRegex(ValueError, 'continuous'):
            resolve_experiments({'dataset': 'eeg', 'data': {'filter_scope': 'continuous'}})
        self.assertEqual(resolve_experiments({'dataset': 'eeg', 'data': {'subjects': [3]}})[0]['data']['filter_scope'], 'run')

    def test_preset_is_a_matched_thirty_fit_comparison(self):
        document = load_preset('eegnet-pre-spatial-a03-a04-a09')
        configs = [c for arm in document['experiments']
                   for c in resolve_experiments(merge(document['base'], arm))]
        self.assertEqual(len(configs), 6)
        self.assertEqual(sum(len(c['seeds']) for c in configs), 30)
        self.assertEqual({c['attention'] for c in configs}, {'agfl', 'mha'})
        self.assertEqual({c['subject_id'] for c in configs}, {'A03', 'A04', 'A09'})
        by_subject = {}
        for c in configs:
            by_subject.setdefault(c['subject_id'], set()).add(comparison_identity(c))
            model_options, training, data = c['model_options'], c['training'], c['data']
            self.assertEqual(model_options['electrode_architecture'], 'pre_spatial')
            self.assertEqual(model_options['temp_kernel'], 125)
            self.assertEqual((data['filter_scope'], data['artifact_policy']), ('run', 'include'))
            self.assertEqual((training['epochs'], training['learning_rate'], training['loss'],
                              training['checkpoint_criterion'], training['save_checkpoints']),
                             (500, 1e-3, 'cross_entropy', 'loss', True))
            self.assertEqual(c['attention_options']['heads'], 4)
            self.assertFalse(c['attention_options'].get('temporal_bias', False))
            self.assertFalse(c['attention_options'].get('output_gate', False))
            self.assertEqual(c['study']['comparisons'], [['agfl_base', 'mha_base']])
        # Within each subject the two arms share one comparison identity.
        self.assertTrue(all(len(identities) == 1 for identities in by_subject.values()))
        old = deepcopy(configs[0])
        old['model_options']['electrode_architecture'] = 'spatial_fusion'
        self.assertNotEqual(comparison_identity(old), comparison_identity(configs[0]))


if __name__ == '__main__':
    unittest.main()
