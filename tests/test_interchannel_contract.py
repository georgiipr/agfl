"""Optional cluster checks. Do not execute project code in the local workspace."""
import json
from pathlib import Path
import unittest

import torch

from agfl.config import merge, resolve_config, resolve_experiments
from agfl.models import available_models, get_model_spec
from agfl.presets import load_preset, preset_names, retired_presets
from tests.model_fixtures import small_model_options


class InterchannelContractTests(unittest.TestCase):
    def test_every_eeg_model_rejects_temporal_bias_before_loading_data(self):
        for model in available_models():
            with self.subTest(model=model), self.assertRaisesRegex(ValueError, 'temporal_bias'):
                resolve_config({'model': model, 'dataset': 'eeg', 'attention': 'mha',
                                'attention_options': {'temporal_bias': True}})

    def test_temporal_eeg_cannot_be_launched_via_config_or_direct_build(self):
        for key in ('eegnet', 'eegencoder', 'dstseegencoder'):
            with self.subTest(model=key):
                with self.assertRaisesRegex(ValueError, 'across electrodes'):
                    resolve_config({'model': key, 'model_options': {'attention_axis': 'time'}})
                with self.assertRaisesRegex(ValueError, 'across electrodes'):
                    get_model_spec(key).build({'attention_axis': 'time'},
                        {'modality': 'eeg', 'channels': 4, 'samples': 1000, 'num_classes': 4})

    def test_new_eegnet_graph_is_22_by_22_regardless_of_time_length(self):
        for samples in (128, 256):
            for attention in ('agfl', 'mha'):
                with self.subTest(samples=samples, attention=attention):
                    options = {**small_model_options('eegnet', 'eeg'), 'electrode_dim': 16}
                    model = get_model_spec('eegnet').build(options,
                        {'modality': 'eeg', 'channels': 22, 'samples': samples, 'num_classes': 4},
                        attention, {'heads': 4}).eval()
                    values = torch.randn(2, 22, samples, requires_grad=True)
                    logits = model(values)
                    layer = model.attn_blocks[0]
                    graph = layer.last_adj if attention == 'agfl' else layer.last_attn
                    self.assertEqual(graph.shape[-2:], (22, 22))
                    self.assertEqual(layer.num_tokens, 22)
                    self.assertEqual(layer.qkv.in_features if attention == 'mha' else layer.options['dim'], 16)
                    self.assertEqual(logits.shape, (2, 4))
                    logits.square().sum().backward()
                    self.assertTrue(torch.isfinite(values.grad).all())

    def test_electrodes_remain_separate_until_attention(self):
        model = get_model_spec('eegnet').build(small_model_options('eegnet', 'eeg'),
            {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 4}, 'mha').eval()
        captured = []
        hook = model.attn_blocks[0].register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().clone()))
        x = torch.randn(2, 4, 65)
        changed = x.clone()
        changed[:, 2] += torch.randn_like(changed[:, 2]) * 3
        try:
            with torch.no_grad():
                model(x)
                model(changed)
        finally:
            hook.remove()
        # Eval mode fixes normalization/dropout so this tests feature routing,
        # not the expected batch-statistics coupling during training.
        torch.testing.assert_close(captured[0][:, [0, 1, 3]], captured[1][:, [0, 1, 3]], rtol=0, atol=0)
        self.assertFalse(torch.equal(captured[0][:, 2], captured[1][:, 2]))

    def test_old_checkpoint_construction_keeps_original_axes_and_width(self):
        spec = get_model_spec('eegnet')
        metadata = {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 4}
        for axis in ('electrode', 'time'):
            options = {**small_model_options('eegnet', 'eeg'), 'attention_axis': axis}
            model = spec.rebuild_saved(options, metadata, 'mha').eval()
            self.assertFalse(model.compact_electrodes)
            self.assertFalse(any('electrode_projection' in k for k in model.state_dict()))
            self.assertEqual(model.token_axis, axis)
            width = options['f2'] * (65 // options['pk1'] // options['pk2']) if axis == 'electrode' else options['f2']
            self.assertEqual(model.attn_blocks[0].qkv.in_features, width)
            restored = spec.rebuild_saved(options, metadata, 'mha').eval()
            restored.load_state_dict(model.state_dict(), strict=True)
            values = torch.randn(2, 4, 65)
            with torch.no_grad():
                torch.testing.assert_close(restored(values), model(values), rtol=0, atol=0)

    def test_preset_catalog_excludes_archived_and_stale_temporal_names(self):
        self.assertFalse(set(preset_names()) & retired_presets())
        for name in retired_presets():
            with self.assertRaisesRegex(ValueError, 'retired temporal EEG'):
                load_preset(name)
            original = json.loads((Path(__file__).parents[1] / 'agfl/presets/archive/eeg_temporal' / (name + '.json')).read_text())
            base = original.get('base', original)
            self.assertEqual(base['model_options']['attention_axis'], 'time')

    def test_interchannel_presets_have_correct_complete_matrices(self):
        for name, count in [('eegnet-interchannel-comparison', 15), ('eegnet-interchannel-attentions', 315),
                            ('eegnet-a03-a04-interchannel', 30)]:
            document = load_preset(name)
            configs = [c for arm in document['experiments'] for c in resolve_experiments(merge(document['base'], arm))]
            self.assertEqual(sum(len(c['seeds']) for c in configs), count)
            self.assertTrue(all(c['model_options']['attention_axis'] == 'electrode' for c in configs))
            self.assertTrue(all(not c['attention_options'].get('temporal_bias', False) for c in configs))
            self.assertEqual(len({json.dumps(c['model_options'], sort_keys=True) for c in configs}), 1)

    def test_ecg_keeps_time_nodes(self):
        for model_key in available_models():
            model = get_model_spec(model_key).build(small_model_options(model_key, 'ecg'),
                {'modality': 'ecg', 'channels': 2, 'samples': 65, 'num_classes': 3}, 'mha', {'heads': 2})
            self.assertIn(model.token_axis, {'time', 'time_patch'})
            if model_key == 'eegnet':
                self.assertFalse(model.compact_electrodes)
                self.assertNotIn('electrode_dim', get_model_spec(model_key).defaults_for('ecg'))


if __name__ == '__main__':
    unittest.main()
