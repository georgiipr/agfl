"""Optional cluster checks. Do not execute project code in the local workspace."""
import unittest

import torch

from agfl_speech.config import merge, resolve_config, resolve_experiments
from agfl_speech.models import available_models, get_model_spec
from agfl_speech.presets import load_preset, preset_names
from tests.model_fixtures import small_model_options


class InterchannelContractTests(unittest.TestCase):
    def test_every_eeg_model_rejects_temporal_bias_before_loading_data(self):
        for model in available_models():
            for attention in ('agfl', 'mha'):
                with self.subTest(model=model, attention=attention), \
                        self.assertRaisesRegex(ValueError, 'temporal_bias'):
                    resolve_config({'model': model, 'dataset': 'si_hom', 'attention': attention,
                                    'attention_options': {'temporal_bias': True}})

    def test_temporal_eeg_cannot_be_launched_via_config_or_direct_build(self):
        metadata = {'modality': 'eeg', 'channels': 19, 'samples': 500, 'num_classes': 8}
        with self.assertRaisesRegex(ValueError, 'across electrodes'):
            resolve_config({'model': 'eegnet', 'model_options': {'attention_axis': 'time'}})
        with self.assertRaisesRegex(ValueError, 'across electrodes'):
            get_model_spec('eegnet').build({'attention_axis': 'time'}, metadata)
        # The other backbones have no axis selector at all.
        for key in ('eegnet_original', 'signal_transformer'):
            with self.subTest(model=key):
                with self.assertRaisesRegex(ValueError, 'attention_axis'):
                    resolve_config({'model': key, 'model_options': {'attention_axis': 'time'}})
                with self.assertRaisesRegex(ValueError, 'attention_axis'):
                    get_model_spec(key).build({'attention_axis': 'time'}, metadata)

    def test_eegnet_graph_is_19_by_19_regardless_of_time_length(self):
        for samples in (128, 500):
            for attention in ('agfl', 'mha', 'hcann'):
                with self.subTest(samples=samples, attention=attention):
                    options = {**small_model_options('eegnet'), 'electrode_dim': 16}
                    model = get_model_spec('eegnet').build(options,
                        {'modality': 'eeg', 'channels': 19, 'samples': samples, 'num_classes': 8},
                        attention, {'heads': 4}).eval()
                    values = torch.randn(2, 19, samples, requires_grad=True)
                    logits = model(values)
                    layer = model.attn_blocks[0]
                    graph = layer.last_adj if attention == 'agfl' else layer.last_attn
                    self.assertEqual(tuple(graph.shape[-2:]), (19, 19))
                    self.assertEqual((layer.token_axis, layer.num_tokens), ('electrode', 19))
                    self.assertEqual(layer.options['dim'], 16)
                    self.assertEqual(tuple(logits.shape), (2, 8))
                    logits.square().sum().backward()
                    self.assertTrue(torch.isfinite(values.grad).all())

    def test_electrodes_remain_separate_until_attention(self):
        for key in available_models():
            with self.subTest(model=key):
                model = get_model_spec(key).build(small_model_options(key),
                    {'modality': 'eeg', 'channels': 4, 'samples': 65, 'num_classes': 4},
                    'mha', {'heads': 2}).eval()
                layer = next(m for m in model.modules() if getattr(m, 'is_attention', False))
                captured = []
                hook = layer.register_forward_pre_hook(
                    lambda module, args: captured.append(args[0].detach().clone()))
                x = torch.randn(2, 4, 65)
                changed = x.clone()
                changed[:, 2] += torch.randn_like(changed[:, 2]) * 3
                try:
                    with torch.no_grad():
                        model(x)
                        model(changed)
                finally:
                    hook.remove()
                # Tokens are [graphs, electrodes, features] in every backbone.
                self.assertEqual(captured[0].shape[1], 4)
                # Eval mode fixes normalization/dropout so this tests feature routing,
                # not the expected batch-statistics coupling during training.
                torch.testing.assert_close(captured[0][:, [0, 1, 3]], captured[1][:, [0, 1, 3]], rtol=0, atol=0)
                self.assertFalse(torch.equal(captured[0][:, 2], captured[1][:, 2]))

    def test_every_packaged_preset_keeps_attention_across_electrodes(self):
        for name in preset_names():
            with self.subTest(preset=name):
                document = load_preset(name)
                arms = ([merge(document['base'], arm) for arm in document['experiments']]
                        if 'experiments' in document else [document])
                configs = [c for arm in arms for c in resolve_experiments(arm)]
                self.assertTrue(configs)
                for c in configs:
                    self.assertEqual((c['dataset'], c['model_variant']), ('si_hom', 'eeg'))
                    self.assertEqual(c['model_options'].get('attention_axis', 'electrode'), 'electrode')
                    self.assertFalse(c['attention_options'].get('temporal_bias', False))

    def test_only_eeg_variants_exist(self):
        for key in available_models():
            with self.subTest(model=key):
                spec = get_model_spec(key)
                self.assertEqual(set(spec.variants), {'eeg'})
                with self.assertRaisesRegex(ValueError, 'implementation for modality'):
                    spec.build({}, {'modality': 'ecg', 'channels': 2, 'samples': 65, 'num_classes': 3}, 'mha')


if __name__ == '__main__':
    unittest.main()
