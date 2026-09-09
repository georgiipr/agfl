"""Frozen-weight parity for preserved native layouts, on the target machine."""
import os
import pytest
import torch
from agfl.models import get_model_spec
from agfl.datasets import load_dataset
from tests.references.models.eegnet import EEGNet
from tests.references.models.conformer import ConformerModel
from tests.references.agfl import AGFL as OriginalAGFL

CASES = [('eegnet', 'eeg'), ('conformer', 'ecg')]


def assert_frozen_weight_parity(key, modality, x):
    original = (EEGNet(attention_type='agfl') if modality == 'eeg' else ConformerModel(mode='agfl')).double().eval()
    # Nonzero taps exercise the graph rather than only output biases.
    with torch.no_grad():
        for layer in original.modules():
            if isinstance(layer, OriginalAGFL):
                for graph_filter in layer.filters:
                    graph_filter.alpha_logits.copy_(torch.linspace(-.2, .7, len(graph_filter.alpha_logits)))
    spec = get_model_spec(key)
    options = {'attention_axis': 'time', 'attention_residual': False} if modality == 'eeg' else {}
    model = spec.build(options, {'modality': modality, 'channels': x.shape[1], 'samples': x.shape[2],
                                 'num_classes': 4 if modality == 'eeg' else 2}, 'agfl').double().eval()
    model.load_state_dict(original.state_dict(), strict=True)
    first, second = x.double().requires_grad_(), x.double().requires_grad_()
    expected = original(first if modality == 'eeg' else first[:, 0])
    actual = model(second)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize('key,modality', CASES)
def test_native_layout_with_frozen_original_weights(key, modality):
    shape = (2, 22, 1000) if modality == 'eeg' else (2, 1, 256)
    assert_frozen_weight_parity(key, modality, torch.randn(*shape))


@pytest.mark.real_data
@pytest.mark.skipif(os.environ.get('AGFL_REAL_DATA') != '1', reason='Enable real recordings on the target machine')
@pytest.mark.parametrize('key,modality', CASES)
def test_real_recording_frozen_weight_parity(key, modality):
    if modality == 'eeg':
        data = {'data_dir': os.environ.get('AGFL_EEG_DIR', '../ml'), 'subjects': [1]}
    else:
        data = {'data_dir': os.environ.get('AGFL_ECG_DIR', '../mit-bih-arrhythmia-database-1.0.0'), 'records': ['100']}
    bundle = load_dataset(modality, data)
    assert_frozen_weight_parity(key, modality, torch.from_numpy(bundle.x[:2]))
