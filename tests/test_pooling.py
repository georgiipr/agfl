"""Regression checks for deterministic EEG pooling; run on the target machine."""
import json

import pytest
import torch

from agfl.models._shared.pooling import DeterministicAdaptiveAvgPool1d


@pytest.mark.parametrize('length,bins', [(1000, 8), (65, 8), (7, 1), (4, 9), (8, 8)])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_pooling_matches_native_cpu_values_and_input_gradients(length, bins, noncontiguous):
    # Native CPU pooling is the independent reference for both bin boundaries
    # and accumulation of gradients where adaptive windows overlap.
    generator = torch.Generator().manual_seed(19)
    values = torch.randn(2, 3, length * 2 if noncontiguous else length,
                         dtype=torch.float64, generator=generator)
    if noncontiguous:
        values = values[..., ::2]
    reference_input = values.detach().requires_grad_()
    actual_input = values.detach().requires_grad_()
    expected = torch.nn.functional.adaptive_avg_pool1d(reference_input, bins)
    actual = DeterministicAdaptiveAvgPool1d(bins)(actual_input)
    upstream = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
    expected.backward(upstream)
    actual.backward(upstream)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-12, atol=1e-12)


@pytest.fixture
def strict_determinism():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=False)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required on target machine')
@pytest.mark.parametrize('length,bins', [(1000, 8), (65, 8), (4, 9)])
def test_cuda_pooling_backward_is_deterministic_and_matches_cpu(length, bins, strict_determinism):
    generator = torch.Generator().manual_seed(31)
    values = torch.randn(2, 3, length, generator=generator, dtype=torch.float64)
    upstream = torch.randn(2, 3, bins, generator=generator, dtype=torch.float64)
    reference_input = values.clone().requires_grad_()
    expected = torch.nn.functional.adaptive_avg_pool1d(reference_input, bins)
    expected.backward(upstream)
    runs = []
    for _ in range(2):
        x = values.cuda().requires_grad_()
        actual = DeterministicAdaptiveAvgPool1d(bins)(x)
        actual.backward(upstream.cuda())
        runs.append((actual.detach().cpu(), x.grad.cpu()))
    for actual, gradient in runs:
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(gradient, reference_input.grad, rtol=1e-12, atol=1e-12)
    for first, second in zip(runs[0], runs[1]):
        torch.testing.assert_close(first, second, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required on target machine')
@pytest.mark.parametrize('amp', [False, True])
def test_cuda_eeg_training_completes_with_strict_determinism(tmp_path, monkeypatch, strict_determinism, amp):
    from agfl.engine import run_experiment

    # Restore the global backend policy after the engine configures it.
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    monkeypatch.setattr(torch.backends.cudnn, 'benchmark', False)
    monkeypatch.setattr(torch.backends.cudnn, 'deterministic', True)
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    previous_threads = torch.get_num_threads()
    try:
        results = run_experiment({
            'model': 'conformer', 'attention': 'agfl', 'dataset': 'synthetic_eeg', 'device': 'cuda',
            'seeds': [0], 'deterministic': True,
            'data': {'subjects': 3, 'samples_per_subject': 4, 'channels': 22, 'samples': 1000},
            'training': {'epochs': 2, 'batch_size': 4, 'amp': amp},
            'output_dir': str(tmp_path / 'results'), 'split_dir': str(tmp_path / 'splits'),
        })
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert results[0]['status'] == 'completed'
        assert results[0]['test']['n_samples'] == 4
        saved = next((tmp_path / 'results').rglob('result.json'))
        history = json.loads(saved.with_name('history.json').read_text())
        assert len(history) == 2
        assert saved.with_name('checkpoint.pt').is_file()
        assert saved.with_name('predictions.npz').is_file()
    finally:
        torch.set_num_threads(previous_threads)
