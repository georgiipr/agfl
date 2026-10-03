"""Target-machine checks for the shared non-AMP gradient safety check."""
import pytest
import torch

from agfl_speech.engine import gradients_are_finite


DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA check for the experiment machine'))]


@pytest.mark.parametrize('device', DEVICES)
def test_finite_check_includes_later_parameters_and_ignores_missing_gradients(device):
    parameters = [torch.nn.Parameter(torch.ones(3, device=device)) for _ in range(4)]
    parameters[0].grad = torch.zeros_like(parameters[0])
    parameters[2].grad = torch.tensor([1e20, -1e20, 0.], device=device)
    parameters[3].requires_grad_(False)
    assert gradients_are_finite(iter(parameters)) is True
    for invalid in (float('nan'), float('inf'), -float('inf')):
        parameters[2].grad[1] = invalid
        assert gradients_are_finite(iter(parameters)) is False
    parameters[2].grad[1] = -1e20
    assert gradients_are_finite(iter(parameters)) is True


def test_empty_or_unused_parameters_are_finite():
    assert gradients_are_finite(iter(())) is True
    assert gradients_are_finite(iter([torch.nn.Parameter(torch.ones(2))])) is True
