"""Reproduce saved checkpoints from the retired EEGNet power experiment.

The active EEGNet backbone is unchanged. These classes are selected only when
an archived configuration explicitly requests ``temporal_statistics=mean_logvar``.
Module names, initialization order and equations retain that experiment's
checkpoint format so its unfavorable results remain inspectable.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .eeg import EEGModel
from .ecg import ECGModel


class _ZeroLinear(nn.Linear):
    """An initially inactive branch must not consume the backbone's RNG."""
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class TemporalStatisticsFusion(nn.Module):
    """Historical power-study feature fusion, retained for reproduction only."""
    def __init__(self, features, out_features, bins, samples):
        super().__init__()
        if (any(type(value) is not int or value < 1
                for value in (features, out_features, bins, samples))
                or features < 2 or samples // bins < 2):
            raise ValueError('Temporal log-variance needs at least two features and two samples per bin')
        self.features, self.bins, self.samples = features, bins, samples
        self.bin_edges = tuple(i * samples // bins for i in range(bins + 1))
        self.projection = _ZeroLinear(features, out_features, bias=False)

    def log_variance(self, values):
        """Population log variance, [B,features,height,samples] -> [...,bins]."""
        if (values.ndim != 4 or values.shape[1] != self.features
                or values.shape[-1] != self.samples or values.shape[2] < 1):
            raise ValueError('Temporal statistics input has an incompatible shape')
        with torch.autocast(device_type=values.device.type, enabled=False):
            working = values if values.dtype == torch.float64 else values.float()
            variances = []
            for start, end in zip(self.bin_edges[:-1], self.bin_edges[1:]):
                window = working[..., start:end]
                centered = window - window.mean(dim=-1, keepdim=True)
                variances.append(centered.square().mean(dim=-1))
            return (torch.stack(variances, dim=-1) + 1e-6).log()

    def normalized_statistics(self, values):
        """Normalize feature contrasts within each trial/time bin, without BN."""
        with torch.autocast(device_type=values.device.type, enabled=False):
            log_variance = self.log_variance(values).movedim(1, -1)
            normalized = F.layer_norm(log_variance, (self.features,), eps=1e-5)
            return normalized.movedim(-1, 1)

    def forward(self, values):
        with torch.autocast(device_type=values.device.type, enabled=False):
            statistics = self.normalized_statistics(values).movedim(1, -1)
            projected = F.linear(statistics, self.projection.weight.to(statistics.dtype))
            return projected.movedim(-1, 1)


class _PowerStudyReproduction:
    def __init__(self, options, metadata, attention):
        super().__init__(options, metadata, attention)
        # The former backbone created this after FC/Xavier initialization and
        # max-norm clipping. Preserve that order and the state-dictionary key.
        self.temporal_statistics = TemporalStatisticsFusion(
            options['f1'] * options['d'], options['f2'], self.steps, metadata['samples'])

    def convolve(self, x):
        spatial = self.block2[1](self.block2[0](self.block1(x)))
        encoded = spatial
        for layer in list(self.block2.children())[2:]:
            encoded = layer(encoded)
        encoded = self.block3(encoded)
        power = self.temporal_statistics(spatial)
        return encoded + power.to(encoded.dtype)


class EEGPowerReproduction(_PowerStudyReproduction, EEGModel):
    """Archived power-study EEG variant with its original checkpoint keys."""


class ECGPowerReproduction(_PowerStudyReproduction, ECGModel):
    """Archived power-study ECG variant with its original checkpoint keys."""
