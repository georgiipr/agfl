"""Small reusable layers; model topology remains in each model package."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def check_input(x, metadata):
    expected = (int(metadata['channels']), int(metadata['samples']))
    if x.ndim != 3 or tuple(x.shape[1:]) != expected:
        raise ValueError(f'Expected [B,C,T] with C,T={expected}, got {tuple(x.shape)}')


def positive_options(options, *names):
    for name in names:
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f'model_options.{name} must be a positive integer')


class PositionalEncoding(nn.Module):
    def __init__(self, dim, tokens):
        super().__init__()
        position = torch.arange(tokens, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.) / dim))
        encoding = torch.zeros(tokens, dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency[:dim // 2])
        self.register_buffer('pe', encoding[None])

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.padding, 0)))


class ElectrodeReadout(nn.Module):
    """Learn a signed spatial filter per feature, retaining electrode identity."""
    def __init__(self, channels, features, mode='learned'):
        super().__init__()
        if mode not in {'learned', 'mean'}:
            raise ValueError('spatial_readout must be learned or mean')
        self.projection = (nn.Conv1d(features, features, channels, groups=features, bias=False)
                           if mode == 'learned' else None)

    def forward(self, tokens):
        if self.projection is None:
            return tokens.mean(dim=1)
        return self.projection(tokens.transpose(1, 2)).squeeze(-1)
