"""Conformer half-step FFNs, attention and gated depthwise convolution."""
import torch
from torch import nn
from torch.nn import functional as F
from .._shared.layers import positive_options


class FeedForward(nn.Module):
    def __init__(self, dim, expansion, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * expansion), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * expansion, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConvModule(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.batchnorm = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, 1)

    def forward(self, x):
        x = F.glu(self.pointwise1(x.transpose(1, 2)), dim=1)
        return self.pointwise2(F.silu(self.batchnorm(self.depthwise(x)))).transpose(1, 2)


class ConformerBlock(nn.Module):
    def __init__(self, options, attention, tokens, index, axis):
        super().__init__()
        dim = options['dim']
        self.ff1 = FeedForward(dim, options['expansion'], options['dropout'])
        self.ff2 = FeedForward(dim, options['expansion'], options['dropout'])
        self.attn = attention(dim, tokens, index, options['depth'], axis)
        # Physical electrode order is not a one-dimensional neighborhood.
        self.conv = ConvModule(dim, options['temporal_kernel'] if axis == 'time' else 1)
        self.norm1, self.norm2, self.norm3, self.norm4 = [nn.LayerNorm(dim) for _ in range(4)]

    def forward(self, x):
        x = x + .5 * self.ff1(self.norm1(x))
        x = x + self.attn(self.norm2(x))
        x = x + self.conv(self.norm3(x))
        return x + .5 * self.ff2(self.norm4(x))


class ConformerBackbone(nn.Module):
    def configure(self, options, metadata, attention, tokens, axis):
        positive_options(options, 'dim', 'depth', 'expansion', 'temporal_kernel')
        if options['temporal_kernel'] % 2 != 1 or not 0 <= options['dropout'] < 1:
            raise ValueError('Conformer requires an odd temporal_kernel and dropout in [0,1)')
        self.metadata, self.num_tokens, self.token_axis = metadata, tokens, axis
        self.layers = nn.ModuleList([ConformerBlock(options, attention, tokens, i, axis) for i in range(options['depth'])])
        self.norm = nn.LayerNorm(options['dim'])
        self.cls = nn.Linear(options['dim'], metadata['num_classes'])
        self.pos_emb = nn.Parameter(torch.randn(1, tokens, options['dim']))

    @property
    def classifier(self):
        return self.cls

    def classify(self, tokens):
        x = tokens + .1 * self.pos_emb
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.cls(x.mean(1) + x.max(1).values)
