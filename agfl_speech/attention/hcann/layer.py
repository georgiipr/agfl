"""Multi-head self-attention as written in the dataset authors' HCANN model."""
import math

from torch import nn


class HCANNAttention(nn.Module):
    """Bias-free joint Q/K/V projection, softmax routing, output projection.

    This is the ``Attention`` module of the authors' HCANN code: scaled
    dot-product attention whose Q/K/V projection has no bias and whose output
    passes a linear layer and dropout. In HCANN its tokens are the electrode
    rows of every convolutional feature map; here they are the electrode tokens
    of the selected backbone, so it mixes electrodes exactly like ``mha`` and
    ``agfl`` do. It differs from ``mha`` only in the missing Q/K/V bias and the
    optional output dropout and pre-normalization.
    """

    def __init__(self, dim, heads, *, dropout=0.0, pre_norm=False):
        super().__init__()
        if type(dim) is not int or type(heads) is not int or heads < 1 or dim % heads:
            raise ValueError('dim must be divisible by heads')
        if type(pre_norm) is not bool:
            raise ValueError('pre_norm must be a boolean')
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError('dropout must lie in [0, 1)')
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.options = {'dim': dim, 'heads': heads, 'dropout': dropout, 'pre_norm': pre_norm}
        self.norm = nn.LayerNorm(dim) if pre_norm else None
        self.to_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.to_out = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def project(self, x):
        """Return Q, K and V as [batch, heads, tokens, head_dim]."""
        batch, tokens, _ = x.shape
        if self.norm is not None:
            x = self.norm(x)
        return tuple(part.reshape(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
                     for part in self.to_qkv(x).chunk(3, dim=-1))

    def routing(self, q, k):
        return (q @ k.transpose(-1, -2) * self.head_dim ** -0.5).softmax(-1)

    def forward(self, x):
        batch, tokens, _ = x.shape
        q, k, v = self.project(x)
        weights = self.routing(q, k)
        self.last_attn = weights.detach()
        return self.to_out((weights @ v).transpose(1, 2).reshape(batch, tokens, self.dim))
