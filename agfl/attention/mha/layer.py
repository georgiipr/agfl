"""Standard MHA, with optional matched temporal-bias/output-gate controls."""
import torch
from torch import nn

from ..projections import ProjectedMixer
from ..agfl.temporal_gate import RelativeTemporalBias, TokenOutputGate


class MultiHeadAttention(ProjectedMixer):
    def __init__(self, dim, heads, *, temporal_bias=False, output_gate=False):
        if type(temporal_bias) is not bool or type(output_gate) is not bool:
            raise ValueError('temporal_bias and output_gate must be booleans')
        super().__init__(dim, heads)
        self.options = {'dim': dim, 'heads': heads, 'temporal_bias': temporal_bias,
                        'output_gate': output_gate}
        # The exact AGFL modules, per head, with zero initialization and no RNG
        # draws. Disabled options add no checkpoint tensors or forward changes.
        self.temporal_bias = RelativeTemporalBias(heads) if temporal_bias else None
        self.output_gates = nn.ModuleList([TokenOutputGate(self.head_dim) for _ in range(heads)]
                                          if output_gate else [])

    def forward(self, x):
        q, k, v = self.project(x)
        scores = q @ k.transpose(-1, -2) / self.head_dim**.5
        if self.temporal_bias is not None:
            if getattr(self, 'token_axis', 'time') != 'time':
                raise ValueError('temporal_bias requires time tokens, not electrode tokens')
            scores = scores + self.temporal_bias(x.shape[1]).to(scores.dtype)
        weights = scores.softmax(-1)
        self.last_attn = weights.detach()
        message = weights @ v
        if self.output_gates:
            # MHA's first hop is its message. The gate descriptor is exactly
            # [LN(V), LN(A V - V)], as in the retained polynomial AGFL recipe.
            message = torch.stack([gate(message[:, h], v[:, h], message[:, h])
                                   for h, gate in enumerate(self.output_gates)], dim=1)
        return self.combine(message)
