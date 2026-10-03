"""Optional temporal-score bias and message gate; no changes to default AGFL.

Both modules initialize without random draws and preserve the retained forward
function. Keeping them here makes this experiment removable as a single unit.
"""
import torch
from torch import nn
from torch.nn import functional as F


def validate_options(options):
    for key in ('temporal_bias', 'output_gate'):
        if type(options.get(key, False)) is not bool:
            raise ValueError(f'{key} must be a boolean')
    if options.get('temporal_bias', False):
        if options['top_k_stage'] != 'scores' or options['graph_normalization'] != 'softmax':
            raise ValueError('temporal_bias requires top_k_stage=scores and graph_normalization=softmax')
    if options.get('output_gate', False) and options.get('coefficient_conditioning') != 'token_contrast':
        raise ValueError('output_gate requires coefficient_conditioning=token_contrast')


class RelativeTemporalBias(nn.Module):
    """Per-head distance, signed lag and self-edge score offsets.

    The signed linear lag is softmax-equivalent to a source-position ramp;
    it is not a separately expressive past/future relation. Offsets precede
    both top-k selection and softmax. N=1 is well-defined too.
    """
    def __init__(self, heads):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(heads, 3))

    def forward(self, tokens):
        index = torch.arange(tokens, device=self.weight.device)
        lag = index[None, :] - index[:, None]  # destination i, source j
        relative = lag.to(self.weight.dtype) / max(tokens - 1, 1)
        basis = torch.stack((relative.abs(), relative, (lag == 0).to(relative.dtype)))
        return torch.einsum('hf,fij->hij', self.weight, basis)


class TokenOutputGate(nn.Module):
    """Scale each head/token message by 2*sigmoid(w*d+b), initially one."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, 2 * dim))
        self.bias = nn.Parameter(torch.zeros(1))

    def gate_values(self, values, first_hop):
        if values.shape != first_hop.shape:
            raise ValueError('Output gate first_hop must match values')
        with torch.autocast(device_type=values.device.type, enabled=False):
            working = values if values.dtype == torch.float64 else values.float()
            contrast = first_hop.to(working.dtype) - working
            descriptor = torch.cat((F.layer_norm(working, (working.shape[-1],), eps=1e-5),
                                    F.layer_norm(contrast, (working.shape[-1],), eps=1e-5)), dim=-1)
            scores = F.linear(descriptor, self.weight.to(working.dtype), self.bias.to(working.dtype))
            return 2 * scores.sigmoid()

    def forward(self, message, values, first_hop):
        return message * self.gate_values(values, first_hop).to(message.dtype)
