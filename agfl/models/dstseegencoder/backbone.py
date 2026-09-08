"""DSTS down-projection, parallel temporal/attention branches and logit ensemble."""
import torch
from torch import nn
from torch.nn import functional as F
from .._shared.layers import CausalConv1d, PositionalEncoding, positive_options


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.proj, self.out = nn.Linear(dim, 2 * hidden), nn.Linear(hidden, dim)

    def forward(self, x):
        gate, value = self.proj(x).chunk(2, dim=-1)
        return self.out(F.silu(gate) * value)


class AttentionBlock(nn.Module):
    def __init__(self, dim, options, attention, tokens, index, axis):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(dim, eps=1e-6), nn.RMSNorm(dim, eps=1e-6)
        self.attn = attention(dim, tokens, index, options['depth'], axis)
        self.mlp = SwiGLU(dim, max(1, int(dim * options['mlp_ratio'])))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


def temporal_stage(dim, dilation, dropout, activation):
    return nn.Sequential(CausalConv1d(dim, dim, 3, dilation), nn.BatchNorm1d(dim), activation(), nn.Dropout(dropout),
                         CausalConv1d(dim, dim, 3, dilation), nn.BatchNorm1d(dim), activation(), nn.Dropout(dropout))


class Branch(nn.Module):
    def __init__(self, options, metadata, attention, tokens, axis):
        super().__init__()
        dim = options['dim']
        self.dropout = nn.Dropout(options['dropout'])
        self.stage1 = temporal_stage(dim, 1, options['dropout'], nn.SiLU)
        self.stage2 = temporal_stage(dim, 2, options['dropout'], nn.ReLU)
        self.electrode_position = PositionalEncoding(dim, tokens) if axis == 'electrode' else nn.Identity()
        self.transformer = nn.Sequential(*[AttentionBlock(dim, options, attention, tokens, i, axis) for i in range(options['depth'])])
        self.classifier = nn.Sequential(nn.Linear(dim, dim * 2), nn.ReLU(), nn.Dropout(options['dropout']),
                                        nn.Linear(dim * 2, metadata['num_classes']))

    def encode(self, x, batch, channels, electrode_tokens):
        x = self.dropout(x)
        stage1 = self.stage1(x)
        temporal = F.relu(self.stage2(stage1) + stage1)[:, :, -1]
        if electrode_tokens:
            tokens = temporal.reshape(batch, channels, -1)
            return tokens.mean(1) + self.transformer(self.electrode_position(tokens)).mean(1)
        return temporal + self.transformer(x.transpose(1, 2))[:, -1]


class DSTSBackbone(nn.Module):
    def __init__(self, options, metadata, attention, *, electrode_tokens):
        super().__init__()
        positive_options(options, 'dim', 'temporal_kernel', 'pool1', 'pool2', 'num_branches', 'depth')
        if not 0 <= options['dropout'] < 1 or options['mlp_ratio'] <= 0:
            raise ValueError('Invalid DSTS dropout or mlp_ratio')
        self.metadata, self.electrode_tokens = metadata, electrode_tokens
        self.token_axis = 'electrode' if electrode_tokens else 'time'
        dim, first = options['dim'], max(1, options['dim'] // 2)
        channels = 1 if electrode_tokens else metadata['channels']
        self.down_projector = nn.Sequential(
            nn.Conv2d(1, first, (options['temporal_kernel'], 1)), nn.BatchNorm2d(first),
            nn.Conv2d(first, dim, (1, channels)), nn.BatchNorm2d(dim), nn.ELU(),
            nn.AvgPool2d((options['pool1'], 1)), nn.Dropout(options['dropout']),
            nn.ZeroPad2d((0, 0, 7, 8)), nn.Conv2d(dim, dim, (16, 1)), nn.BatchNorm2d(dim), nn.ELU(),
            nn.AvgPool2d((options['pool2'], 1)), nn.Dropout(options['dropout']))
        steps = (metadata['samples'] - options['temporal_kernel'] + 1) // options['pool1'] // options['pool2']
        if steps < 1:
            raise ValueError('DSTS temporal kernel/pooling leaves no samples')
        self.num_tokens = metadata['channels'] if electrode_tokens else steps
        self.branches = nn.ModuleList([Branch(options, metadata, attention, self.num_tokens, self.token_axis) for _ in range(options['num_branches'])])
        self.feature_readout = nn.Identity()

    def classify(self, x, batch, channels):
        features = self.down_projector(x).squeeze(-1)
        branches = self.feature_readout(torch.stack([branch.encode(features, batch, channels, self.electrode_tokens) for branch in self.branches], dim=1))
        return torch.stack([branch.classifier(branches[:, index]) for index, branch in enumerate(self.branches)]).mean(0)
