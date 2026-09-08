"""EEGEncoder convolution + causal TCN + parallel attention fusion."""
from torch import nn
from torch.nn import functional as F
from .._shared.layers import CausalConv1d, PositionalEncoding, positive_options


class TemporalResidual(nn.Module):
    def __init__(self, incoming, outgoing, kernel, dilation, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            CausalConv1d(incoming, outgoing, kernel, dilation), nn.BatchNorm1d(outgoing), nn.ELU(), nn.Dropout(dropout),
            CausalConv1d(outgoing, outgoing, kernel, dilation), nn.BatchNorm1d(outgoing), nn.ELU(), nn.Dropout(dropout))
        self.residual = nn.Conv1d(incoming, outgoing, 1) if incoming != outgoing else nn.Identity()

    def forward(self, x):
        # Each residual block is evaluated once; no repeated previous block.
        return F.elu(self.layers(x) + self.residual(x))


class EEGEncoderBackbone(nn.Module):
    def __init__(self, options, metadata, attention, *, electrode_tokens):
        super().__init__()
        positive_options(options, 'eegn_F1', 'eegn_D', 'eegn_kernelSize', 'pool1', 'pool2', 'tcn_depth', 'tcn_kernelSize', 'tcn_filters')
        if not 0 <= options['dropout'] < 1:
            raise ValueError('EEGEncoder dropout must lie in [0,1)')
        self.metadata = metadata
        self.token_axis = 'electrode' if electrode_tokens else 'time'
        f1, f2 = options['eegn_F1'], options['eegn_F1'] * options['eegn_D']
        channels = 1 if electrode_tokens else metadata['channels']
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, f1, (options['eegn_kernelSize'], 1), padding='same', bias=False), nn.BatchNorm2d(f1),
            nn.Conv2d(f1, f2, (1, channels), groups=f1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((options['pool1'], 1)), nn.Dropout(options['dropout']),
            nn.Conv2d(f2, f2, (16, 1), padding='same', bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((options['pool2'], 1)), nn.Dropout(options['dropout']))
        steps = metadata['samples'] // options['pool1'] // options['pool2']
        if steps < 1:
            raise ValueError('EEGEncoder pooling leaves no time samples')
        dim = options['tcn_filters']
        self.tcn_block = nn.Sequential(*[TemporalResidual(f2 if i == 0 else dim, dim, options['tcn_kernelSize'], 2**i, options['dropout'])
                                         for i in range(options['tcn_depth'])])
        self.attention_input = nn.Linear(f2, dim) if not electrode_tokens and f2 != dim else nn.Identity()
        self.num_tokens = metadata['channels'] if electrode_tokens else steps
        self.pe = PositionalEncoding(dim, self.num_tokens)
        self.attn_blocks = nn.ModuleList([attention(dim, self.num_tokens, token_axis=self.token_axis)])
        self.aa_drop = nn.Dropout(options['dropout'])
        self.classifier = nn.Sequential(nn.Linear(dim, max(1, dim // 2)), nn.ELU(), nn.Dropout(options['dropout']),
                                        nn.Linear(max(1, dim // 2), metadata['num_classes']))

    def convolve(self, x):
        return self.aa_drop(self.conv_block(x.unsqueeze(1).transpose(2, 3)).squeeze(-1))

    def temporal_prediction(self, features):
        local = self.tcn_block(features)[:, :, -1]
        tokens = self.attention_input(features.transpose(1, 2))
        mixed = self.attn_blocks[0](self.pe(tokens)).mean(dim=1)
        return self.classifier(local + self.aa_drop(mixed))
