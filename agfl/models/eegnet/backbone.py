"""EEGNet's temporal, depthwise and separable convolutional feature extractor."""
import torch
from torch import nn
from .._shared.layers import ElectrodeReadout, PositionalEncoding, positive_options


class EEGNetBackbone(nn.Module):
    def __init__(self, options, metadata, attention, *, electrode_tokens):
        super().__init__()
        positive_options(options, 'temp_kernel', 'f1', 'd', 'f2', 'pk1', 'pk2')
        if not 0 <= options['dropout_rate'] < 1 or min(options['max_norm1'], options['max_norm2']) <= 0:
            raise ValueError('Invalid EEGNet dropout or max norm')
        self.metadata = metadata
        self.max_norm1, self.max_norm2 = options['max_norm1'], options['max_norm2']
        self.token_axis = 'electrode' if electrode_tokens else 'time'
        f1, expanded, f2 = options['f1'], options['f1'] * options['d'], options['f2']
        channels = 1 if electrode_tokens else metadata['channels']
        self.block1 = nn.Sequential(nn.Conv2d(1, f1, (1, options['temp_kernel']), padding='same', bias=False), nn.BatchNorm2d(f1))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, expanded, (channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(expanded), nn.ELU(), nn.AvgPool2d((1, options['pk1'])), nn.Dropout(options['dropout_rate']))
        self.block3 = nn.Sequential(
            nn.Conv2d(expanded, expanded, (1, 16), padding='same', groups=expanded, bias=False),
            nn.Conv2d(expanded, f2, 1, bias=False), nn.BatchNorm2d(f2), nn.ELU(),
            nn.AvgPool2d((1, options['pk2'])), nn.Dropout(options['dropout_rate']))
        steps = metadata['samples'] // options['pk1'] // options['pk2']
        if steps < 1:
            raise ValueError('EEGNet pooling leaves no temporal samples; reduce pk1/pk2')
        dim = f2 * steps if electrode_tokens else f2
        self.num_tokens = metadata['channels'] if electrode_tokens else steps
        self.electrode_position = PositionalEncoding(dim, self.num_tokens) if electrode_tokens else nn.Identity()
        self.attn_blocks = nn.ModuleList([attention(dim, self.num_tokens, token_axis=self.token_axis)])
        self.attention_residual = options['attention_residual']
        if type(self.attention_residual) is not bool:
            raise ValueError('attention_residual must be boolean')
        # Share each feature's learned electrode filter across pooled time bins.
        self.spatial_readout = ElectrodeReadout(metadata['channels'], f2, options['spatial_readout']) if electrode_tokens else None
        self.f2, self.steps = f2, steps
        self.flatten = nn.Flatten()
        # Calculate the shape without a dummy forward altering BatchNorm/RNG.
        self.fc = nn.Linear(f2 * steps, metadata['num_classes'])
        self.clip_weights()

    @property
    def classifier(self):
        return self.fc

    def convolve(self, x):
        return self.block3(self.block2(self.block1(x)))

    def clip_weights(self):
        with torch.no_grad():
            for layer, maximum in ((self.block2[0], self.max_norm1), (self.fc, self.max_norm2)):
                layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=maximum))
            if self.spatial_readout is not None and self.spatial_readout.projection is not None:
                weights = self.spatial_readout.projection.weight
                weights.copy_(torch.renorm(weights, p=2, dim=0, maxnorm=self.max_norm1))
