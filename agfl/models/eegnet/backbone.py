"""EEGNet's temporal, depthwise and separable convolutional feature extractor."""
import math
import torch
from torch import nn
from .._shared.layers import ElectrodeIdentity, ElectrodeReadout, PositionalEncoding, positive_options


ELECTRODE_ARCHITECTURES = ('compact', 'spatial_fusion', 'pre_spatial')


class EEGNetBackbone(nn.Module):
    def __init__(self, options, metadata, attention, *, electrode_tokens):
        super().__init__()
        positive_options(options, 'temp_kernel', 'f1', 'd', 'f2', 'pk1', 'pk2')
        if not 0 <= options['dropout_rate'] < 1 or min(options['max_norm1'], options['max_norm2']) <= 0:
            raise ValueError('Invalid EEGNet dropout or max norm')
        self.metadata = metadata
        self.max_norm1, self.max_norm2 = options['max_norm1'], options['max_norm2']
        momentum, eps = options['batch_norm_momentum'], options['batch_norm_eps']
        if not math.isfinite(momentum) or not 0 < momentum <= 1 or not math.isfinite(eps) or eps <= 0:
            raise ValueError('Invalid EEGNet BatchNorm momentum or epsilon')
        if not 0 <= options['attention_dropout'] < 1:
            raise ValueError('EEGNet attention_dropout must lie in [0, 1)')
        if options['initialization'] not in {'pytorch', 'xavier'}:
            raise ValueError('EEGNet initialization must be pytorch or xavier')
        architecture = options.get('electrode_architecture', 'compact')
        if architecture not in ELECTRODE_ARCHITECTURES:
            raise ValueError('EEGNet electrode_architecture must be compact, spatial_fusion or pre_spatial')
        # pre_spatial: the selected attention mixes the electrodes at every time
        # step of the shared temporal-filter output, BEFORE EEGNet's own
        # full-channel depthwise spatial convolution. Everything reaching the
        # classifier passes through that spatial filter, ELU, pooling and the
        # separable convolution; there is no parallel per-electrode path.
        self.pre_spatial = electrode_tokens and architecture == 'pre_spatial'
        self.compact_electrodes = (electrode_tokens and not self.pre_spatial
                                   and options.get('electrode_dim') is not None)
        if self.compact_electrodes:
            positive_options(options, 'electrode_dim')
        self.spatial_fusion = electrode_tokens and architecture == 'spatial_fusion'
        if self.spatial_fusion and not self.compact_electrodes:
            raise ValueError('EEGNet spatial_fusion requires a positive electrode_dim')
        def batch_norm(features):
            return nn.BatchNorm2d(features, momentum=momentum, eps=eps)
        self.token_axis = 'electrode' if electrode_tokens else 'time'
        f1, expanded, f2 = options['f1'], options['f1'] * options['d'], options['f2']
        # Compact/fusion electrode encoders convolve one electrode at a time; the
        # temporal layout and pre_spatial keep the full-channel spatial filter.
        channels = 1 if electrode_tokens and not self.pre_spatial else metadata['channels']
        self.block1 = nn.Sequential(nn.Conv2d(1, f1, (1, options['temp_kernel']), padding='same', bias=False), batch_norm(f1))
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, expanded, (channels, 1), groups=f1, bias=False),
            batch_norm(expanded), nn.ELU(), nn.AvgPool2d((1, options['pk1'])), nn.Dropout(options['dropout_rate']))
        self.block3 = nn.Sequential(
            nn.Conv2d(expanded, expanded, (1, 16), padding='same', groups=expanded, bias=False),
            nn.Conv2d(expanded, f2, 1, bias=False), batch_norm(f2), nn.ELU(),
            nn.AvgPool2d((1, options['pk2'])), nn.Dropout(options['dropout_rate']))
        steps = metadata['samples'] // options['pk1'] // options['pk2']
        if steps < 1:
            raise ValueError('EEGNet pooling leaves no temporal samples; reduce pk1/pk2')
        dim = (f1 if self.pre_spatial else options['electrode_dim'] if self.compact_electrodes
               else f2 * steps if electrode_tokens else f2)
        self.num_tokens = metadata['channels'] if electrode_tokens else steps
        if self.pre_spatial:
            # Tokens are the f1 temporal-filter outputs of one electrode at one
            # time step; the identity embedding lets the graph depend on the
            # sensor, not only on instantaneous amplitudes.
            self.electrode_position = ElectrodeIdentity(dim, self.num_tokens)
        elif self.compact_electrodes:
            # The projection sees one electrode only, including its time bins.
            self.electrode_projection = nn.Linear(f2 * steps, dim, bias=False)
            self.electrode_position = ElectrodeIdentity(dim, self.num_tokens)
        else:
            self.electrode_position = PositionalEncoding(dim, self.num_tokens) if electrode_tokens else nn.Identity()
        self.attn_blocks = nn.ModuleList([attention(dim, self.num_tokens, token_axis=self.token_axis)])
        self.attention_dropout = nn.Dropout(options['attention_dropout'])
        self.attention_residual = options['attention_residual']
        if type(self.attention_residual) is not bool:
            raise ValueError('attention_residual must be boolean')
        # Share each feature's learned electrode filter across pooled time bins.
        # pre_spatial has no readout: its spatial filter is block2's convolution.
        readout_features = dim if self.compact_electrodes else f2
        self.spatial_readout = (ElectrodeReadout(metadata['channels'], readout_features, options['spatial_readout'])
                                if electrode_tokens and not self.pre_spatial else None)
        self.f2, self.steps = f2, steps
        self.flatten = nn.Flatten()
        if self.spatial_fusion:
            # Spatial filtering must see signed, full-resolution electrode
            # waveforms BEFORE ELU/pooling. A late readout of independently
            # rectified and compressed electrodes cannot reproduce this path.
            # Share block1 only: the two paths need separate BN/dropout after
            # their different spatial operations.
            self.spatial_block2 = nn.Sequential(
                nn.Conv2d(f1, expanded, (metadata['channels'], 1), groups=f1, bias=False),
                batch_norm(expanded), nn.ELU(), nn.AvgPool2d((1, options['pk1'])),
                nn.Dropout(options['dropout_rate']))
            self.spatial_block3 = nn.Sequential(
                nn.Conv2d(expanded, expanded, (1, 16), padding='same', groups=expanded, bias=False),
                nn.Conv2d(expanded, f2, 1, bias=False), batch_norm(f2), nn.ELU(),
                nn.AvgPool2d((1, options['pk2'])), nn.Dropout(options['dropout_rate']))
        # Calculate the shape without a dummy forward altering BatchNorm/RNG.
        classifier_features = dim if self.compact_electrodes else f2 * steps
        if self.spatial_fusion:
            classifier_features += f2 * steps
        self.fc = nn.Linear(classifier_features, metadata['num_classes'])
        if options['initialization'] == 'xavier':
            # Initialize the EEGNet feature extractor/head, leaving AGFL's
            # documented graph/filter initialization untouched.
            blocks = [self.block1, self.block2, self.block3, self.fc]
            if self.compact_electrodes:
                blocks.append(self.electrode_projection)
            if self.spatial_fusion:
                blocks.extend([self.spatial_block2, self.spatial_block3])
            for block in blocks:
                for layer in block.modules():
                    if isinstance(layer, (nn.Conv2d, nn.Linear)):
                        nn.init.xavier_uniform_(layer.weight)
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)
        self.clip_weights()

    @property
    def classifier(self):
        return self.fc

    def convolve(self, x):
        return self.block3(self.block2(self.block1(x)))

    def clip_weights(self):
        with torch.no_grad():
            constrained = [(self.block2[0], self.max_norm1), (self.fc, self.max_norm2)]
            if self.spatial_fusion:
                constrained.append((self.spatial_block2[0], self.max_norm1))
            for layer, maximum in constrained:
                layer.weight.copy_(torch.renorm(layer.weight, p=2, dim=0, maxnorm=maximum))
            if self.spatial_readout is not None and self.spatial_readout.projection is not None:
                weights = self.spatial_readout.projection.weight
                weights.copy_(torch.renorm(weights, p=2, dim=0, maxnorm=self.max_norm1))
