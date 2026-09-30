import torch

from .._shared.layers import check_input
from .backbone import EEGNetBackbone


class EEGModel(EEGNetBackbone):
    model_variant = 'eeg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] != 'electrode' and not getattr(attention, 'historical', False):
            raise ValueError('EEGNet EEG attention must operate across electrodes')
        if options['attention_axis'] not in {'electrode', 'time'}:
            raise ValueError('EEGNet EEG attention_axis must be electrode or time')
        super().__init__(options, metadata, attention, electrode_tokens=options['attention_axis'] == 'electrode')

    def forward(self, x):
        check_input(x, self.metadata)
        batch, channels, samples = x.shape
        if self.token_axis == 'electrode':
            if self.spatial_fusion:
                # Temporal filters share parameters, not electrode values.
                # Evaluate the shared BatchNorm exactly once per input batch.
                temporal = self.block1(x.unsqueeze(1))  # [B,F1,C,T]
                spatial_features = self.spatial_block3(self.spatial_block2(temporal)).flatten(1)
                electrode_features = temporal.permute(0, 2, 1, 3).reshape(
                    batch * channels, temporal.shape[1], 1, samples)
                encoded = self.block3(self.block2(electrode_features))
            else:
                # Preserve the exact computation of already saved checkpoints.
                encoded = self.convolve(x.reshape(batch * channels, 1, 1, samples))
            local = encoded.reshape(batch, channels, -1)
            if self.compact_electrodes:
                local = self.electrode_projection(local)
                mixed = self.attention_dropout(self.attn_blocks[0](self.electrode_position(local)))
                features = self.spatial_readout(local + mixed if self.attention_residual else mixed)
                if self.spatial_fusion:
                    # Keep every pooled bin from the spatial EEGNet path. The
                    # 32-feature attention summary is additional information,
                    # rather than the only route to the classifier.
                    features = torch.cat((spatial_features, features), dim=1)
                return self.fc(features)
            mixed = self.attention_dropout(self.attn_blocks[0](self.electrode_position(local)))
            if self.attention_residual:
                mixed = local + mixed
            # [B,C,F,T] -> [B*T,C,F] -> learned spatial filtering -> [B,F,T].
            spatial = mixed.reshape(batch, channels, self.f2, self.steps).permute(0, 3, 1, 2)
            spatial = self.spatial_readout(spatial.reshape(batch * self.steps, channels, self.f2))
            return self.fc(spatial.reshape(batch, self.steps, self.f2).transpose(1, 2).flatten(1))
        encoded = self.convolve(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        mixed = self.attention_dropout(self.attn_blocks[0](encoded))
        return self.fc(self.flatten(encoded + mixed if self.attention_residual else mixed))
