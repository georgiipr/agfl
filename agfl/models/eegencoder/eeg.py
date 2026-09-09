from .._shared.layers import check_input
from .backbone import EEGEncoderBackbone


class EEGModel(EEGEncoderBackbone):
    model_variant = 'eeg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] not in {'electrode', 'time'}:
            raise ValueError('EEGEncoder EEG attention_axis must be electrode or time')
        super().__init__(options, metadata, attention, electrode_tokens=options['attention_axis'] == 'electrode')

    def forward(self, x):
        check_input(x, self.metadata)
        if self.token_axis == 'time':
            return self.temporal_prediction(self.convolve(x))
        batch, channels, samples = x.shape
        features = self.convolve(x.reshape(batch * channels, 1, samples))
        local = self.tcn_block(features)[:, :, -1].reshape(batch, channels, -1)
        mixed = self.spatial_readout(self.attn_blocks[0](self.pe(local)))
        return self.classifier(self.spatial_readout(local) + self.aa_drop(mixed))
