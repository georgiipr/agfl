from .._shared.layers import check_input
from .backbone import EEGNetBackbone


class EEGModel(EEGNetBackbone):
    model_variant = 'eeg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] not in {'electrode', 'time'}:
            raise ValueError('EEGNet EEG attention_axis must be electrode or time')
        super().__init__(options, metadata, attention, electrode_tokens=options['attention_axis'] == 'electrode')

    def forward(self, x):
        check_input(x, self.metadata)
        batch, channels, samples = x.shape
        if self.token_axis == 'electrode':
            # Apply EEGNet's local feature extractor independently per electrode.
            encoded = self.convolve(x.reshape(batch * channels, 1, 1, samples))
            tokens = self.electrode_position(encoded.reshape(batch, channels, -1))
            return self.fc(self.attn_blocks[0](tokens).mean(dim=1))
        encoded = self.convolve(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        return self.fc(self.flatten(self.attn_blocks[0](encoded)))
