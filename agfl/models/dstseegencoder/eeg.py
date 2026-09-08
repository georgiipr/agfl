from .._shared.layers import check_input
from .backbone import DSTSBackbone


class EEGModel(DSTSBackbone):
    model_variant = 'eeg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] not in {'electrode', 'time'}:
            raise ValueError('DSTS EEG attention_axis must be electrode or time')
        super().__init__(options, metadata, attention, electrode_tokens=options['attention_axis'] == 'electrode')

    def forward(self, x):
        check_input(x, self.metadata)
        batch, channels, samples = x.shape
        arranged = x.reshape(batch * channels, 1, samples, 1) if self.electrode_tokens else x.transpose(1, 2).unsqueeze(1)
        return self.classify(arranged, batch, channels)
