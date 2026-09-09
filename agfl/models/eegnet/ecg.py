from .._shared.layers import check_input
from .backbone import EEGNetBackbone


class ECGModel(EEGNetBackbone):
    model_variant = 'ecg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] != 'time':
            raise ValueError('EEGNet ECG attention operates over time')
        super().__init__(options, metadata, attention, electrode_tokens=False)

    def forward(self, x):
        check_input(x, self.metadata)
        # Collapse the few ECG leads, retaining the pooled temporal sequence.
        encoded = self.convolve(x.unsqueeze(1)).squeeze(2).transpose(1, 2)
        mixed = self.attention_dropout(self.attn_blocks[0](encoded))
        return self.fc(self.flatten(encoded + mixed if self.attention_residual else mixed))
