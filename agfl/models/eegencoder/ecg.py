from .._shared.layers import check_input
from .backbone import EEGEncoderBackbone


class ECGModel(EEGEncoderBackbone):
    model_variant = 'ecg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] != 'time':
            raise ValueError('EEGEncoder ECG attention operates over time')
        super().__init__(options, metadata, attention, electrode_tokens=False)

    def forward(self, x):
        check_input(x, self.metadata)
        return self.temporal_prediction(self.convolve(x))
