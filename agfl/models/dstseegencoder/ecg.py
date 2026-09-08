from .._shared.layers import check_input
from .backbone import DSTSBackbone


class ECGModel(DSTSBackbone):
    model_variant = 'ecg'

    def __init__(self, options, metadata, attention):
        if options['attention_axis'] != 'time':
            raise ValueError('DSTS ECG attention operates over time')
        super().__init__(options, metadata, attention, electrode_tokens=False)

    def forward(self, x):
        check_input(x, self.metadata)
        return self.classify(x.transpose(1, 2).unsqueeze(1), x.shape[0], x.shape[1])
