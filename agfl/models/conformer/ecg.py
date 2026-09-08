from torch import nn
from .._shared.layers import check_input
from .backbone import ConformerBackbone


class ECGModel(ConformerBackbone):
    model_variant = 'ecg'

    def __init__(self, options, metadata, attention):
        super().__init__()
        self.input = nn.Linear(metadata['channels'], options['dim'])
        self.configure(options, metadata, attention, metadata['samples'], 'time')

    def forward(self, x):
        check_input(x, self.metadata)
        return self.classify(self.input(x.transpose(1, 2)))
