from torch import nn
from .._shared.layers import check_input, positive_options
from .._shared.pooling import DeterministicAdaptiveAvgPool1d
from .backbone import ConformerBackbone


class EEGModel(ConformerBackbone):
    model_variant = 'eeg'

    def __init__(self, options, metadata, attention):
        super().__init__()
        positive_options(options, 'temporal_bins', 'dim', 'temporal_kernel')
        dim, bins = options['dim'], options['temporal_bins']
        self.input = nn.Sequential(nn.Conv1d(1, dim, options['temporal_kernel'], padding=options['temporal_kernel'] // 2),
                                   nn.BatchNorm1d(dim), nn.SiLU(), DeterministicAdaptiveAvgPool1d(bins),
                                   nn.Flatten(), nn.Linear(dim * bins, dim))
        self.configure(options, metadata, attention, metadata['channels'], 'electrode')

    def forward(self, x):
        check_input(x, self.metadata)
        batch, channels, samples = x.shape
        tokens = self.input(x.reshape(batch * channels, 1, samples)).reshape(batch, channels, -1)
        return self.classify(tokens)
