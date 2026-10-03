"""The dataset authors' EEGNet with the selected attention across electrodes."""
from ..eegnet.eeg import EEGModel


MAX_NORM_SCHEDULES = ('init', 'every_step')


class OriginalEEGNet(EEGModel):
    """Authors' blocks; attention mixes electrodes between block1 and block2.

    block1 (temporal filters + BatchNorm), block2 (full-channel depthwise
    spatial filter, BatchNorm, ELU, pooling, dropout), block3 (separable
    convolution, BatchNorm, ELU, pooling, dropout) and the linear classifier are
    the authors' layers with the authors' sizes. The attention sees one node per
    electrode at every time step, carrying that electrode's F1 temporal-filter
    outputs, and its residual output feeds the authors' spatial filter. This is
    the project EEGNet's ``pre_spatial`` layout, so both EEGNets host the
    attention at the same place and differ only in their settings.
    """

    def __init__(self, options, metadata, attention):
        if options['max_norm_schedule'] not in MAX_NORM_SCHEDULES:
            raise ValueError('eegnet_original max_norm_schedule must be init or every_step')
        if options['electrode_architecture'] != 'pre_spatial':
            raise ValueError('eegnet_original hosts the attention before its spatial filter '
                             '(electrode_architecture=pre_spatial)')
        if options['f1'] * options['d'] != options['f2']:
            # The authors' separable convolution, Conv2d(D*F1, F2, groups=F2), is
            # depthwise only for D*F1 == F2; other widths are a different model.
            raise ValueError('eegnet_original requires f2 == f1 * d')
        backbone = {key: value for key, value in options.items() if key != 'max_norm_schedule'}
        backbone.update(attention_axis='electrode', electrode_dim=None,
                        spatial_readout='learned', initialization='pytorch')
        # The backbone constructor applies the constraint once (first call below).
        super().__init__(backbone, metadata, attention)
        self.max_norm_schedule = options['max_norm_schedule']

    # Class-level defaults: the backbone constructor calls clip_weights() before
    # this subclass can assign instance attributes.
    max_norm_schedule = 'init'
    _max_norm_applied = False

    def clip_weights(self):
        # The shared engine calls this hook after every optimizer step; the
        # authors constrain the weights only once, at construction.
        if self.max_norm_schedule == 'every_step' or not self._max_norm_applied:
            super().clip_weights()
            self._max_norm_applied = True
