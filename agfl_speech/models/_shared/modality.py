"""The scientific meaning of a graph node is fixed by the signal modality."""


def validate_attention_domain(modality, model_options, attention_options):
    if modality == 'eeg':
        if model_options.get('attention_axis', 'electrode') != 'electrode':
            raise ValueError('EEG attention must operate across electrodes '
                             '(model_options.attention_axis=electrode).')
        if attention_options.get('temporal_bias', False):
            raise ValueError('EEG electrode graphs cannot use temporal_bias: channel indices '
                             'are not time or physical distance. Disable temporal_bias.')
        if 'electrode_dim' in model_options:
            dim = model_options['electrode_dim']
            if type(dim) is not int or dim < 1:
                raise ValueError('EEGNet electrode_dim must be a positive integer for new runs')
            heads = attention_options.get('heads', 4)
            if type(heads) is not int or heads < 1 or dim % heads:
                raise ValueError('EEGNet electrode_dim must be divisible by attention heads')
        architecture = model_options.get('electrode_architecture')
        if architecture is not None and architecture not in ('compact', 'spatial_fusion', 'pre_spatial'):
            raise ValueError('EEGNet electrode_architecture must be compact, spatial_fusion or pre_spatial')
        if 'max_norm_schedule' in model_options:
            # eegnet_original: the dataset authors' network; fail at plan time,
            # not when the model is built inside a queued job.
            if model_options['max_norm_schedule'] not in ('init', 'every_step'):
                raise ValueError('eegnet_original max_norm_schedule must be init or every_step')
            if architecture != 'pre_spatial':
                raise ValueError('eegnet_original hosts the attention before its spatial filter '
                                 '(electrode_architecture=pre_spatial)')
            widths = [model_options.get(key) for key in ('f1', 'd', 'f2')]
            if any(type(value) is not int for value in widths) or widths[0] * widths[1] != widths[2]:
                raise ValueError('eegnet_original requires f2 == f1 * d')
        if architecture == 'pre_spatial':
            # Attention runs on the f1 temporal-filter outputs of every electrode.
            f1, heads = model_options.get('f1', 16), attention_options.get('heads', 4)
            if type(f1) is not int or f1 < 1 or type(heads) is not int or heads < 1 or f1 % heads:
                raise ValueError('EEGNet pre_spatial attention uses model_options.f1 features per electrode; '
                                 'f1 must be divisible by attention_options.heads')
    else:
        raise ValueError('Only EEG datasets are supported')


def validate_model_graph(model, metadata):
    """Fail before training if any backbone puts attention on the wrong axis."""
    layers = [m for m in model.modules() if getattr(m, 'is_attention', False)]
    if not layers:
        raise ValueError('A model must contain the selected attention mechanism')
    for layer in [model, *layers]:
        axis = getattr(layer, 'token_axis', None)
        if axis != 'electrode' or layer.num_tokens != metadata['channels']:
            raise ValueError('Every EEG attention layer must have one node per input '
                             'electrode, with token_axis=electrode')
