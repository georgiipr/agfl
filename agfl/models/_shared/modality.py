"""The scientific meaning of a graph node is fixed by the signal modality."""


def validate_attention_domain(modality, model_options, attention_options):
    if modality == 'eeg':
        if model_options.get('attention_axis', 'electrode') != 'electrode':
            raise ValueError('EEG attention must operate across electrodes. Temporal EEG '
                             'configurations are retired; saved results remain available '
                             'to analyze/diagnose. Use the eegnet-interchannel preset.')
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
        if architecture == 'pre_spatial':
            # Attention runs on the f1 temporal-filter outputs of every electrode.
            f1, heads = model_options.get('f1', 16), attention_options.get('heads', 4)
            if type(f1) is not int or f1 < 1 or type(heads) is not int or heads < 1 or f1 % heads:
                raise ValueError('EEGNet pre_spatial attention uses model_options.f1 features per electrode; '
                                 'f1 must be divisible by attention_options.heads')
        if model_options.get('temporal_statistics', 'mean') != 'mean':
            raise ValueError('The retired EEGNet power experiment is available only for saved-checkpoint replay')
    elif modality == 'ecg' and model_options.get('attention_axis', 'time') != 'time':
        raise ValueError('ECG attention must operate over time.')


def validate_model_graph(model, metadata):
    """Fail before training if any backbone puts attention on the wrong axis."""
    layers = [m for m in model.modules() if getattr(m, 'is_attention', False)]
    if not layers:
        raise ValueError('A model must contain the selected attention mechanism')
    for layer in [model, *layers]:
        axis = getattr(layer, 'token_axis', None)
        if metadata['modality'] == 'eeg':
            if axis != 'electrode' or layer.num_tokens != metadata['channels']:
                raise ValueError('Every EEG attention layer must have one node per input '
                                 'electrode, with token_axis=electrode')
        elif axis not in {'time', 'time_patch'}:
            raise ValueError('Every ECG attention layer must use time or time_patch tokens')
