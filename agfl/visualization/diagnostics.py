"""Read-only checkpoint diagnostics; no optimizer or experiment-metric updates."""
import copy
import json
from pathlib import Path

import numpy as np

from agfl.storage import write_json
from .common import FigureWriter, slug
from .maps import mixer_map, map_statistics


def selected_indices(labels, indices, limit, seed):
    """Deterministic class-balanced sampling for visualization, not evaluation."""
    if type(limit) is not int or limit < 1:
        raise ValueError('max_samples must be a positive integer')
    indices = np.asarray(indices, dtype=int)
    rng = np.random.default_rng(seed)
    pools = [list(rng.permutation(indices[np.asarray(labels)[indices] == label])) for label in np.unique(np.asarray(labels)[indices])]
    selected = []
    while len(selected) < min(limit, len(indices)):
        for pool in pools:
            if pool and len(selected) < limit:
                selected.append(int(pool.pop()))
    return np.asarray(sorted(selected), dtype=int)


def model_inputs(bundle, indices, normalization):
    values = bundle.x[indices].copy()
    if normalization:
        values = (values - normalization['mean']) / normalization['std']
    return np.asarray(values, dtype=np.float32)


def eeg_info(metadata):
    import mne
    names, fs = metadata.get('channel_names'), metadata.get('sampling_rate')
    if not names or fs is None:
        raise ValueError('Scalp maps require channel names and a recorded sampling rate')
    montage = mne.channels.make_standard_montage('standard_1020')
    if not set(names) <= set(montage.ch_names):
        raise ValueError('Some channel positions are unavailable in the standard 10–20 montage')
    info = mne.create_info(names, sfreq=fs, ch_types='eeg')
    info.set_montage(montage)
    return info


def topomap(writer, values, info, name, caption):
    import mne
    figure, axis = writer.plt.subplots(figsize=(5, 4), constrained_layout=True)
    artist, _ = mne.viz.plot_topomap(np.asarray(values), info, axes=axis, show=False, contours=0)
    figure.colorbar(artist, ax=axis)
    axis.set_title(caption, fontsize=10)
    writer.save(figure, name, caption)


def signal_figures(writer, values, labels, metadata, info):
    from scipy.signal import welch, stft
    from scipy.integrate import trapezoid

    fs = metadata.get('sampling_rate')
    times = np.arange(values.shape[-1]) / fs if fs else np.arange(values.shape[-1])
    time_label = 'Time (s)' if fs else 'Sample'
    names = metadata.get('channel_names') or [f'Channel {i}' for i in range(values.shape[1])]
    classes = np.unique(labels)
    figure, axes = writer.plt.subplots(len(classes), 1, figsize=(11, max(3, 2.8 * len(classes))), squeeze=False, constrained_layout=True)
    for axis, label in zip(axes[:, 0], classes):
        signal = values[np.flatnonzero(labels == label)[0]]
        scale = max(float(np.std(signal, axis=-1).mean()) * 3, 1e-8)
        for channel in range(len(names)):
            axis.plot(times, signal[channel] / scale + channel, linewidth=.65)
        axis.set(yticks=range(len(names)), yticklabels=names, xlabel=time_label,
                 title=f'First selected example of class {label}; shared display scale {scale:.3g}', ylabel='Model input / scale + offset')
        axis.tick_params(axis='y', labelsize=6)
    writer.save(figure, 'signals/examples', 'Preprocessed model inputs. Amplitudes reflect the saved normalization; these are not raw-voltage traces.')
    if fs is None:
        writer.skip('signals/spectra', 'Dataset does not declare a physical sampling rate')
        return
    frequencies, spectra = welch(values, fs=fs, nperseg=min(values.shape[-1], int(fs)), axis=-1)
    spectra = spectra.mean(axis=0)
    figure, axes = writer.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for channel, name in enumerate(names):
        axes[0].semilogy(frequencies, np.maximum(spectra[channel], 1e-30), label=name)
    axes[0].set(xlabel='Frequency (Hz)', ylabel='Model-input power / Hz', title='Mean Welch spectra')
    if len(names) <= 4:
        axes[0].legend()
    artist = axes[1].imshow(10 * np.log10(np.maximum(spectra, 1e-30)), origin='lower', aspect='auto',
                            extent=(frequencies[0], frequencies[-1], -.5, len(names) - .5))
    axes[1].set(xlabel='Frequency (Hz)', yticks=range(len(names)), yticklabels=names, title='Channel spectra (dB)')
    axes[1].tick_params(axis='y', labelsize=7)
    figure.colorbar(artist, ax=axes[1])
    writer.save(figure, 'signals/spectra', 'Spectra averaged over the explicitly selected diagnostic examples, in model-input units.')
    if info is not None:
        for band, low, high in [('alpha', 8, 13), ('beta', 13, 30)]:
            mask = (frequencies >= low) & (frequencies <= high)
            if mask.sum() < 2:
                writer.skip('signals/' + band, 'Insufficient frequency bins for this band')
                continue
            power = trapezoid(spectra[:, mask], frequencies[mask], axis=-1)
            topomap(writer, power, info, 'signals/' + band + '_power', f'{band} power of normalized/model-input signals')
        channels = [name for name in ('C3', 'C4') if name in names]
        if channels:
            figure, axes = writer.plt.subplots(len(classes), len(channels), squeeze=False,
                                              figsize=(6 * len(channels), 3 * len(classes)), constrained_layout=True)
            for row, label in enumerate(classes):
                for column, name in enumerate(channels):
                    signal = values[labels == label, names.index(name)]
                    f, t, z = stft(signal, fs=fs, nperseg=min(signal.shape[-1], int(fs)), boundary=None, padded=False, axis=-1)
                    power = np.abs(z)**2
                    mask = f <= min(45, fs / 2)
                    artist = axes[row, column].pcolormesh(t, f[mask], 10 * np.log10(np.maximum(power.mean(0)[mask], 1e-30)), shading='auto')
                    axes[row, column].set(xlabel='Time (s)', ylabel='Frequency (Hz)', title=f'{name}, class {label}')
                    figure.colorbar(artist, ax=axes[row, column])
            writer.save(figure, 'signals/c3_c4_stft', 'Class-average C3/C4 STFT power of selected model inputs; descriptive only.')


def embedding_figure(writer, features, labels, method, seed):
    from sklearn.decomposition import PCA
    if min(features.shape) < 2 or len(features) < 3:
        writer.skip('embedding', 'At least three examples and two feature dimensions are required')
        return
    if not np.isfinite(features).all():
        raise ValueError('Nonfinite classifier features')
    if method == 'pca':
        coordinates = PCA(n_components=2, svd_solver='full').fit_transform(features)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reduced = PCA(n_components=min(50, features.shape[1], len(features)-1), svd_solver='full').fit_transform(features)
        coordinates = TSNE(n_components=2, perplexity=min(30., (len(features)-1)/3),
                           init='random', random_state=seed, learning_rate='auto', n_jobs=1).fit_transform(reduced)
    else:
        try:
            import umap
        except ImportError:
            writer.skip('embedding/umap', "Requires the optional 'umap' extra on the target machine")
            return
        coordinates = umap.UMAP(n_components=2, n_neighbors=min(15, len(features)-1),
                                init='random', random_state=seed, n_jobs=1).fit_transform(features)
    np.savez_compressed(writer.root / 'embedding.npz', features=features, coordinates=coordinates, labels=labels)
    figure, axis = writer.plt.subplots(figsize=(7, 5), constrained_layout=True)
    for label in np.unique(labels):
        selected = labels == label
        axis.scatter(*coordinates[selected].T, label=f'Class {label}', s=16, alpha=.7)
    axis.set(xlabel='Component 1', ylabel='Component 2', title=f'{method.upper()} of classifier-input features')
    axis.legend()
    writer.save(figure, 'embedding/' + method, 'Embedding is fitted to the selected diagnostic examples; separation is not an accuracy estimate.')


def graph_figures(writer, name, state, metadata, info, first_input):
    mean = state['sum'] / state['count']
    first = state['first']
    graph_dir = writer.root / 'mixers'
    graph_dir.mkdir(exist_ok=True)
    np.savez_compressed(graph_dir / (slug(name) + '.npz'), mean=mean, first=first)
    stats = {'sparsity_per_head': (state['sparsity'] / state['count']).tolist(),
             'entropy_per_head': [float(value / state['count']) if valid else None
                                  for value, valid in zip(state['entropy'], state['entropy_valid'])],
             'kind': state['kind'], 'token_axis': state['token_axis'],
             'samples': state['count'], 'zero_tolerance': 1e-8}
    heads = mean.shape[0]
    figure, axes = writer.plt.subplots(1, heads, figsize=(4 * heads, 4), squeeze=False, constrained_layout=True)
    tokens = mean.shape[-1]
    electrode = state['token_axis'] == 'electrode'
    axis_name = 'electrode' if electrode else 'time token'
    labels = metadata.get('channel_names') if electrode else None
    for head, axis in enumerate(axes[0]):
        matrix = mean[head]
        signed = (matrix < 0).any()
        magnitude = max(float(np.abs(matrix).max()), 1e-12)
        artist = axis.imshow(matrix, cmap='RdBu_r' if signed else 'viridis',
                             vmin=-magnitude if signed else 0, vmax=magnitude)
        axis.set(title=f'Head {head}', xlabel='Source ' + axis_name,
                 ylabel='Destination ' + axis_name)
        if labels and len(labels) == tokens:
            axis.set(xticks=range(tokens), yticks=range(tokens), xticklabels=labels, yticklabels=labels)
            axis.tick_params(axis='x', rotation=90, labelsize=6)
            axis.tick_params(axis='y', labelsize=6)
        figure.colorbar(artist, ax=axis, shrink=.7)
    figure.suptitle(name + ': ' + state['kind'], fontsize=10)
    writer.save(figure, 'mixers/' + slug(name) + '_heatmaps', 'Mean diagnostic mixing maps. AGFL maps show A before filtering; effective baseline weights can be signed.')
    figure, axes = writer.plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].bar(range(heads), stats['sparsity_per_head'])
    axes[0].set(title='Fraction |weight| ≤ 1e-8', xlabel='Head', ylim=(0, 1))
    axes[1].bar(range(heads), [np.nan if value is None else value for value in stats['entropy_per_head']])
    axes[1].set(title='Mean row entropy (nats)', xlabel='Head')
    if any(value is None for value in stats['entropy_per_head']):
        axes[1].text(.5, .95, 'Undefined for signed/unnormalized maps', transform=axes[1].transAxes, ha='center', va='top', fontsize=8)
    writer.save(figure, 'mixers/' + slug(name) + '_statistics', 'Statistics are averaged per example/head before graph averaging. Entropy is defined only for probability rows.')
    incoming = mean.mean(axis=(0, 1))
    if electrode and info is not None and tokens == metadata['channels']:
        topomap(writer, incoming, info, 'mixers/' + slug(name) + '_source_weights', 'Mean source-electrode weight; not causal importance')
    else:
        figure, axis = writer.plt.subplots(figsize=(8, 3), constrained_layout=True)
        axis.plot(range(tokens), incoming, marker='.')
        axis.set(xlabel='Source ' + axis_name + ' index',
                 ylabel='Mean source weight', title=name)
        writer.save(figure, 'mixers/' + slug(name) + '_source_weights', 'Mean source-token weights over selected examples; not a saliency estimate.')
    if metadata['modality'] == 'ecg':
        fs = metadata.get('sampling_rate')
        times = np.arange(first_input.shape[-1]) / fs if fs else np.arange(first_input.shape[-1])
        figure, axes = writer.plt.subplots(2, 1, figsize=(10, 5), constrained_layout=True)
        for channel in range(first_input.shape[0]):
            axes[0].plot(times, first_input[channel], label=f'Lead {channel}')
        axes[0].set(xlabel='Time (s)' if fs else 'Sample', ylabel='Model input', title='First selected ECG example')
        axes[0].legend()
        axes[1].plot(range(tokens), first.mean(axis=(0, 1)), marker='.')
        axes[1].set(xlabel='Source time token', ylabel='Mean source weight', title=name + ': same example')
        writer.save(figure, 'mixers/' + slug(name) + '_ecg_mapping', 'ECG example and its own source-token weights, averaged across heads and destination tokens.')
    return stats


def filter_figures(writer, model):
    from agfl.attention.agfl.layer import AGFL
    for name, module in model.named_modules():
        if not isinstance(module, AGFL):
            continue
        coefficients = []
        for head, graph_filter in enumerate(module.filters):
            alpha = graph_filter.alpha_logits.detach()
            mode = graph_filter.options['coefficient_activation']
            if mode == 'softmax':
                alpha = alpha.softmax(-1)
            elif mode == 'sigmoid':
                alpha = alpha.sigmoid()
            elif mode == 'relu':
                alpha = alpha.relu()
            coefficients.append(alpha.cpu().numpy())
            projections = list(graph_filter.W) if graph_filter.projection == 'separate' else ([graph_filter.W] if graph_filter.projection == 'shared' else [])
            if projections:
                figure, axes = writer.plt.subplots(1, len(projections), figsize=(4 * len(projections), 3.5), squeeze=False, constrained_layout=True)
                for index, (axis, projection) in enumerate(zip(axes[0], projections)):
                    weights = projection.weight.detach().cpu().numpy()
                    limit = max(float(np.abs(weights).max()), 1e-12)
                    artist = axis.imshow(weights, cmap='RdBu_r', vmin=-limit, vmax=limit)
                    axis.set(title=f'Head {head}, projection {index}', xlabel='Input feature', ylabel='Output feature')
                    figure.colorbar(artist, ax=axis)
                writer.save(figure, f'filters/{slug(name)}_head_{head}', 'Learned feature-projection weights; their axes are features, not physical electrodes.')
        figure, axis = writer.plt.subplots(figsize=(7, 4), constrained_layout=True)
        for head, alpha in enumerate(coefficients):
            axis.plot(range(len(alpha)), alpha, marker='o', label=f'Head {head}')
        axis.axhline(0, color='black', linewidth=.7)
        axis.set(xlabel='Hop order k', ylabel='Effective coefficient', title=name)
        axis.legend()
        writer.save(figure, 'filters/' + slug(name) + '_coefficients', 'Learned AGFL coefficients after the configured coefficient activation.')


def diagnose_run(run_dir, output_dir, *, device='cpu', partition='validation', max_samples=256,
                 embedding='tsne', data_dir=None, labels_dir=None, data_path=None):
    import torch
    from agfl.datasets import load_dataset, validate_split
    from agfl.datasets.base import source_fingerprint
    from agfl.config import experiment_identity, comparison_identity, experiment_selection
    from agfl.models import get_model_spec
    from agfl.reproducibility import seed_everything, provenance

    directory = Path(run_dir).resolve()
    config = json.loads((directory / 'config.json').read_text())
    split = json.loads((directory / 'split.json').read_text())
    if partition not in {'validation', 'test'} or embedding not in {'pca', 'tsne', 'umap'}:
        raise ValueError('Invalid diagnostic partition or embedding')
    if type(max_samples) is not int or max_samples < 1:
        raise ValueError('max_samples must be a positive integer')
    model_key, attention_key = experiment_selection(config)
    spec = get_model_spec(model_key)
    if config.get('schema_version', 1) < 2:
        if model_key != 'signal_transformer':
            raise ValueError('This original-backbone v1 checkpoint requires its training checkout for diagnostics; analyze can still plot its saved results.')
        # Translate only construction options. Retain the raw saved config for
        # checkpoint validation, identities and scientific provenance.
        from agfl.attention import get_attention_spec
        model_options = {k: v for k, v in config['model_options'].items()
                         if k in spec.defaults_for(config['model_variant'])}
        attention_options = {k: v for k, v in config['model_options'].items()
                             if k in get_attention_spec(attention_key).defaults}
    else:
        model_options, attention_options = config['model_options'], config['attention_options']
    # Saved checkpoints predating spatial readouts must reconstruct their original
    # forward path. Only construction settings change; saved identities stay intact.
    model_options = copy.deepcopy(model_options)
    defaults = spec.defaults_for(config['model_variant'])
    for key, previous in {'spatial_readout': 'mean', 'attention_residual': False}.items():
        if key in defaults and key not in model_options:
            model_options[key] = previous
    checkpoint_path = directory / 'checkpoint.pt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if checkpoint['config'] != config or checkpoint['split_id'] != split['split_id']:
        raise ValueError('Checkpoint/config/split identity mismatch')
    data = copy.deepcopy(config['data'])
    for name, value in [('data_dir', data_dir), ('labels_dir', labels_dir), ('path', data_path)]:
        if value is not None:
            if name not in data:
                raise ValueError(f'The selected dataset does not support {name}')
            data[name] = value
    bundle = load_dataset(config['dataset'], data)
    if bundle.fingerprint != config['dataset_fingerprint'] or checkpoint['dataset_fingerprint'] != bundle.fingerprint:
        raise ValueError('Diagnostic dataset does not match the saved training data')
    validate_split(bundle, split)
    indices = selected_indices(bundle.y, split[partition], max_samples, config['seeds'][0])
    if not len(indices):
        raise ValueError('No examples in the requested partition')
    values = model_inputs(bundle, indices, checkpoint.get('normalization'))
    labels = bundle.y[indices]
    seed_everything(config['seeds'][0], config['deterministic'], config.get('threads', 1))
    current = provenance()
    model = spec.build(model_options, bundle.metadata, attention_key, attention_options)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(device).eval()
    writer = FigureWriter(output_dir)
    info = None
    if bundle.metadata['modality'] == 'eeg':
        try:
            info = eeg_info(bundle.metadata)
        except ValueError as error:
            writer.skip('scalp_maps', error)
    signal_figures(writer, values, labels, bundle.metadata, info)
    train_indices = selected_indices(bundle.y, split['train'], max_samples, config['seeds'][0])
    csp_indices = []
    if info is not None and len(np.unique(bundle.y[train_indices])) >= 2:
        from mne.decoding import CSP
        training = model_inputs(bundle, train_indices, checkpoint.get('normalization')).astype(np.float64)
        try:
            csp = CSP(n_components=min(4, training.shape[1]), reg=.1, log=True, norm_trace=False)
            csp.fit(training, bundle.y[train_indices])
            csp_indices = train_indices.tolist()
            for component, pattern in enumerate(csp.patterns_[:csp.n_components]):
                topomap(writer, pattern, info, f'signals/csp_{component}', f'CSP pattern {component}, fitted on training examples only')
        except (ValueError, np.linalg.LinAlgError) as error:
            writer.skip('signals/csp', error)
    filter_figures(writer, model)
    features, probabilities, maps, hooks = [], [], {}, []
    readout = model.feature_readout if hasattr(model, 'feature_readout') else model.classifier
    hooks.append(readout.register_forward_pre_hook(
        lambda module, args: features.append(args[0].detach().flatten(1).cpu().numpy())))
    def capture(name):
        def hook(module, args, output):
            matrix, kind = mixer_map(module, args[0])
            if matrix is None:
                if not any(entry['name'] == name for entry in writer.skipped):
                    writer.skip(name, kind)
                return
            array = matrix.detach().cpu().numpy()
            statistics = map_statistics(array)
            count = len(array)
            if name not in maps:
                maps[name] = {'sum': np.zeros_like(array[0], dtype=np.float64), 'count': 0,
                              'first': array[0], 'kind': kind, 'token_axis': module.token_axis,
                              'sparsity': np.zeros(array.shape[1]), 'entropy': np.zeros(array.shape[1]),
                              'entropy_valid': np.ones(array.shape[1], dtype=bool)}
            state = maps[name]
            state['sum'] += array.sum(axis=0, dtype=np.float64)
            state['count'] += count
            state['sparsity'] += np.asarray(statistics['sparsity_per_head']) * count
            for head, entropy in enumerate(statistics['entropy_per_head']):
                if entropy is None:
                    state['entropy_valid'][head] = False
                else:
                    state['entropy'][head] += entropy * count
        return hook
    for name, module in model.named_modules():
        if getattr(module, 'is_attention', False):
            hooks.append(module.register_forward_hook(capture(name)))
    try:
        with torch.no_grad():
            for start in range(0, len(values), 16):
                logits = model(torch.from_numpy(values[start:start+16]).to(device))
                probabilities.append(logits.softmax(-1).cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    embedding_figure(writer, np.concatenate(features), labels, embedding, config['seeds'][0])
    statistics = {name: graph_figures(writer, name, state, bundle.metadata, info, values[0]) for name, state in maps.items()}
    write_json(writer.root / 'mixer_statistics.json', statistics)
    saved_provenance = config.get('provenance', {})
    prediction_check = None
    if (directory / 'predictions.npz').is_file():
        with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
            if saved[f'{partition}_ids'].astype(str).tolist() != split['sample_ids'][partition]:
                raise ValueError('Saved prediction IDs differ from the split')
            positions = {index: offset for offset, index in enumerate(split[partition])}
            expected = saved[f'{partition}_probabilities'][[positions[index] for index in indices]]
        actual = np.concatenate(probabilities)
        prediction_check = {'max_absolute_difference': float(np.max(np.abs(actual - expected))),
                            'matches': bool(np.allclose(actual, expected, rtol=1e-4, atol=1e-5)),
                            'rtol': 1e-4, 'atol': 1e-5}
        if not prediction_check['matches']:
            writer.skip('prediction_consistency', 'Reconstructed probabilities differ from the saved run; review recorded source/device/package differences before interpreting diagnostics')
    else:
        writer.skip('prediction_consistency', 'Saved predictions are unavailable')
    return writer.finish({
        'schema_version': 1, 'kind': 'checkpoint_diagnostics', 'run_dir': str(directory),
        'dataset': config['dataset'], 'model': model_key, 'attention': attention_key, 'seed': config['seeds'][0],
        'experiment_id': experiment_identity(config), 'comparison_id': comparison_identity(config),
        'split_id': split['split_id'], 'dataset_fingerprint': bundle.fingerprint,
        'checkpoint': source_fingerprint(checkpoint_path), 'partition': partition, 'max_samples': max_samples,
        'sample_ids': [bundle.sample_ids[i] for i in indices], 'labels': labels.tolist(),
        'csp_training_sample_ids': [bundle.sample_ids[i] for i in csp_indices],
        'sampling': 'Class-balanced diagnostic subset; not the full evaluation cohort',
        'prediction_check': prediction_check,
        'normalization': config['data'].get('normalization'), 'device': device, 'embedding': embedding,
        'training_provenance': saved_provenance, 'diagnostic_provenance': current,
        'source_matches_training': current['source_sha256'] == saved_provenance.get('source_sha256'),
        'note': 'Diagnostics use a strictly loaded checkpoint and fingerprint-matched data, never update saved metrics, and are not causal explanations. Source differences are recorded because post-hoc plotting code may be added after training.',
    })
