"""Read-only checkpoint diagnostics; no optimizer or experiment-metric updates."""
import copy
import json
from pathlib import Path

import numpy as np

from agfl.storage import write_json
from agfl.result_paths import resolve_artifact_run
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


def _base_coefficients(graph_filter):
    """Activated checkpoint parameters, before any trial-dependent adjustment."""
    alpha = graph_filter.alpha_logits.detach()
    mode = graph_filter.options['coefficient_activation']
    if mode == 'softmax':
        return alpha.softmax(-1)
    if mode == 'sigmoid':
        return alpha.sigmoid()
    if mode == 'relu':
        return alpha.relu()
    return alpha


class TrialCoefficientCapture:
    """Capture every selected trial/token from the actual per-head inputs."""

    def __init__(self, model):
        from agfl.attention.agfl.layer import AGFL
        self.layers, self.hooks = {}, []
        self.sample_ids = None
        for name, module in model.named_modules():
            if not isinstance(module, AGFL) or module.options.get('coefficient_conditioning', 'static') == 'static':
                continue
            self.layers[name] = {
                'module': module,
                'coefficients': [[] for _ in module.filters],
                'sample_ids': [[] for _ in module.filters],
                'hop_diagnostics': [[] for _ in module.filters],
            }
            for head, graph_filter in enumerate(module.filters):
                self.hooks.append(graph_filter.register_forward_hook(self._capture(name, head)))

    def start_batch(self, sample_ids):
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        if self.sample_ids.ndim != 1:
            raise ValueError('Coefficient diagnostic sample IDs must be one-dimensional')

    def _capture(self, name, head):
        def hook(graph_filter, args, output):
            import torch
            if self.sample_ids is None:
                raise ValueError('Set diagnostic sample IDs before capturing coefficients')
            # GraphFilter receives (graph, values). Neither the graph nor a
            # pooled trial descriptor alone reconstructs token-local routing.
            values = args[1].detach()
            if len(values) != len(self.sample_ids):
                raise ValueError('Coefficient diagnostic batch and sample IDs differ')
            with torch.no_grad():
                token_local = graph_filter.options.get('coefficient_conditioning', 'static') == 'token_contrast'
                if token_local:
                    from agfl.attention.agfl.layer import propagate
                    hops = tuple(propagate(args[0].detach(), values, graph_filter.options['K'],
                                           graph_filter.options['agfl_variant'],
                                           graph_filter.options['hop_normalization']))
                    effective = graph_filter.coefficient_values(values, first_hop=hops[1])
                    # Inspect actual propagated representations, not powers of
                    # a graph averaged across trials. Norms are descriptive;
                    # cancellation and the later output projection still matter.
                    stacked = torch.stack(hops, dim=-2)
                    hop_norms = torch.linalg.vector_norm(stacked, dim=-1)
                    normalized_hops = stacked / hop_norms.clamp_min(1e-12).unsqueeze(-1)
                    cosines = normalized_hops @ normalized_hops.transpose(-1, -2)
                    valid = hop_norms > 1e-12
                    valid_pairs = valid.unsqueeze(-1) & valid.unsqueeze(-2)
                    projected = torch.stack([
                        graph_filter.W[hop](value) if graph_filter.projection == 'separate' else graph_filter.W(value)
                        for hop, value in enumerate(hops)
                    ], dim=-2)
                    terms = projected * effective.to(projected.dtype).unsqueeze(-1)
                    contribution_norms = torch.linalg.vector_norm(terms, dim=-1)
                    self.layers[name]['hop_diagnostics'][head].append({
                        'propagated_hop_cosines': cosines.cpu().numpy().copy(),
                        'propagated_hop_cosine_valid': valid_pairs.cpu().numpy().copy(),
                        'hop_contribution_norms': contribution_norms.cpu().numpy().copy(),
                    })
                else:
                    effective = graph_filter.coefficient_values(values)
                coefficients = effective.detach().cpu().numpy().copy()
            expected_shape = ((len(values), values.shape[-2], graph_filter.alpha_logits.numel()) if token_local
                              else (len(values), graph_filter.alpha_logits.numel()))
            if coefficients.shape != expected_shape:
                raise ValueError('Unexpected trial coefficient dimensions')
            if not np.isfinite(coefficients).all():
                raise ValueError('Nonfinite trial-conditioned coefficients')
            self.layers[name]['coefficients'][head].append(coefficients)
            self.layers[name]['sample_ids'][head].append(self.sample_ids.copy())
        return hook


def coefficient_conditioning_figures(writer, capture, sample_ids, labels, partition):
    """Save actual coefficients with explicit trial/head/(token)/hop axes."""
    if not capture.layers:
        return None
    sample_ids = np.asarray(sample_ids, dtype=str)
    labels = np.asarray(labels)
    if sample_ids.ndim != 1 or labels.shape != sample_ids.shape or not len(sample_ids):
        raise ValueError('Coefficient diagnostic IDs and labels must identify the same nonempty subset')
    layers = {}
    for name, state in capture.layers.items():
        module = state['module']
        token_local = module.options.get('coefficient_conditioning', 'static') == 'token_contrast'
        per_head = []
        for batches, id_batches in zip(state['coefficients'], state['sample_ids']):
            if not batches or not np.array_equal(np.concatenate(id_batches), sample_ids):
                raise ValueError(f'Trial coefficient order/count differs from diagnostic sample IDs: {name}')
            per_head.append(np.concatenate(batches, axis=0))
        effective = np.stack(per_head, axis=1)
        base = np.stack([_base_coefficients(item).cpu().numpy() for item in module.filters])
        scale = float(module.options.get('coefficient_conditioning_scale', .5))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('Conditioned coefficient diagnostics require a positive finite scale')
        taps = effective.shape[-1]
        base_broadcast = base[None, :, None, :] if token_local else base[None]
        deltas = effective.astype(np.float64) - base_broadcast.astype(np.float64)
        normalized = deltas / scale
        # Centered tanh adjustments have a tighter attainable bound than the
        # configured scale: scale/2 * (tanh(g) - mean_hops(tanh(g))).
        delta_bound = scale * (taps - 1) / taps if token_local else scale
        bound_normalized = deltas / delta_bound if delta_bound > 0 else np.zeros_like(deltas)
        tolerance = max(1e-7, float(np.abs(base).max()) * np.finfo(effective.dtype).eps * taps * 8)
        if not np.isfinite(deltas).all() or np.any(np.abs(deltas) > delta_bound + tolerance):
            raise ValueError('Trial coefficient deltas exceed their configured conditioning bound')
        delta_sum = deltas.sum(axis=-1)
        if token_local and np.any(np.abs(delta_sum) > tolerance):
            raise ValueError('Token coefficient adjustments do not sum to zero across hops')
        gates = np.stack([item.coefficient_gate.weight.detach().cpu().numpy() for item in module.filters])
        if not np.isfinite(gates).all():
            raise ValueError('Nonfinite trial-conditioning gate weights')
        suffix = '_token_coefficients' if token_local else '_trial_coefficients'
        array_path = Path('filters') / (slug(name) + suffix + '.npz')
        (writer.root / array_path).parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            'sample_ids': sample_ids, 'labels': labels, 'base_coefficients': base,
            'effective_coefficients': effective, 'coefficient_deltas': deltas,
            'normalized_deltas': normalized, 'bound_normalized_deltas': bound_normalized,
            'gate_weights': gates, 'head_indices': np.arange(effective.shape[1]),
            'hop_orders': np.arange(taps),
        }
        if token_local:
            arrays.update(token_indices=np.arange(effective.shape[2]),
                          within_trial_token_std=effective.std(axis=2),
                          delta_sum_over_hops=delta_sum)
            hop_batches = state.get('hop_diagnostics', [])
            if hop_batches and all(hop_batches):
                for key in ('propagated_hop_cosines', 'propagated_hop_cosine_valid', 'hop_contribution_norms'):
                    arrays[key] = np.stack([
                        np.concatenate([batch[key] for batch in batches], axis=0)
                        for batches in hop_batches
                    ], axis=1)
                    if arrays[key].shape[:3] != effective.shape[:3] or not np.isfinite(arrays[key]).all():
                        raise ValueError('Invalid token hop diagnostics or trial/token count mismatch')
        np.savez_compressed(writer.root / array_path, **arrays)
        heads = []
        for head in range(effective.shape[1]):
            coefficients, delta = effective[:, head], deltas[:, head]
            reduction_axes = (0, 1) if token_local else 0
            heads.append({
                'head': head,
                'base_coefficients': base[head].tolist(),
                'effective_mean': coefficients.mean(axis=reduction_axes).tolist(),
                'effective_std': coefficients.std(axis=reduction_axes).tolist(),
                'effective_min': coefficients.min(axis=reduction_axes).tolist(),
                'effective_max': coefficients.max(axis=reduction_axes).tolist(),
                'delta_mean': delta.mean(axis=reduction_axes).tolist(),
                'delta_std': delta.std(axis=reduction_axes).tolist(),
                'delta_mean_absolute': np.abs(delta).mean(axis=reduction_axes).tolist(),
                'delta_max_absolute': np.abs(delta).max(axis=reduction_axes).tolist(),
                'saturation_fraction': (np.abs(bound_normalized[:, head]) > .95).mean(axis=reduction_axes).tolist(),
                'gate_frobenius_norm': float(np.linalg.norm(gates[head])),
                'gate_learnable': module.filters[head].coefficient_gate.weight.requires_grad,
            })
            if token_local:
                heads[-1].update({
                    'effective_mean_per_token': coefficients.mean(axis=0).tolist(),
                    'effective_std_per_token': coefficients.std(axis=0).tolist(),
                    'within_trial_token_std_mean': coefficients.std(axis=1).mean(axis=0).tolist(),
                    'within_trial_token_std_max': coefficients.std(axis=1).max(axis=0).tolist(),
                    'delta_sum_over_hops_max_absolute': float(np.abs(delta_sum[:, head]).max()),
                })
                if 'propagated_hop_cosines' in arrays:
                    valid = arrays['propagated_hop_cosine_valid'][:, head]
                    counts = valid.sum(axis=(0, 1))
                    total = (arrays['propagated_hop_cosines'][:, head] * valid).sum(axis=(0, 1))
                    heads[-1].update({
                        'propagated_hop_cosine_mean': [
                            [float(total[left, right] / counts[left, right]) if counts[left, right] else None
                             for right in range(taps)] for left in range(taps)],
                        'propagated_hop_cosine_valid_counts': counts.tolist(),
                        'hop_contribution_norm_mean': arrays['hop_contribution_norms'][:, head].mean(axis=(0, 1)).tolist(),
                    })
        layers[name] = {
            'coefficient_conditioning': module.options.get('coefficient_conditioning', 'static'),
            'coefficient_conditioning_scale': scale,
            'samples': len(sample_ids), 'array_file': array_path.as_posix(),
            'coefficient_array_axes': ['trial', 'head', 'token', 'hop'] if token_local else ['trial', 'head', 'hop'],
            'base_array_axes': ['head', 'hop'],
            'gate_array_axes': ['head', 'hop', 'descriptor_feature'],
            'standard_deviation_ddof': 0, 'saturation_normalized_threshold': .95,
            'attainable_delta_bound': delta_bound,
            'saturation_normalization': 'attainable_delta_bound',
            'summary_reduction_axes': ['trial', 'token'] if token_local else ['trial'],
            'heads': heads,
        }
        if token_local:
            layers[name].update({
                'tokens': effective.shape[2], 'token_axis': getattr(module, 'token_axis', 'unspecified'),
                'within_trial_token_std_axes': ['trial', 'head', 'hop'],
                'delta_sum_over_hops_axes': ['trial', 'head', 'token'],
                'delta_sum_over_hops_max_absolute': float(np.abs(delta_sum).max()),
                'zero_sum_check_tolerance': tolerance,
            })
            if 'propagated_hop_cosines' in arrays:
                layers[name].update({
                    'propagated_hop_cosine_axes': ['trial', 'head', 'token', 'hop_left', 'hop_right'],
                    'hop_contribution_norm_axes': ['trial', 'head', 'token', 'hop'],
                    'hop_diagnostic_note': 'Cosines compare actual propagated hops before optional feature projections. '
                                           'Pairs with a hop norm <= 1e-12 are marked invalid. Contribution norms include '
                                           'hop projections and routed coefficients, before AGFL output projection/residual. '
                                           'They are not causal importance and do not account for cancellation between terms.',
                })
            _token_coefficient_heatmaps(writer, name, effective, bound_normalized, partition)
            continue
        figure, axes = writer.plt.subplots(2, effective.shape[1], squeeze=False,
                                          figsize=(4 * effective.shape[1], 7), constrained_layout=True)
        limit = max(float(np.abs(effective).max()), 1e-12)
        for head in range(effective.shape[1]):
            for row, array, title in (
                    (0, effective[:, head], 'Effective coefficients'),
                    (1, normalized[:, head], 'Delta / conditioning scale')):
                axis = axes[row, head]
                bound = limit if row == 0 else 1.
                artist = axis.imshow(array, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
                axis.set(title=f'Head {head}: {title}', xlabel='Hop order k',
                         ylabel='Diagnostic trial row', xticks=range(array.shape[1]))
                figure.colorbar(artist, ax=axis)
        writer.save(figure, 'filters/' + slug(name) + '_trial_coefficients',
                    f'Actual coefficients and bounded adjustments on the selected {partition} trials. '
                    'Rows follow sample_ids in the accompanying NPZ and diagnostic manifest; these are not causal explanations.')
    summary = {
        'schema_version': 2 if any(layer['coefficient_conditioning'] == 'token_contrast' for layer in layers.values()) else 1,
        'partition': partition, 'sample_ids': sample_ids.tolist(),
        'labels': labels.tolist(), 'layers': layers,
        'note': 'Coefficients are recomputed without gradients from each GraphFilter forward input, '
                'across every diagnostic batch. Deltas are relative to activated checkpoint base coefficients. '
                'normalized_deltas divide by configured scale; bound_normalized_deltas divide by the attainable '
                'per-hop bound (equal to scale for trial_power, scale*(taps-1)/taps for token_contrast). '
                'Saturation means abs(bound_normalized_delta) > 0.95; variation is descriptive only.',
    }
    write_json(writer.root / 'coefficient_conditioning.json', summary)
    return summary


def _token_coefficient_heatmaps(writer, name, effective, normalized, partition):
    """Keep every trial and token visible rather than averaging their routing."""
    limit = max(float(np.abs(effective).max()), 1e-12)
    for head in range(effective.shape[1]):
        figure, axes = writer.plt.subplots(2, effective.shape[-1], squeeze=False,
                                          figsize=(4 * effective.shape[-1], 7), constrained_layout=True)
        for hop in range(effective.shape[-1]):
            for row, array, title in (
                    (0, effective[:, head, :, hop], 'Effective coefficient'),
                    (1, normalized[:, head, :, hop], 'Delta / attainable bound')):
                axis = axes[row, hop]
                bound = limit if row == 0 else 1.
                artist = axis.imshow(array, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
                axis.set(title=f'Head {head}, hop {hop}: {title}', xlabel='Destination token index',
                         ylabel='Diagnostic trial row', xticks=range(array.shape[1]))
                figure.colorbar(artist, ax=axis)
        writer.save(figure, f'filters/{slug(name)}_token_coefficients_head_{head}',
                    f'Actual destination-token coefficients on selected {partition} trials. '
                    'Columns retain every token; rows follow sample_ids in the accompanying NPZ and manifest. '
                    'The second row scales adjustments by their attainable bound, not by the larger configured scale.')


def filter_figures(writer, model):
    from agfl.attention.agfl.layer import AGFL
    statistics = {}
    for name, module in model.named_modules():
        if not isinstance(module, AGFL):
            continue
        conditioning = module.options.get('coefficient_conditioning', 'static')
        coefficients, heads = [], []
        for head, graph_filter in enumerate(module.filters):
            raw_alpha = graph_filter.alpha_logits.detach()
            alpha = _base_coefficients(graph_filter)
            coefficients.append(alpha.cpu().numpy())
            builder = module.builders[head]
            raw_temperature = builder.temperature.detach()
            temperature_active = builder.scaling == 'temperature'
            if not temperature_active:
                clamp_status = 'inactive'
            elif bool(raw_temperature < .1):
                clamp_status = 'below_minimum'
            elif bool(raw_temperature > 5.):
                clamp_status = 'above_maximum'
            elif bool(raw_temperature == .1):
                clamp_status = 'lower_boundary'
            elif bool(raw_temperature == 5.):
                clamp_status = 'upper_boundary'
            else:
                clamp_status = 'interior'
            heads.append({
                'head': head,
                'coefficients_raw': raw_alpha.cpu().tolist(),
                'coefficients_effective': alpha.cpu().tolist() if conditioning == 'static' else None,
                'coefficients_base': alpha.cpu().tolist(),
                'coefficients_learnable': graph_filter.alpha_logits.requires_grad,
                'temperature_raw': float(raw_temperature.cpu()),
                'temperature_effective': float(raw_temperature.clamp(.1, 5.).cpu()) if temperature_active else None,
                'temperature_active': temperature_active,
                'temperature_learnable': builder.temperature.requires_grad,
                'temperature_clamped': clamp_status in {'below_minimum', 'above_maximum'},
                'temperature_clamp_status': clamp_status,
                'projections': [],
            })
            projections = list(graph_filter.W) if graph_filter.projection == 'separate' else ([graph_filter.W] if graph_filter.projection == 'shared' else [])
            if projections:
                figure, axes = writer.plt.subplots(1, len(projections), figsize=(4 * len(projections), 3.5), squeeze=False, constrained_layout=True)
                for index, (axis, projection) in enumerate(zip(axes[0], projections)):
                    weights = projection.weight.detach().cpu().numpy()
                    heads[-1]['projections'].append({
                        'projection_index': index,
                        'hop_order': index if graph_filter.projection == 'separate' else None,
                        'shape': list(weights.shape),
                        'learnable': projection.weight.requires_grad,
                        'frobenius_norm': float(np.linalg.norm(weights)),
                        'distance_from_identity': float(np.linalg.norm(
                            weights - np.eye(*weights.shape, dtype=weights.dtype))),
                    })
                    limit = max(float(np.abs(weights).max()), 1e-12)
                    artist = axis.imshow(weights, cmap='RdBu_r', vmin=-limit, vmax=limit)
                    axis.set(title=f'Head {head}, projection {index}', xlabel='Input feature', ylabel='Output feature')
                    figure.colorbar(artist, ax=axis)
                writer.save(figure, f'filters/{slug(name)}_head_{head}', 'Learned feature-projection weights; their axes are features, not physical electrodes.')
        figure, axis = writer.plt.subplots(figsize=(7, 4), constrained_layout=True)
        for head, alpha in enumerate(coefficients):
            axis.plot(range(len(alpha)), alpha, marker='o', label=f'Head {head}')
        axis.axhline(0, color='black', linewidth=.7)
        axis.set(xlabel='Hop order k', ylabel='Effective coefficient' if conditioning == 'static' else 'Base coefficient', title=name)
        axis.legend()
        caption = 'Learned AGFL coefficients after the configured coefficient activation.'
        if conditioning != 'static':
            caption = ('Learned AGFL base coefficients before trial-dependent adjustments. '
                       'Actual per-trial coefficients are saved in coefficient_conditioning.json and its NPZ arrays.')
        if conditioning == 'token_contrast':
            caption = ('Learned AGFL base coefficients before destination-token adjustments. '
                       'Actual coefficients for every trial and token are saved in coefficient_conditioning.json '
                       'and its NPZ arrays; these base values alone do not describe the routed filter.')
        writer.save(figure, 'filters/' + slug(name) + '_coefficients', caption)
        temperature_init = float(module.options.get('temperature_init', 1.0))
        temperature_active = module.options['score_scaling'] == 'temperature'
        statistics[name] = {
            'coefficient_init': module.options['coefficient_init'],
            'coefficient_activation': module.options['coefficient_activation'],
            'coefficient_conditioning': conditioning,
            'coefficient_conditioning_scale': float(module.options.get('coefficient_conditioning_scale', .5)) if conditioning != 'static' else None,
            'coefficient_scope': ('sample_independent' if conditioning == 'static' else
                                  'base_before_token_adjustment' if conditioning == 'token_contrast' else
                                  'base_before_trial_adjustment'),
            'agfl_variant': module.options['agfl_variant'],
            'filter_projection': module.options['filter_projection'],
            'filter_projection_init': module.options.get('filter_projection_init', 'pytorch'),
            'score_scaling': module.options['score_scaling'],
            'temperature_init': temperature_init,
            'temperature_active': temperature_active,
            'temperature_clamp_bounds': [.1, 5.] if temperature_active else None,
            'heads': heads,
            'note': 'Learned values are read from the loaded checkpoint; initialization settings come from the saved configuration. Inactive temperatures do not scale scores; their effective value is null. Clamp status distinguishes exact boundaries from values outside the interval.',
        }
        if temperature_active:
            figure, axis = writer.plt.subplots(figsize=(7, 4), constrained_layout=True)
            indices = [head['head'] for head in heads]
            axis.plot(indices, [head['temperature_raw'] for head in heads], marker='o', label='Checkpoint parameter')
            axis.plot(indices, [head['temperature_effective'] for head in heads], marker='x', label='Effective after clamp')
            axis.axhline(temperature_init, color='black', linestyle=':', linewidth=1., label='Configured initialization')
            axis.set(xticks=indices, xlabel='Head', ylabel='Temperature', title=name)
            axis.legend()
            writer.save(figure, 'filters/' + slug(name) + '_temperature',
                        'Per-head score temperatures from the loaded checkpoint. Effective values are clamped to [0.1, 5]. With softmax normalization, lower values sharpen probabilities at fixed scores.')
    if statistics:
        write_json(writer.root / 'filter_statistics.json', statistics)
    return statistics


def diagnose_run(run_dir, output_dir, *, device='cpu', partition='validation', max_samples=256,
                 embedding='tsne', data_dir=None, labels_dir=None, data_path=None):
    # Resolve report copies before importing model/data dependencies, so a
    # report-only download receives an actionable missing-checkpoint message.
    directory = resolve_artifact_run(run_dir)
    import torch
    from agfl.datasets import load_dataset, validate_split
    from agfl.datasets.base import source_fingerprint
    from agfl.config import experiment_identity, comparison_identity, experiment_selection
    from agfl.models import get_model_spec
    from agfl.reproducibility import seed_everything, provenance

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
    filter_statistics = filter_figures(writer, model)
    coefficient_capture = TrialCoefficientCapture(model)
    from .temporal_statistics import TemporalStatisticsCapture, temporal_statistics_figures
    temporal_capture = TemporalStatisticsCapture(model)
    sample_ids = [bundle.sample_ids[i] for i in indices]
    features, probabilities, maps, hooks = [], [], {}, [*coefficient_capture.hooks, *temporal_capture.hooks]
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
                coefficient_capture.start_batch(sample_ids[start:start+16])
                temporal_capture.start_batch(sample_ids[start:start+16])
                logits = model(torch.from_numpy(values[start:start+16]).to(device))
                probabilities.append(logits.softmax(-1).cpu().numpy())
    finally:
        for hook in hooks:
            hook.remove()
    conditioning_statistics = coefficient_conditioning_figures(
        writer, coefficient_capture, sample_ids, labels, partition)
    temporal_statistics = temporal_statistics_figures(
        writer, temporal_capture, sample_ids, labels,
        bundle.metadata.get('label_names') or [f'Class {i}' for i in range(bundle.metadata['num_classes'])], partition)
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
        'sample_ids': sample_ids, 'labels': labels.tolist(),
        'csp_training_sample_ids': [bundle.sample_ids[i] for i in csp_indices],
        'sampling': 'Class-balanced diagnostic subset; not the full evaluation cohort',
        'prediction_check': prediction_check,
        'filter_statistics_file': 'filter_statistics.json' if filter_statistics else None,
        'coefficient_conditioning_file': 'coefficient_conditioning.json' if conditioning_statistics else None,
        'temporal_statistics_file': 'temporal_statistics.json' if temporal_statistics else None,
        'checkpoint_policy': checkpoint.get('checkpoint_policy', {'weights': 'raw'}),
        'normalization': config['data'].get('normalization'), 'device': device, 'embedding': embedding,
        'training_provenance': saved_provenance, 'diagnostic_provenance': current,
        'source_matches_training': current['source_sha256'] == saved_provenance.get('source_sha256'),
        'note': 'Diagnostics use a strictly loaded checkpoint and fingerprint-matched data, never update saved metrics, and are not causal explanations. Source differences are recorded because post-hoc plotting code may be added after training.',
    })
