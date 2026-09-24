"""Saved diagnostics for EEGNet's explicit pre-attention log-variance branch."""
from pathlib import Path

import numpy as np

from agfl.storage import write_json


class TemporalStatisticsCapture:
    def __init__(self, model):
        self.module = getattr(model, 'temporal_statistics', None)
        self.sample_ids = None
        self.ids, self.log_variances, self.normalized, self.adjustments = [], [], [], []
        self.hooks = [] if self.module is None else [self.module.register_forward_hook(self._capture)]

    def start_batch(self, sample_ids):
        self.sample_ids = np.asarray(sample_ids, dtype=str)
        if self.sample_ids.ndim != 1 or not len(self.sample_ids):
            raise ValueError('Temporal statistics require a nonempty batch of sample IDs')

    def _capture(self, module, args, output):
        import torch
        if self.sample_ids is None:
            raise ValueError('Set temporal-statistic sample IDs before each forward batch')
        values = args[0].detach()
        batch = len(self.sample_ids)
        if values.ndim != 4 or len(values) % batch:
            raise ValueError('Temporal statistics and trial IDs have incompatible shapes')

        def pack(tensor):
            # Electrode-token EEGNet flattens [trial,electrode] into its batch;
            # time-token EEGNet uses one spatial site per trial after filtering.
            array = tensor.detach().to(dtype=torch.float32).cpu().numpy()
            if not np.isfinite(array).all():
                raise ValueError('Nonfinite temporal-statistic diagnostic features')
            sites = len(array) // batch
            return array.reshape(batch, sites, *array.shape[1:]).transpose(0, 1, 3, 2, 4).reshape(
                batch, sites * array.shape[2], array.shape[1], array.shape[3]).copy()

        with torch.no_grad():
            self.log_variances.append(pack(module.log_variance(values)))
            self.normalized.append(pack(module.normalized_statistics(values)))
            self.adjustments.append(pack(output))
        self.ids.append(self.sample_ids.copy())


def temporal_statistics_figures(writer, capture, sample_ids, labels, class_names, partition):
    if capture.module is None:
        return None
    sample_ids, labels = np.asarray(sample_ids, dtype=str), np.asarray(labels)
    if (sample_ids.ndim != 1 or labels.shape != sample_ids.shape or not len(sample_ids)
            or not capture.ids or not np.array_equal(np.concatenate(capture.ids), sample_ids)):
        raise ValueError('Temporal-statistic trial IDs/order differ from the diagnostic subset')
    arrays = [np.concatenate(values, axis=0) for values in (
        capture.log_variances, capture.normalized, capture.adjustments)]
    if any(len(values) != len(sample_ids) for values in arrays):
        raise ValueError('Temporal-statistic capture omitted or duplicated trials')
    logvar, normalized, adjustments = arrays
    module = capture.module
    weights = module.projection.weight.detach().float().cpu().numpy()
    if not all(np.isfinite(values).all() for values in (*arrays, weights)):
        raise ValueError('Nonfinite temporal-statistic arrays or projection weights')
    relative = Path('features/temporal_statistics.npz')
    (writer.root / relative).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(writer.root / relative, sample_ids=sample_ids, labels=labels,
                        log_variance=logvar, normalized_log_variance=normalized,
                        additive_features=adjustments, projection_weights=weights,
                        bin_edges=np.asarray(module.bin_edges))
    rms = np.sqrt(np.square(adjustments.astype(np.float64)).mean(axis=(1, 2, 3)))
    classes = sorted(set(labels.tolist()))
    class_summary = []
    for index in classes:
        selected = labels == index
        class_summary.append({
            'class_id': int(index), 'class_name': class_names[index], 'samples': int(selected.sum()),
            'additive_rms_mean': float(rms[selected].mean()),
            'additive_rms_std': float(rms[selected].std()),
        })
    summary = {
        'schema_version': 1, 'partition': partition, 'samples': len(sample_ids),
        'sample_ids': sample_ids.tolist(), 'labels': labels.tolist(), 'array_file': relative.as_posix(),
        'array_axes': ['trial', 'spatial_site', 'feature', 'time_bin'],
        'bin_edges': list(module.bin_edges), 'variance_ddof': 0,
        'projection_frobenius_norm': float(np.linalg.norm(weights)),
        'additive_rms_mean': float(rms.mean()), 'additive_rms_max': float(rms.max()),
        'by_class': class_summary,
        'note': 'Actual spatial Conv+BatchNorm features before ELU, pooling and dropout. '
                'Uniform bins cover the complete trial and differ from the original floor-pooling bins. '
                'Class averages describe latent features, not causal effects or EEG frequency-band attribution.',
    }
    write_json(writer.root / 'temporal_statistics.json', summary)
    figure, axes = writer.plt.subplots(1, len(classes), squeeze=False,
                                       figsize=(4 * len(classes), 5), constrained_layout=True)
    limit = max(float(np.abs(normalized).max()), 1e-12)
    for axis, index in zip(axes[0], classes):
        means = normalized[labels == index].mean(axis=(0, 1))
        artist = axis.imshow(means, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        axis.set(title=class_names[index], xlabel='Time bin', ylabel='Learned spatial feature')
        figure.colorbar(artist, ax=axis)
    writer.save(figure, 'features/temporal_statistics_classes',
                f'Mean normalized log-variance features on selected {partition} trials; '
                'latent feature axes are not scalp locations or known frequency bands.')
    figure, axis = writer.plt.subplots(figsize=(6, 5), constrained_layout=True)
    bound = max(float(np.abs(weights).max()), 1e-12)
    artist = axis.imshow(weights, aspect='auto', cmap='RdBu_r', vmin=-bound, vmax=bound)
    axis.set(xlabel='Log-variance feature', ylabel='Feature entering attention',
             title='Learned temporal-statistic projection')
    figure.colorbar(artist, ax=axis)
    writer.save(figure, 'features/temporal_statistics_projection',
                'Projection starts at zero. Its learned values and additive feature RMS are recorded '
                'in temporal_statistics.json; activity alone does not establish an accuracy benefit.')
    return summary
