"""Electrode graph figures with explicit sensor identities and edge direction."""
import csv
import numpy as np

from .common import slug


def electrode_graph_figures(writer, name, mean, class_means, class_counts, metadata, info):
    channels = metadata['channels']
    names = metadata.get('channel_names') or [f'Electrode {i + 1}' for i in range(channels)]
    if len(names) != channels or mean.shape[-2:] != (channels, channels):
        raise ValueError('Electrode maps must match the recording channel order and count')
    prefix = 'mixers/' + slug(name)
    with (writer.root / (prefix + '_electrode_edges.csv')).open('w', newline='') as stream:
        out = csv.writer(stream)
        out.writerow(['head', 'source_electrode', 'destination_electrode', 'mean_weight'])
        for head, matrix in enumerate(mean):
            for destination in range(channels):
                for source in range(channels):
                    out.writerow([head, names[source], names[destination], float(matrix[destination, source])])

    for label, matrix in sorted(class_means.items()):
        heads = matrix.shape[0]
        figure, axes = writer.plt.subplots(1, heads, figsize=(5 * heads, 4.5), squeeze=False, constrained_layout=True)
        bound = max(float(np.abs(matrix).max()), 1e-12)
        signed = bool((matrix < 0).any())
        for head, axis in enumerate(axes[0]):
            artist = axis.imshow(matrix[head], cmap='RdBu_r' if signed else 'viridis',
                                 vmin=-bound if signed else 0., vmax=bound)
            axis.set(title=f'Class {label}, head {head}, n={class_counts[label]}',
                     xlabel='Source electrode', ylabel='Destination electrode',
                     xticks=range(channels), yticks=range(channels),
                     xticklabels=names, yticklabels=names)
            axis.tick_params(axis='x', rotation=90, labelsize=6)
            axis.tick_params(axis='y', labelsize=6)
            figure.colorbar(artist, ax=axis, shrink=.75)
        writer.save(figure, prefix + f'_class_{label}_electrodes',
                    'Class-conditional routing maps from the selected diagnostic partition. '
                    'Rows receive information from columns. These are model weights, not brain connectivity.')

    if info is None:
        writer.skip(prefix + '_electrode_connections', 'Scalp connections require known electrode coordinates; labeled matrices and all edge weights remain available')
        return
    positions = np.asarray([channel['loc'][:2] for channel in info['chs']])
    if positions.shape != (channels, 2) or not np.isfinite(positions).all():
        raise ValueError('Scalp coordinates must follow the same electrode order as the graph')
    weights = mean.mean(axis=0)
    edges = [(destination, source) for destination in range(channels) for source in range(channels)
             if destination != source and abs(weights[destination, source]) > 1e-8]
    edges.sort(key=lambda edge: abs(weights[edge]), reverse=True)
    shown = edges[:channels]
    maximum = max([abs(weights[edge]) for edge in shown], default=1.)
    figure, axis = writer.plt.subplots(figsize=(7, 6), constrained_layout=True)
    for destination, source in shown:
        weight = weights[destination, source]
        axis.annotate('', xy=positions[destination], xytext=positions[source],
                      arrowprops={'arrowstyle': '->', 'color': 'tab:blue' if weight >= 0 else 'tab:red',
                                  'alpha': .65, 'lw': .5 + 2 * abs(weight) / maximum,
                                  'connectionstyle': 'arc3,rad=.12', 'shrinkA': 6, 'shrinkB': 6})
    axis.scatter(*positions.T, s=45, color='black', zorder=3)
    for label, position in zip(names, positions):
        axis.annotate(label, position, xytext=(3, 4), textcoords='offset points', fontsize=8)
    axis.set(aspect='equal', xlabel='Head x coordinate (m)', ylabel='Head y coordinate (m)',
             title=f'{name}: strongest {len(shown)} off-diagonal routing edges')
    axis.margins(.2)
    writer.save(figure, prefix + '_electrode_connections',
                'Head x/y projection of standard electrode positions. Arrows point source to destination; '
                'blue/red indicate positive/negative head-averaged weights. Only the strongest C '
                'off-diagonal edges are displayed for legibility; the complete directed matrices '
                'are saved in CSV/NPZ. Display selection is not learned sparsity or anatomical connectivity.')
