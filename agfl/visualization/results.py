"""Figures from saved histories, probabilities and validated seed aggregates."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc

from agfl.metrics import classification_metrics
from agfl.config import experiment_selection


def selection_label(run):
    model, attention = experiment_selection(run['config'])
    subject = run['config'].get('subject_id')
    return f'{model} / {attention}' + (f' / {subject}' if subject else '')

from .common import FigureWriter, slug

METRICS = ('accuracy', 'roc_auc', 'f1')
TITLES = ('Accuracy', 'ROC-AUC', 'Macro F1')


def read_predictions(run, partition):
    directory = Path(run['source']).parent
    with np.load(directory / 'predictions.npz', allow_pickle=False) as saved:
        y = saved[f'{partition}_targets']
        probabilities = saved[f'{partition}_probabilities']
        ids = saved[f'{partition}_ids'].astype(str)
    if y.ndim != 1 or ids.ndim != 1 or len(ids) != len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError(f'Prediction labels and IDs must be aligned one-dimensional arrays: {directory}')
    declared_classes = run['config'].get('resolved_metadata', {}).get('num_classes')
    if probabilities.ndim != 2 or (declared_classes is not None and probabilities.shape[1] != declared_classes):
        raise ValueError(f'Prediction class dimension differs from the saved configuration: {directory}')
    split = json.loads((directory / 'split.json').read_text())
    if split['split_id'] != run['split_id'] or split['fingerprint'] != run['dataset_fingerprint']:
        raise ValueError(f'Split identity differs from result: {directory}')
    if ids.tolist() != split['sample_ids'][partition]:
        raise ValueError(f'Prediction sample order differs from split: {directory}')
    metrics = classification_metrics(y, probabilities)
    for metric in METRICS:
        expected, actual = run[partition].get(metric), metrics[metric]
        if (expected is None) != (actual is None) or (expected is not None and not np.isclose(expected, actual, rtol=1e-7, atol=1e-9)):
            raise ValueError(f'Saved {partition} predictions disagree with {metric}: {directory}')
    return y, probabilities


def class_names(run, count):
    names = run['config'].get('resolved_metadata', {}).get('label_names')
    return list(names) if names and len(names) == count else [f'Class {i}' for i in range(count)]


def plot_history(writer, run, prefix):
    path = Path(run['source']).with_name('history.json')
    if not path.is_file():
        writer.skip(prefix + '/training', 'history.json unavailable')
        return
    history = json.loads(path.read_text())
    if not history:
        writer.skip(prefix + '/training', 'Empty training history')
        return
    epochs = [row['epoch'] for row in history]
    figure, axes = writer.plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='Train')
    axes[0].plot(epochs, [r['validation']['loss'] for r in history], label='Validation')
    axes[0].set(title='Loss', ylabel='Loss')
    for metric, title in zip(METRICS, TITLES):
        axes[1].plot(epochs, [r['validation'].get(metric) if r['validation'].get(metric) is not None else np.nan for r in history], label=title)
    if all('train_accuracy' in row for row in history):
        axes[1].plot(epochs, [r['train_accuracy'] for r in history], linestyle='--', label='Training accuracy (train mode)')
    axes[1].set(title='Validation metrics / training accuracy', ylabel='Score', ylim=(0, 1))
    axes[2].plot(epochs, [r['learning_rate'] for r in history])
    axes[2].set(title='Learning rate', ylabel='Learning rate')
    for axis in axes:
        axis.set_xlabel('Epoch')
        axis.grid(alpha=.2)
        if run.get('best_checkpoint_epoch') is not None:
            axis.axvline(run['best_checkpoint_epoch'], color='black', linestyle=':', label='Selected checkpoint')
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.suptitle(f"{run['dataset']} / {selection_label(run)} / seed {run['seed']}")
    writer.save(figure, prefix + '/training', 'Training loss, validation metrics and LR; test data do not enter epoch selection.')


def plot_predictions(writer, run, partition, prefix):
    directory = Path(run['source']).parent
    if not all((directory / name).is_file() for name in ('predictions.npz', 'split.json')):
        writer.skip(prefix + '/' + partition, 'Saved predictions or split manifest unavailable')
        return
    y, probabilities = read_predictions(run, partition)
    count = probabilities.shape[1]
    names = class_names(run, count)
    matrix = confusion_matrix(y, probabilities.argmax(-1), labels=np.arange(count))
    totals = matrix.sum(axis=1, keepdims=True)
    rates = np.divide(matrix, totals, out=np.zeros_like(matrix, dtype=float), where=totals > 0)
    figure, axes = writer.plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, values, title in zip(axes, (matrix, rates), ('Counts', 'Row-normalized')):
        artist = axis.imshow(values, cmap='Blues', vmin=0, vmax=None if title == 'Counts' else 1)
        for row in range(count):
            for column in range(count):
                value = str(matrix[row, column]) if title == 'Counts' else (f'{rates[row, column]:.2f}' if totals[row, 0] else 'N/A')
                axis.text(column, row, value, ha='center', va='center', fontsize=8,
                          color='white' if values[row, column] > values.max() * .55 else 'black')
        axis.set(xticks=np.arange(count), yticks=np.arange(count), xticklabels=names, yticklabels=names,
                 xlabel='Predicted', ylabel='True', title=title)
        axis.tick_params(axis='x', rotation=35)
        figure.colorbar(artist, ax=axis)
    figure.suptitle(f"{partition}: {selection_label(run)}, seed {run['seed']}")
    writer.save(figure, f'{prefix}/{partition}_confusion', f'{partition.capitalize()} confusion matrices; absent true classes are marked N/A.')
    figure, axis = writer.plt.subplots(figsize=(6, 5), constrained_layout=True)
    available = 0
    for index, name in enumerate(names):
        target = y == index
        if target.all() or not target.any():
            writer.skip(f'{prefix}/{partition}_roc/{name}', 'ROC requires positive and negative samples')
            continue
        fpr, tpr, _ = roc_curve(target, probabilities[:, index])
        axis.plot(fpr, tpr, label=f'{name}: AUC={auc(fpr, tpr):.3f}')
        available += 1
    if available:
        axis.plot([0, 1], [0, 1], 'k--', linewidth=.8)
        axis.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1),
                 title=f"{partition} one-vs-rest ROC / seed {run['seed']}")
        axis.legend(fontsize=8)
        writer.save(figure, f'{prefix}/{partition}_roc', 'ROC curves from this seed only; repeated test subjects are not pooled across seeds.')
    else:
        writer.plt.close(figure)


def plot_comparison(writer, experiments, runs_by_experiment, prefix, title):
    figure, axes = writer.plt.subplots(1, 3, figsize=(max(12, len(experiments) * 1.7), 4.8), constrained_layout=True)
    for axis, metric, metric_title in zip(axes, METRICS, TITLES):
        for position, experiment in enumerate(experiments):
            values = [r['test'].get(metric) for r in runs_by_experiment[experiment['experiment_id']]]
            values = [value for value in values if value is not None]
            summary = experiment['metrics']['test_' + metric]
            if values:
                axis.scatter(position + np.linspace(-.12, .12, len(values)), values, s=18, alpha=.7)
                axis.errorbar(position, summary['mean'], yerr=summary['std'], fmt='ks', capsize=5)
            axis.text(position, 1.02, f'n={len(values)}', ha='center', fontsize=8)
        axis.set(xticks=range(len(experiments)), xticklabels=[f"{e['model']} / {e['attention']}\n{e.get('subject_id') or 'cohort'} / {e['experiment_id'][:8]}" for e in experiments],
                 ylim=(0, 1.1), title=metric_title, ylabel='Test score')
        axis.tick_params(axis='x', rotation=30)
        axis.grid(axis='y', alpha=.2)
    figure.suptitle(title)
    writer.save(figure, prefix, 'Points are seeds; black squares and bars show mean and sample SD, not confidence intervals. IDs map to aggregation.json.')


def paired_differences(agfl_runs, baseline_runs, metric):
    def key(run):
        return run['seed'], run['split_id'], run['dataset_fingerprint'], run['comparison_id']
    a = {key(run): run for run in agfl_runs}
    b = {key(run): run for run in baseline_runs}
    return [(k[0], a[k]['test'][metric] - b[k]['test'][metric]) for k in sorted(a.keys() & b.keys())
            if a[k]['test'].get(metric) is not None and b[k]['test'].get(metric) is not None]


def plot_diagnostic_comparisons(writer, diagnostics_root, runs_by_experiment):
    cohorts = defaultdict(list)
    seen = set()
    available = {(run['experiment_id'], run['seed'], run['split_id'], run['dataset_fingerprint']): run
                 for runs in runs_by_experiment.values() for run in runs}
    for path in sorted(Path(diagnostics_root).rglob('manifest.json')):
        manifest = json.loads(path.read_text())
        if manifest.get('kind') != 'checkpoint_diagnostics':
            continue
        key = tuple(manifest.get(name) for name in ('experiment_id', 'seed', 'split_id', 'dataset_fingerprint'))
        if key not in available or not (manifest.get('prediction_check') or {}).get('matches'):
            writer.skip(path, 'No matching result with verified reconstructed predictions')
            continue
        statistics = json.loads(path.with_name('mixer_statistics.json').read_text())
        if not statistics:
            continue
        sparsity = np.mean([value for layer in statistics.values() for value in layer['sparsity_per_head']])
        entropies = [value for layer in statistics.values() for value in layer['entropy_per_head']]
        entropy = float(np.mean(entropies)) if entropies and all(value is not None for value in entropies) else None
        run = available[key]
        cohort = (run['comparison_id'], manifest['partition'], manifest['max_samples'])
        if (cohort, key) in seen:
            writer.skip(path, 'Duplicate diagnostic for this run/partition/sample limit')
            continue
        seen.add((cohort, key))
        cohorts[cohort].append((run, float(sparsity), entropy))
    for (family, partition, limit), records in cohorts.items():
        figure, axes = writer.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for run, sparsity, entropy in records:
            for axis, value in zip(axes, (sparsity, entropy)):
                if value is not None:
                    axis.scatter(value, run['test']['accuracy'], s=22)
                    axis.annotate(f"{selection_label(run)} s{run['seed']}", (value, run['test']['accuracy']),
                                  xytext=(3, 3), textcoords='offset points', fontsize=7)
        axes[0].set(xlabel='Mean map sparsity (|weight| ≤ 1e-8)', xlim=(-.02, 1.02))
        axes[1].set(xlabel='Mean probability-row entropy (nats)')
        for axis in axes:
            axis.set(ylabel='Saved test accuracy', ylim=(0, 1))
        writer.save(figure, f'diagnostic_comparisons/{slug(family)}_{partition}_{limit}',
                    f'Descriptive graph statistics on up to {limit} class-balanced {partition} samples versus saved test accuracy. This is not evidence of a causal effect; AGFL A and baseline effective maps have different meanings.')


def generate_result_plots(aggregation, output_dir, diagnostics_root=None):
    writer = FigureWriter(output_dir)
    by_experiment, families = defaultdict(list), defaultdict(list)
    for source in aggregation['sources']:
        run = json.loads(Path(source).read_text())
        run['source'] = source
        by_experiment[run['experiment_id']].append(run)
        prefix = f"runs/{slug(run['experiment_id'])}/seed_{run['seed']}"
        plot_history(writer, run, prefix)
        for partition in ('validation', 'test'):
            plot_predictions(writer, run, partition, prefix)
        subjects = run.get('test_by_subject', {})
        if subjects:
            figure, axes = writer.plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            for axis, metric, title in zip(axes, METRICS, TITLES):
                axis.bar(list(subjects), [values.get(metric) if values.get(metric) is not None else np.nan for values in subjects.values()])
                axis.set(title=title, ylim=(0, 1), ylabel='Test score')
                axis.tick_params(axis='x', rotation=45)
            writer.save(figure, prefix + '/test_subjects', 'Per-subject test scores within this seed; undefined metrics are omitted.')
    for experiment in aggregation['experiments']:
        families[experiment['comparison_id']].append(experiment)
    for family, experiments in sorted(families.items()):
        prefix = f'comparisons/{slug(family)}'
        title = f"{experiments[0]['dataset']} / {experiments[0]['model']}"
        plot_comparison(writer, experiments, by_experiment, prefix + '/metrics', title)
        ablations = [e for e in experiments if e['attention'] == 'agfl']
        if len(ablations) > 1:
            plot_comparison(writer, ablations, by_experiment, prefix + '/agfl_ablations', title + ' / AGFL ablations')
        figure, axes = writer.plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, metric, label in zip(axes, METRICS, TITLES):
            for experiment in experiments:
                stats = experiment['metrics']['test_' + metric]
                if stats['mean'] is not None:
                    axis.errorbar(experiment['parameter_count'], stats['mean'], yerr=stats['std'], fmt='o', capsize=3)
                    axis.annotate(f"{experiment['attention']} {experiment['experiment_id'][:6]}",
                                  (experiment['parameter_count'], stats['mean']), xytext=(3, 5), textcoords='offset points', fontsize=7)
            axis.set(xlabel='Parameters', ylabel=label, ylim=(0, 1))
        writer.save(figure, prefix + '/capacity', 'Parameter counts versus mean test scores; bars are sample SD across seeds.')
    for comparison in aggregation['comparisons']:
        metric = comparison['metric']
        pairs = paired_differences(by_experiment[comparison['agfl_experiment_id']], by_experiment[comparison['baseline_experiment_id']], metric)
        if not pairs:
            continue
        figure, axis = writer.plt.subplots(figsize=(7, 4), constrained_layout=True)
        axis.axhline(0, color='black', linewidth=.8)
        axis.plot([str(seed) for seed, _ in pairs], [delta for _, delta in pairs], 'o')
        axis.set(xlabel='Matched seed', ylabel=f'AGFL − {comparison["baseline"]}: {metric}',
                 title=f"Paired test differences (n={len(pairs)})")
        name = f"paired/{comparison['agfl_experiment_id']}_{comparison['baseline_experiment_id']}_{metric}"
        writer.save(figure, name, 'Only identical seed/split/data/protocol pairs are shown. Seed-level inference remains exploratory.')
    if diagnostics_root is not None:
        plot_diagnostic_comparisons(writer, diagnostics_root, by_experiment)
    return writer.finish({'schema_version': 1, 'kind': 'saved_results', 'sources': aggregation['sources'],
                          'diagnostics_root': str(Path(diagnostics_root).resolve()) if diagnostics_root is not None else None,
                          'methodology': aggregation['methodology']})
