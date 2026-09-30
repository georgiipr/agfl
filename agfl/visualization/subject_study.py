"""Subject means and contrasts for a declared attention experiment matrix."""
from textwrap import fill
import numpy as np
from .common import slug

METRICS = ('accuracy', 'roc_auc', 'f1')
TITLES = ('Accuracy', 'ROC-AUC', 'Macro F1')


def plot_subject_studies(writer, studies):
    for study in studies:
        prefix = 'subject_study/' + slug(study['name'])
        arms, subjects = study['plan']['arms'], study['plan']['subjects']
        labels = {r['arm']: fill(r['attention_label'], 30) for r in study['averages']}
        status = ('Complete' if study['complete'] else 'INCOMPLETE') + f" / {study.get('token_axis', 'unspecified')} graph"
        rows = {(r['arm'], r['subject'], r['metric']): r for r in study['per_subject'] if r['partition'] == 'test'}
        for metric, title in zip(METRICS, TITLES):
            matrix = np.array([[rows[(a, s, metric)]['mean'] if rows[(a, s, metric)]['mean'] is not None else np.nan
                                for s in subjects] for a in arms])
            figure, axis = writer.plt.subplots(figsize=(max(10, len(subjects) + 3), max(5, len(arms) * .95)), constrained_layout=True)
            artist = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, aspect='auto', cmap='viridis')
            for i in range(len(arms)):
                for j in range(len(subjects)):
                    value = matrix[i, j]
                    axis.text(j, i, 'missing' if np.isnan(value) else f'{value * 100:.1f}', ha='center', va='center',
                              fontsize=9, color='black' if np.isnan(value) or value > .65 else 'white')
            axis.set(xticks=range(len(subjects)), xticklabels=subjects, yticks=range(len(arms)),
                     yticklabels=[labels[a] for a in arms], xlabel='Individually trained subject',
                     title=f'{status}: test {title.lower()} (%) / mean of {len(study["plan"]["seeds"])} seeds')
            figure.colorbar(artist, ax=axis, label='Score (0–1)')
            writer.save(figure, f'{prefix}/subjects_{metric}',
                        'Each cell averages the declared seeds for one subject and attention. Missing, incomplete or protocol-mismatched cells are marked missing; no trial pooling.')
        figure, axes = writer.plt.subplots(1, 3, figsize=(15, max(5, len(arms) * .95)), sharey=True, constrained_layout=True)
        averages = {(r['arm'], r['metric']): r for r in study['averages'] if r['partition'] == 'test'}
        for axis, metric, title in zip(axes, METRICS, TITLES):
            for index, arm in enumerate(arms):
                row = averages[(arm, metric)]
                values = [rows[(arm, s, metric)]['mean'] for s in subjects]
                values = [v for v in values if v is not None]
                if values:
                    axis.scatter(values, index + np.linspace(-.12, .12, len(values)), s=20, alpha=.7)
                if row['mean'] is not None:
                    axis.errorbar(row['mean'], index, xerr=row['between_subject_sd'], fmt='ks', capsize=4)
                axis.text(1.01, index, f"{row['n_subjects']}/{len(subjects)}", va='center', fontsize=8)
            axis.set(xlim=(0, 1.13), ylim=(len(arms) - .5, -.5), yticks=range(len(arms)),
                     xlabel='Test score', title=title)
            axis.grid(axis='x', alpha=.2)
        axes[0].set_yticklabels([labels[a] for a in arms])
        figure.suptitle(f'{status}: equal-weight subject means / {study["completed_runs"]}/{study["expected_runs"]} fits')
        writer.save(figure, prefix + '/averaged_metrics',
                    'Dots are subject means over seeds. Squares average all declared subjects equally; bars are between-subject sample SD, not confidence intervals. No overall square is shown for incomplete coverage.')
        for metric, title in zip(METRICS, TITLES):
            comparisons = [r for r in study['comparisons'] if r['metric'] == metric]
            if not comparisons:
                continue
            figure, axis = writer.plt.subplots(figsize=(12, max(5, len(comparisons) * .9)), constrained_layout=True)
            for index, row in enumerate(comparisons):
                values = [v * 100 for v in row['subject_deltas'].values()]
                axis.scatter(values, index + np.linspace(-.13, .13, len(values)), s=24)
                if row['mean_delta'] is not None:
                    axis.plot(row['mean_delta'] * 100, index, 'kD', markersize=7)
            axis.axvline(0, color='black', lw=.8)
            axis.set(yticks=range(len(comparisons)), ylim=(len(comparisons) - .5, -.5),
                     yticklabels=[fill(r['candidate_label'] + ' − ' + r['baseline_label'], 42) for r in comparisons],
                     xlabel=f'Test {title.lower()} difference (percentage points)',
                     title=f'{status}: subject-paired differences')
            axis.grid(axis='x', alpha=.2)
            writer.save(figure, f'{prefix}/paired_subjects_{metric}',
                        'Dots are differences of matched subject means. Diamonds are overall contrasts only when the full declared experiment matrix is complete. Subjects, not seeds, are the units in subject_study_comparisons.csv.')
