"""Plot saved raw-audit measurements without recordings, checkpoints or inference."""
import base64
import html
import json
from pathlib import Path

import numpy as np

from agfl.datasets.base import source_fingerprint
from agfl.eeg_audit import CLASSES, report_text
from agfl.storage import run_directory
from .common import FigureWriter


def plot_eeg_audit(audit_dir):
    root = Path(audit_dir).expanduser().resolve()
    with run_directory(root):
        if (root/'failure.json').exists():
            raise ValueError('The data audit failed; complete audit-eeg before plotting')
        audit = json.loads((root/'audit.json').read_text())
        if audit.get('kind') != 'bci2a_raw_audit' or audit.get('schema_version') != 1:
            raise ValueError('Expected a supported audit-eeg output directory')
        (root/'report.html').unlink(missing_ok=True)
        for name in ('index.md', 'manifest.json'):
            (root/'figures'/name).unlink(missing_ok=True)
        writer = FigureWriter(root/'figures')
        _inventory_figures(writer, audit)
        _quality_figures(writer, audit)
        manifest = writer.finish({'kind': 'bci2a_raw_audit_figures', 'subject_id': audit['subject_id'],
                                  'audit_file': source_fingerprint(root/'audit.json'),
                                  'plot_source': source_fingerprint(Path(__file__)),
                                  'quality_partitions': ['train', 'validation']})
        # All previews are embedded; the HTML can be downloaded/shared as one file.
        body = ['<!doctype html><html lang="en"><meta charset="utf-8">',
                '<title>BCI IV 2a raw-data audit</title>',
                '<style>body{max-width:1150px;margin:32px auto;padding:0 20px;font-family:system-ui;'
                'line-height:1.5}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}'
                'img{width:100%;height:auto}figure{margin:32px 0}figcaption{margin:12px 0}</style>',
                '<body><h1>BCI IV 2a raw-data audit</h1>',
                '<pre>' + html.escape(report_text(audit)) + '</pre>']
        for figure in manifest['figures']:
            data = base64.b64encode((root/'figures'/figure['files'][0]).read_bytes()).decode('ascii')
            caption = html.escape(figure['caption'])
            body.append(f'<figure><img alt="{caption}" src="data:image/png;base64,{data}">'
                        f'<figcaption>{caption}</figcaption></figure>')
        if manifest['skipped']:
            body.append('<h2>Unavailable figures</h2><ul>')
            body.extend('<li>' + html.escape(f"{s['name']}: {s['reason']}") + '</li>' for s in manifest['skipped'])
            body.append('</ul>')
        body.append('</body></html>')
        temporary = root/'report.html.tmp'
        temporary.write_text('\n'.join(body))
        temporary.replace(root/'report.html')
    return manifest


def _inventory_figures(writer, audit):
    rows = audit['trials']
    if not rows:
        writer.skip('trial_inventory', 'No cue events were found')
        return
    runs = sorted({r['run_id'] for r in rows}, key=lambda s: int(s.rsplit(':', 1)[1]))
    labels = [s.rsplit(':', 1)[1] for s in runs]
    figure, axes = writer.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, categories, key, title in [
        (axes[0], CLASSES, 'class_name', 'All cue events by run and class'),
        (axes[1], ['train', 'validation', 'test', 'excluded'], 'partition', 'Saved assignments by run'),
    ]:
        bottom = np.zeros(len(runs))
        for category in categories:
            values = np.asarray([sum(r['run_id'] == run and r[key] == category for r in rows) for run in runs])
            axis.bar(labels, values, bottom=bottom, label=category)
            bottom += values
        axis.set(title=title, xlabel='Recording run ID (includes preceding calibration blocks)', ylabel='Trials')
        axis.legend(fontsize=8)
    writer.save(figure, 'trial_inventory', 'Annotation counts and saved assignments; test counts are integrity checks only.')
    figure, axes = writer.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot([r['cue_index'] for r in rows], [r['cue_delay_seconds'] for r in rows], '.', ms=3)
    axes[0].axhline(2, color='black', ls='--', label='Expected 2 seconds')
    axes[0].set(xlabel='Original cue index', ylabel='Cue minus trial onset (s)', title='Cue timing')
    axes[0].legend()
    total = np.asarray([sum(r['run_id'] == run for r in rows) for run in runs])
    marked = np.asarray([sum(r['run_id'] == run and r['artifact_marked'] for r in rows) for run in runs])
    axes[1].bar(labels, 100 * marked / total)
    axes[1].set(xlabel='Recording run ID', ylabel='Artifact-marked trials (%)', title='Original artifact annotations')
    writer.save(figure, 'timing_artifacts', 'Event timing and recorded artifact flags; no additional rejection was applied.')


def _quality_figures(writer, audit):
    quality = [r for r in audit['signals']['trials'] if r['status'] == 'measured']
    if not quality:
        writer.skip('signal_quality', 'No finite training/validation windows could be measured')
        return
    channels = audit['channel_names']
    grouped = {part: [r for r in quality if r['partition'] == part] for part in ('train', 'validation')}
    for part, records in grouped.items():
        if not records:
            writer.skip(part + '_signal_measurements', 'No usable windows; this partition has no signal summaries or plot values')
    figure, axes = writer.plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    for part, records in grouped.items():
        if not records:
            continue
        for axis, key, quantile in [(axes[0], 'raw_std_uv', .5), (axes[1], 'raw_p2p_uv', .95)]:
            axis.plot(channels, np.quantile([r[key] for r in records], quantile, axis=0), '.-', label=part)
    for axis, title in zip(axes, ('Median within-trial EEG SD (µV)', '95th percentile within-trial EEG peak-to-peak (µV)')):
        axis.set(ylabel='µV', title=title)
        axis.legend()
    writer.save(figure, 'channel_amplitudes', 'Calibrated raw EEG channel amplitudes, before normalization; percentiles are descriptive, not rejection thresholds.')
    figure, axes = writer.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for part, records in grouped.items():
        if records:
            axes[0].scatter([r['cue_index'] for r in records], [max(r['raw_p2p_uv']) for r in records], s=12, label=part)
            maxima = []
            for r in records:
                correlations = np.abs(np.asarray(r['eeg_eog_correlation'], dtype=float))
                maxima.append(float(np.nanmax(correlations)) if np.isfinite(correlations).any() else np.nan)
            axes[1].scatter([r['cue_index'] for r in records], maxima, s=12, label=part)
    for run in {r['run_id'] for r in audit['trials']}:
        start = min(r['cue_index'] for r in audit['trials'] if r['run_id'] == run)
        for axis in axes:
            axis.axvline(start, color='grey', alpha=.25)
    axes[0].set(xlabel='Original cue index', ylabel='Maximum EEG peak-to-peak (µV)', title='Amplitude across task runs')
    axes[1].set(xlabel='Original cue index', ylabel='Maximum |EEG/EOG Pearson r|', title='EEG/EOG coupling', ylim=(0, 1.02))
    for axis in axes:
        axis.legend()
    writer.save(figure, 'trial_quality', 'Training/validation trial quality in acquisition order; grey lines mark task runs. Correlation alone does not establish contamination.')
    figure, axes = writer.plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    spectra = audit['signals']['spectra']
    frequency = np.asarray(spectra['frequency_hz'])
    for axis, part in zip(axes, ('train', 'validation')):
        data = spectra['partitions'].get(part)
        if data:
            for key, label in [('raw_psd_uv2_hz', 'Raw'), ('filtered_psd_uv2_hz', 'Saved bandpass')]:
                axis.semilogy(frequency, np.maximum(np.asarray(data[key]).mean(axis=0), 1e-20), label=label)
            axis.legend()
        axis.set(title=part, xlabel='Frequency (Hz)', ylabel='Mean EEG PSD (µV²/Hz)', xlim=(0, min(100, audit['sampling_rate']/2)))
    writer.save(figure, 'raw_filtered_spectra', 'Mean Welch spectra across measured trials and EEG channels, in physical units. No train/validation normalization was applied.')
    figure, axes = writer.plt.subplots(1, 2, figsize=(10, 7), constrained_layout=True)
    for axis, (part, records) in zip(axes, grouped.items()):
        if records:
            matrix = np.asarray([r['eeg_eog_correlation'] for r in records], dtype=float)
            # Summarize each pair without warnings for a missing/constant EOG channel.
            median = np.full((22, 3), np.nan)
            for i in range(22):
                for j in range(3):
                    values = np.abs(matrix[:, i, j])
                    if np.isfinite(values).any():
                        median[i, j] = np.median(values[np.isfinite(values)])
            artist = axis.imshow(np.ma.masked_invalid(median), vmin=0, vmax=1, aspect='auto')
            figure.colorbar(artist, ax=axis, label='Median |Pearson r|')
        axis.set(title=part, yticks=range(22), yticklabels=channels, xticks=range(3), xticklabels=['EOG 1', 'EOG 2', 'EOG 3'])
    writer.save(figure, 'eeg_eog_correlations', 'Median absolute within-trial correlations of raw EEG and EOG. Blank cells are undefined, including constant channels.')
    figure, axes = writer.plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for column, band in enumerate(('alpha', 'beta')):
        matrices = []
        for part, records in grouped.items():
            matrix = np.full((4, 3), np.nan)
            for label in range(4):
                members = [r[band + '_power_uv2'] for r in records if r['class_index'] == label]
                if members:
                    matrix[label] = np.median(np.asarray(members)[:, [7, 9, 11]], axis=0)
            matrices.append(matrix)
        finite = np.concatenate([m[np.isfinite(m)] for m in matrices])
        maximum = max(float(finite.max()), 1e-12) if len(finite) else 1.
        for axis, part, matrix in zip(axes[:, column], grouped, matrices):
            artist = axis.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=maximum, aspect='auto')
            axis.set(title=f'{part}: {band}', yticks=range(4), yticklabels=CLASSES,
                     xticks=range(3), xticklabels=['C3', 'Cz', 'C4'])
            figure.colorbar(artist, ax=axis, label='Median power (µV²)')
    writer.save(figure, 'class_band_power', 'Filtered alpha (8–13 Hz) and beta (13–30 Hz) power; shared train/validation scales per band. These are not baseline-relative ERD maps.')
    examples = audit['signals']['examples']
    if examples:
        figure, axes = writer.plt.subplots(len(examples), 2, figsize=(13, 3*len(examples)), squeeze=False, constrained_layout=True)
        for pair, example in zip(axes, examples):
            for axis, key, labels in [(pair[0], 'eeg_uv', ['C3', 'Cz', 'C4']), (pair[1], 'eog_uv', ['EOG 1', 'EOG 2', 'EOG 3'])]:
                for signal, label in zip(example[key], labels):
                    axis.plot(example['time_seconds'], signal, lw=.7, label=label)
                axis.set(title=f"{example['class_name']} — {example['sample_id']}", xlabel='Seconds after cue', ylabel='µV')
                axis.legend(fontsize=8)
        writer.save(figure, 'raw_examples', 'First chronological measured training trial per class: calibrated raw C3/Cz/C4 and EOG, without amplitude rescaling or normalization.')
    else:
        writer.skip('raw_examples', 'No training examples with known classes were measured')
