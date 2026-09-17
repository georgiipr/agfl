"""BCI IV 2a T-session audit; no checkpoint loading or model fitting."""
from bisect import bisect_right
from collections import Counter
from copy import deepcopy
import csv
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from .datasets.base import bandpass_finite_spans, source_fingerprint
from .datasets.eeg import CHANNEL_NAMES
from .storage import run_directory, write_json


REFERENCE = 'https://www.bbci.de/competition/iv/desc_2a.pdf'
CLASSES = ['left_hand', 'right_hand', 'feet', 'tongue']
PARTITIONS = ('train', 'validation', 'test')


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def trial_inventory(events, samples, fs, basename, settings):
    """Interpret original integer event codes, independently of auto event IDs."""
    events = sorted((int(position), int(code)) for position, code in events)
    starts = sorted({p for p, code in events if code == 768})
    runs = sorted({p for p, code in events if code == 32766})
    artifacts = [p for p, code in events if code == 1023]
    cues = [(p, code) for p, code in events if code in {769, 770, 771, 772, 783}]
    offset = int(round(settings['offset_seconds'] * fs))
    rows = []
    for number, (position, code) in enumerate(cues):
        trial = bisect_right(starts, position) - 1
        run = bisect_right(runs, position) - 1
        trial_start = starts[trial] if trial >= 0 else None
        run_start = runs[run] if run >= 0 else 0
        trial_end = starts[trial + 1] if trial + 1 < len(starts) else samples
        run_end = runs[run + 1] if run + 1 < len(runs) else samples
        assigned = trial_start is not None and trial_start >= run_start
        marked = assigned and any(trial_start <= p < min(trial_end, run_end) for p in artifacts)
        start, stop = position + offset, position + offset + settings['window']
        bounded = assigned and 0 <= start < stop <= min(trial_end, run_end, samples)
        reason = ('unassigned_trial' if not assigned else
                  'artifact' if marked and settings['artifact_policy'] == 'exclude' else
                  'out_of_bounds' if not bounded else None)
        label = code - 769 if code != 783 else None
        rows.append({
            'sample_id': f'{basename}:cue:{number:03d}:sample:{position}',
            'cue_index': number, 'cue_code': code, 'cue_sample': position,
            'class_index': label, 'class_name': CLASSES[label] if label is not None else 'unknown',
            'run_id': f'{basename}:run:{run}', 'trial_start': trial_start,
            'cue_delay_seconds': (position - trial_start) / fs if assigned else None,
            'window_start': start, 'window_stop': stop, 'window_within_trial_run': bool(bounded),
            'artifact_marked': bool(marked), 'structural_exclusion': reason,
        })
    return rows


def saved_assignments(split):
    """Reject corrupt IDs before using them to choose any diagnostic examples."""
    assignments, indices = {}, []
    for part in PARTITIONS:
        ids, positions = split['sample_ids'][part], split[part]
        if not ids or len(ids) != len(positions):
            raise ValueError(f'Saved {part} IDs and indices must be nonempty and aligned')
        if any(type(i) is not int or i < 0 for i in positions):
            raise ValueError('Saved indices must be nonnegative integers')
        indices.extend(positions)
        for sample_id in ids:
            if not isinstance(sample_id, str) or not sample_id or sample_id in assignments:
                raise ValueError('Saved sample IDs are invalid, duplicated or overlap partitions')
            assignments[sample_id] = part
    if sorted(indices) != list(range(len(indices))):
        raise ValueError('Saved indices must form a disjoint, complete dataset partition')
    return assignments


def channel_correlations(eeg, eog):
    """Pearson correlations; a constant/nonfinite channel is undefined, not zero."""
    a, b = eeg - eeg.mean(axis=1, keepdims=True), eog - eog.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(a, axis=1)[:, None] * np.linalg.norm(b, axis=1)[None, :]
    out = np.full((len(a), len(b)), np.nan)
    np.divide(a @ b.T, denominator, out=out, where=np.isfinite(denominator) & (denominator > 0))
    return np.clip(out, -1, 1)


def signal_quality(values, rows, fs, settings):
    """Quality summaries use saved training/validation windows only, never test."""
    from scipy.integrate import trapezoid
    from scipy.signal import welch
    from tqdm import tqdm

    quality, examples, spectra = [], {}, {}
    selected = [r for r in rows if r['partition'] in {'train', 'validation'}]
    for row in tqdm(selected, desc='EEG raw-data audit', unit='trial', dynamic_ncols=True,
                    mininterval=1., disable=None):
        if not row['window_within_trial_run']:
            continue
        voltage = values[:25, row['window_start']:row['window_stop']] * 1e6
        eeg, eog = voltage[:22], voltage[22:25]
        item = {'sample_id': row['sample_id'], 'partition': row['partition'],
                'run_id': row['run_id'], 'class_index': row['class_index'],
                'cue_index': row['cue_index'],
                'nonfinite_fraction': (~np.isfinite(voltage)).mean(axis=1).tolist()}
        if not np.isfinite(eeg).all():
            item['status'] = 'nonfinite_eeg'
            quality.append(item)
            continue
        filtered, valid = bandpass_finite_spans(eeg, fs, settings['lowcut'], settings['highcut'],
                                               gdf_missing=True)
        item.update(raw_std_uv=eeg.std(axis=1), raw_p2p_uv=np.ptp(eeg, axis=1),
                    repeated_sample_fraction=(np.diff(eeg, axis=1) == 0).mean(axis=1),
                    constant_eeg_channels=np.flatnonzero(np.ptp(eeg, axis=1) == 0).tolist(),
                    eeg_eog_correlation=channel_correlations(eeg, eog))
        if not valid.all():
            item['status'] = 'unusable_filtered_eeg'
            quality.append(item)
            continue
        freq, raw_psd = welch(eeg, fs=fs, nperseg=min(int(round(fs)), eeg.shape[-1]), axis=-1)
        _, filtered_psd = welch(filtered, fs=fs, nperseg=min(int(round(fs)), eeg.shape[-1]), axis=-1)
        item.update(status='measured', filtered_std_uv=filtered.std(axis=1))
        for name, low, high in [('alpha', 8, 13), ('beta', 13, 30)]:
            mask = (freq >= low) & (freq <= high)
            item[name + '_power_uv2'] = trapezoid(filtered_psd[:, mask], freq[mask], axis=-1)
        part = row['partition']
        if part not in spectra:
            spectra[part] = {'n': 0, 'raw': np.zeros_like(raw_psd), 'filtered': np.zeros_like(filtered_psd)}
        spectra[part]['n'] += 1
        spectra[part]['raw'] += raw_psd
        spectra[part]['filtered'] += filtered_psd
        label = row['class_index']
        if part == 'train' and label is not None and label not in examples:
            # The first chronological training trial of each class, not an accuracy-selected example.
            examples[label] = {'sample_id': row['sample_id'], 'class_name': CLASSES[label],
                               'time_seconds': np.arange(eeg.shape[1]) / fs + settings['offset_seconds'],
                               'eeg_uv': eeg[[7, 9, 11]], 'eog_uv': eog}
        quality.append(item)
    spectral_result = {'frequency_hz': [], 'partitions': {}}
    for part, data in spectra.items():
        spectral_result['frequency_hz'] = freq
        spectral_result['partitions'][part] = {'n': data['n'], 'raw_psd_uv2_hz': data['raw'] / data['n'],
                                              'filtered_psd_uv2_hz': data['filtered'] / data['n']}
    return _jsonable({'trials': quality, 'spectra': spectral_result, 'examples': list(examples.values())})


def _check(checks, name, passed, detail):
    checks.append({'name': name, 'status': 'pass' if passed else 'fail', 'detail': str(detail)})


def _collect(run, config, split, data_dir):
    import mne
    from .datasets import load_dataset, validate_split

    settings = deepcopy(config['data'])
    if (config['dataset'] != 'eeg' or settings['sessions'] != ['T'] or
            len(settings['subjects']) != 1 or settings['filter_scope'] != 'trial'):
        raise ValueError('audit-eeg requires one subject, T session only, and trial-local filtering')
    assignments = saved_assignments(split)
    if data_dir is not None:
        settings['data_dir'] = str(Path(data_dir).expanduser().resolve())
    basename = f"A{settings['subjects'][0]:02d}T"
    source = Path(settings['data_dir']).expanduser() / (basename + '.gdf')
    actual = source_fingerprint(source)
    expected = next((s for s in config['resolved_metadata']['sources'] if s['name'] == source.name), None)
    if actual != expected:
        raise ValueError(f'{source.name} does not match the recording hash in the saved run; use the original file')
    checks = []
    _check(checks, 'recording_identity', True, f"{actual['name']} SHA256 {actual['sha256']}")
    raw = mne.io.read_raw_gdf(str(source), preload=True, verbose='ERROR')
    try:
        fs, samples = float(raw.info['sfreq']), int(raw.n_times)
        # int() avoids the loader's auto-generated annotation-ID mapping.
        events, _ = mne.events_from_annotations(raw, event_id=lambda s: int(s) if s.isdigit() else None,
                                               regexp=None, use_rounding=True, verbose='ERROR')
        coded = [(int(e[0]) - int(raw.first_samp), int(e[2])) for e in events]
        rows = trial_inventory(coded, samples, fs, basename, settings)
        for row in rows:
            row['partition'] = assignments.get(row['sample_id'], 'excluded')
        by_id = {r['sample_id']: r for r in rows}
        _check(checks, 'sampling_rate', fs == 250., f'{fs:g} Hz; expected 250 Hz')
        _check(checks, 'recording_channels', len(raw.ch_names) == 25,
               f'{len(raw.ch_names)} channels; expected 22 EEG followed by 3 EOG')
        _check(checks, 'cue_count', len(rows) == 288, f'{len(rows)} cue events; expected 288')
        _check(checks, 'known_T_labels', all(r['class_index'] is not None for r in rows),
               'Training cues should be 769–772; unknown 783 cues are not inferred')
        task_runs = sorted({r['run_id'] for r in rows}, key=lambda s: int(s.rsplit(':', 1)[1]))
        run_counts = {name: [sum(r['run_id'] == name and r['class_index'] == c for r in rows)
                             for c in range(4)] for name in task_runs}
        _check(checks, 'task_run_counts', len(task_runs) == 6 and all(c == [12]*4 for c in run_counts.values()),
               f'Expected six task runs with 12 cues per class each; observed {run_counts}')
        delays = [r['cue_delay_seconds'] for r in rows]
        _check(checks, 'cue_timing', bool(rows) and all(d is not None and abs(d - 2.) <= 1/fs for d in delays),
               'Cue expected 2 seconds after trial onset, within one sample')
        trial_counts = Counter(r['trial_start'] for r in rows)
        starts = {p for p, code in coded if code == 768}
        _check(checks, 'one_cue_per_trial', set(trial_counts) == starts and all(n == 1 for n in trial_counts.values()),
               f'{len(starts)} trial starts and {len(rows)} cues')
        _check(checks, 'imagery_window', settings['offset_seconds'] >= 0 and
               settings['offset_seconds'] + settings['window']/fs <= 4. + 1/fs,
               f"Saved interval: {settings['offset_seconds']:g} to {settings['offset_seconds'] + settings['window']/fs:g} seconds after cue; imagery ends at 4 seconds")
        missing = sorted(set(assignments) - by_id.keys())
        invalid = [r['sample_id'] for r in rows if r['partition'] != 'excluded' and r['structural_exclusion']]
        _check(checks, 'saved_trial_membership', not missing and not invalid,
               f'Missing cue IDs: {missing}; saved trials rejected by structural checks: {invalid}')
        marked = sum(r['artifact_marked'] for r in rows)
        _check(checks, 'artifact_count', marked == config['resolved_metadata']['artifact_trials_seen'],
               f"Observed {marked}; saved {config['resolved_metadata']['artifact_trials_seen']}")
        for part in PARTITIONS:
            counts = [sum(r['partition'] == part and r['class_index'] == c for r in rows) for c in range(4)]
            _check(checks, part + '_class_counts', counts == split['class_counts'][part],
                   f"Observed {counts}; saved {split['class_counts'][part]}")
        if (run / 'predictions.npz').is_file():
            with np.load(run / 'predictions.npz', allow_pickle=False) as predictions:
                ids = predictions['validation_ids'].tolist()
                labels = predictions['validation_targets'].tolist()
                matches = (ids == split['sample_ids']['validation'] and len(ids) == len(labels) and
                           all(i in by_id and by_id[i]['class_index'] == label for i, label in zip(ids, labels)))
            _check(checks, 'validation_label_alignment', matches,
                   'Original cue labels compared with saved validation targets; test predictions were not read')
        else:
            checks.append({'name': 'validation_label_alignment', 'status': 'unavailable',
                           'detail': 'No predictions.npz in this run; validation targets cannot be cross-checked'})
        loader_error = None
        try:
            bundle = load_dataset('eeg', settings)
            fingerprint = bundle.fingerprint
            _check(checks, 'preprocessing_identity', fingerprint == config['dataset_fingerprint'],
                   f"Saved {config['dataset_fingerprint']}; reloaded {fingerprint}")
            validate_split(bundle, split)
            _check(checks, 'loader_split_alignment', True, 'Loaded trial order and saved split indices/IDs agree')
            mismatches = [i for i, label in zip(bundle.sample_ids, bundle.y)
                          if i not in by_id or by_id[i]['class_index'] != int(label)]
            _check(checks, 'loader_cue_labels', not mismatches, f'Cue-to-loader label discrepancies: {mismatches}')
            _check(checks, 'retained_trial_ids', set(bundle.sample_ids) == set(assignments),
                   'Loader retained IDs compared with the union of saved partitions')
            retained = set(bundle.sample_ids)
            for row in rows:
                row['loader_retained'] = row['sample_id'] in retained
            del bundle
        except Exception as error:
            loader_error = f'{type(error).__name__}: {error}'
            _check(checks, 'loader_reconstruction', False, loader_error)
        # Loading full recordings and reconstructing fingerprints is identity QA.
        # Signal summaries, spectra, examples and correlations omit the test partition.
        signals = signal_quality(raw.get_data(), rows, fs, settings) if len(raw.ch_names) == 25 else {
            'trials': [], 'spectra': {'frequency_hz': [], 'partitions': {}}, 'examples': []}
        durations = {}
        for code in ('768', '769', '770', '771', '772', '783', '1023'):
            d = np.asarray(raw.annotations.duration)[np.asarray(raw.annotations.description) == code]
            if len(d):
                durations[code] = {'count': len(d), 'min_seconds': float(d.min()), 'max_seconds': float(d.max())}
        observed_channels = list(raw.ch_names)
        observed_types = raw.get_channel_types()
    finally:
        raw.close()
    measured = [r for r in signals['trials'] if r['status'] == 'measured']
    _check(checks, 'finite_quality_windows', len(measured) == sum(r['partition'] in {'train', 'validation'} for r in rows),
           f'{len(measured)} usable training/validation windows in raw signal summaries')
    inspected = [r for r in signals['trials'] if 'constant_eeg_channels' in r]
    if inspected:
        constants = [r['sample_id'] for r in inspected if r['constant_eeg_channels']]
        _check(checks, 'nonconstant_eeg_channels', not constants, f'Exact constant EEG channels in trials: {constants}')
    else:
        checks.append({'name': 'nonconstant_eeg_channels', 'status': 'unavailable',
                       'detail': 'No finite raw EEG windows were available'})
    if signals['trials']:
        bad_eog = [r['sample_id'] for r in signals['trials'] if any(x > 0 for x in r['nonfinite_fraction'][22:25])]
        _check(checks, 'finite_eog_windows', not bad_eog, f'Nonfinite raw EOG values in trials: {bad_eog}')
    else:
        checks.append({'name': 'finite_eog_windows', 'status': 'unavailable',
                       'detail': 'No raw EOG windows were available'})
    return _jsonable({
        'schema_version': 1, 'kind': 'bci2a_raw_audit', 'subject_id': f"A{settings['subjects'][0]:02d}",
        'seed': split['seed'], 'run_dir': str(run), 'split_id': split['split_id'],
        'source': actual, 'saved_config_file': source_fingerprint(run/'config.json'),
        'saved_split_file': source_fingerprint(run/'split.json'), 'preprocessing': settings,
        'packages': {name: importlib.metadata.version(name) for name in ('mne', 'numpy', 'scipy')},
        'training_packages': config.get('provenance', {}).get('packages', {}),
        'audit_source': source_fingerprint(Path(__file__)),
        'reference': REFERENCE, 'sampling_rate': fs, 'channel_names': CHANNEL_NAMES,
        'observed_channel_names': observed_channels, 'observed_channel_types': observed_types,
        'event_counts': dict(Counter(str(c) for _, c in coded)),
        'annotation_durations': durations, 'checks': checks, 'trials': rows, 'signals': signals,
        'quality_partitions': ['train', 'validation'], 'loader_error': loader_error,
    })


def report_text(audit):
    rows = audit['trials']
    settings = audit['preprocessing']
    failed = [c for c in audit['checks'] if c['status'] == 'fail']
    lines = [f"# {audit['subject_id']} raw-data audit, seed {audit['seed']}", '',
             f"Structural/identity checks: **{len(failed)} failed**, "
             f"{sum(c['status'] == 'unavailable' for c in audit['checks'])} unavailable.", '',
             'This is a data-quality audit, not an accuracy evaluation. No model was fitted or loaded.',
             'Signal measurements use training and validation trials only. Test IDs/class counts are checked '
             'for integrity; test predictions and test signal summaries are not used.', '',
             f"Recording: `{audit['source']['name']}`; SHA256 `{audit['source']['sha256']}`.", '',
             f"Saved preprocessing: {settings['lowcut']:g}–{settings['highcut']:g} Hz, "
             f"{settings['window']} samples from {settings['offset_seconds']:g} seconds after cue; "
             f"normalization `{settings['normalization']}`, artifact policy `{settings['artifact_policy']}`.", '',
             '| Partition | Left | Right | Feet | Tongue |', '|---|---:|---:|---:|---:|']
    for part in (*PARTITIONS, 'excluded'):
        counts = [sum(r['partition'] == part and r['class_index'] == c for r in rows) for c in range(4)]
        lines.append('| ' + ' | '.join([part, *map(str, counts)]) + ' |')
    lines += ['', '## Checks', '']
    for check in audit['checks']:
        lines.append(f"- **{check['status'].upper()} {check['name']}**: {check['detail']}")
    lines += ['', '## Signal measurements', '',
              'Raw amplitudes are in microvolts after GDF calibration, before filtering or normalization. '
              'Filtered spectra use the saved trial-local bandpass, before normalization. '
              'EEG/EOG correlations are descriptive; high correlation alone does not prove an artifact. '
              'Class power summaries are not baseline-relative ERD or evidence of classification accuracy.', '',
              '| Partition | Measured trials | Median trial maximum EEG peak-to-peak (µV) |',
              '|---|---:|---:|']
    for part in ('train', 'validation'):
        q = [r for r in audit['signals']['trials'] if r['partition'] == part and r['status'] == 'measured']
        value = f"{np.median([max(r['raw_p2p_uv']) for r in q]):.3f}" if q else 'unavailable'
        lines.append(f'| {part} | {len(q)} | {value} |')
    lines += ['', '## Interpretation', '',
              'Resolve failed identity, label or timing checks before changing a model. '
              'Amplitude differences across runs and EEG/EOG correlations motivate further inspection; '
              'they do not automatically justify excluding trials or applying a correction. '
              'No trial, label, split or preprocessing setting was changed by this audit.', '',
              'The event check uses integer-coded annotations separately from the loader mapping, but '
              'both use MNE and the same GDF. Agreement is not independent ground truth for annotations. '
              'Reconstruction differences can reflect a package or source change; see the recorded versions.', '',
              f'Protocol expectations: [official BCI IV 2a description]({REFERENCE}).', '',
              'Raw tables and numerical plot data: `audit.json` and `trials.csv`.',
              'Generate figures and the single-file HTML report with:', '', '```bash',
              'python main.py plot-eeg-audit PATH_TO_THIS_AUDIT_DIRECTORY', '```', '']
    return '\n'.join(lines)


def audit_eeg_run(run_dir, output_dir, *, data_dir=None):
    run, output = Path(run_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve()
    if output == run or output in run.parents:
        raise ValueError('Use a separate output directory for the EEG audit')
    config = json.loads((run/'config.json').read_text())
    split = json.loads((run/'split.json').read_text())
    identity = {'config': source_fingerprint(run/'config.json'), 'split': source_fingerprint(run/'split.json')}
    with run_directory(output):
        plan = output/'audit_plan.json'
        if plan.exists() and json.loads(plan.read_text()) != identity:
            raise ValueError('This audit directory belongs to a different saved run; use a new --output-dir')
        if not plan.exists() and any(p.name != '.run.lock' for p in output.iterdir()):
            raise ValueError('Use an empty output directory or the matching existing EEG audit directory')
        write_json(plan, identity)
        # Invalidate old completion markers/figures before a repeat, preserving unrelated files.
        for name in ('audit.json', 'report.md', 'report.html', 'failure.json',
                     'figures/index.md', 'figures/manifest.json'):
            (output/name).unlink(missing_ok=True)
        try:
            audit = _collect(run, config, split, data_dir)
            with (output/'trials.csv').open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audit['trials'][0]) if audit['trials'] else ['sample_id'])
                writer.writeheader()
                writer.writerows(audit['trials'])
            (output/'report.md').write_text(report_text(audit))
            write_json(output/'audit.json', audit)
        except BaseException as error:
            write_json(output/'failure.json', {'type': type(error).__name__, 'message': str(error)})
            raise
    return audit
