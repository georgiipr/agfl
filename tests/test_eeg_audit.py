"""Run on the target machine: integrity and held-out-data boundaries for raw audits."""
from copy import deepcopy
import json

import numpy as np
import pytest

from agfl.eeg_audit import (audit_eeg_run, channel_correlations, saved_assignments,
                            signal_quality, trial_inventory)


def test_inventory_keeps_original_cue_ids_and_checks_run_boundary():
    settings = {'offset_seconds': 0., 'window': 1000, 'artifact_policy': 'exclude'}
    events = [(0, 32766), (0, 768), (0, 1023), (500, 769),
              (2000, 768), (2500, 770), (3000, 32766)]
    rows = trial_inventory(events, 4500, 250, 'A06T', settings)
    assert rows[0]['structural_exclusion'] == 'artifact'
    assert rows[0]['cue_delay_seconds'] == 2.
    assert rows[1]['sample_id'] == 'A06T:cue:001:sample:2500'
    assert rows[1]['class_index'] == 1
    assert rows[1]['structural_exclusion'] == 'out_of_bounds'


def test_saved_assignments_reject_overlap_and_incomplete_indices():
    split = {'sample_ids': {'train': ['a'], 'validation': ['b'], 'test': ['c']},
             'train': [0], 'validation': [1], 'test': [2]}
    assert saved_assignments(split)['c'] == 'test'
    overlap = deepcopy(split)
    overlap['sample_ids']['test'] = ['a']
    with pytest.raises(ValueError, match='overlap'):
        saved_assignments(overlap)
    split['test'] = [3]
    with pytest.raises(ValueError, match='complete'):
        saved_assignments(split)


def test_eog_correlations_do_not_turn_constant_channels_into_zero():
    wave = np.sin(np.arange(1000) / 10)
    eeg = np.stack([wave, -wave])
    eog = np.stack([2 * wave, np.ones(1000)])
    correlations = channel_correlations(eeg, eog)
    np.testing.assert_allclose(correlations[:, 0], [1, -1])
    assert np.isnan(correlations[:, 1]).all()


def test_quality_excludes_test_windows_even_when_nonfinite():
    values = np.tile(np.sin(np.arange(3000) / 7), (25, 1)) * 1e-5
    values[:, 2000:] = np.nan
    rows = [{'sample_id': part, 'partition': part, 'run_id': 'A06T:run:3',
             'class_index': i, 'cue_index': i, 'window_start': i*1000,
             'window_stop': (i+1)*1000, 'window_within_trial_run': True}
            for i, part in enumerate(('train', 'validation', 'test'))]
    quality = signal_quality(values, rows, 250., {'lowcut': 2., 'highcut': 30., 'offset_seconds': 0.})
    assert {r['sample_id'] for r in quality['trials']} == {'train', 'validation'}
    assert all(r['status'] == 'measured' for r in quality['trials'])
    assert len(quality['examples']) == 1
    assert set(quality['spectra']['partitions']) == {'train', 'validation'}
    json.dumps(quality, allow_nan=False)


@pytest.fixture
def saved_audit_run(monkeypatch, tmp_path):
    import mne
    from agfl.datasets import get_split, load_dataset
    from agfl.datasets.eeg import CHANNEL_NAMES

    source = tmp_path/'A06T.gdf'
    source.write_bytes(b'Synthetic reader fixture, not physiological data')
    time = np.arange(24000) / 250
    signals = np.stack([1e-5*np.sin(2*np.pi*(7 + .2*i)*time) for i in range(25)])
    info = mne.create_info(CHANNEL_NAMES + ['EOG1', 'EOG2', 'EOG3'], 250,
                           ch_types=['eeg']*22 + ['eog']*3)
    raw = mne.io.RawArray(signals, info, verbose='ERROR')
    events = [(0., '32766')]
    for i in range(12):
        events.extend([(8.*i, '768'), (8.*i + 2., str(769 + i % 4))])
    raw.set_annotations(mne.Annotations([e[0] for e in events], [0.]*len(events), [e[1] for e in events]))
    monkeypatch.setattr(mne.io, 'read_raw_gdf', lambda *a, **k: raw.copy())
    bundle = load_dataset('eeg', {'data_dir': str(tmp_path), 'subjects': [6]})
    split = get_split(bundle, {'protocol': 'stratified'}, 0, tmp_path/'splits')
    config = {'dataset': 'eeg', 'subject_id': 'A06', 'data': bundle.metadata['preprocessing'],
              'resolved_metadata': deepcopy(bundle.metadata), 'dataset_fingerprint': bundle.fingerprint}
    run = tmp_path/'run'
    run.mkdir()
    (run/'config.json').write_text(json.dumps(config))
    (run/'split.json').write_text(json.dumps(split))
    # Reading this test array with allow_pickle=False would fail: it must be ignored.
    np.savez(run/'predictions.npz', validation_ids=np.asarray(split['sample_ids']['validation']),
             validation_targets=bundle.y[split['validation']], test_targets=np.asarray([{}], dtype=object))
    return run, source, split, tmp_path/'audit'


def test_raw_audit_reconstructs_identity_and_resumes(saved_audit_run):
    run, source, split, output = saved_audit_run
    first = audit_eeg_run(run, output)
    checks = {c['name']: c['status'] for c in first['checks']}
    assert checks['recording_identity'] == checks['preprocessing_identity'] == 'pass'
    assert checks['loader_split_alignment'] == checks['validation_label_alignment'] == 'pass'
    assert checks['cue_timing'] == 'pass'
    assert checks['cue_count'] == 'fail'  # Fixture has 12, not 288 trials; audit must report this.
    assert {r['sample_id'] for r in first['signals']['trials']} == set(
        split['sample_ids']['train'] + split['sample_ids']['validation'])
    assert (output/'trials.csv').is_file() and (output/'report.md').is_file()
    assert audit_eeg_run(run, output) == first
    source.write_bytes(b'Unexpected replacement recording')
    with pytest.raises(ValueError, match='recording hash'):
        audit_eeg_run(run, output)
    assert (output/'failure.json').is_file() and not (output/'audit.json').exists()


def test_raw_audit_reports_corrupt_validation_labels(saved_audit_run):
    run, _, split, output = saved_audit_run
    with np.load(run/'predictions.npz', allow_pickle=False) as archive:
        targets = archive['validation_targets'].copy()
    targets[0] = (targets[0] + 1) % 4
    np.savez(run/'predictions.npz', validation_ids=np.asarray(split['sample_ids']['validation']),
             validation_targets=targets)
    result = audit_eeg_run(run, output)
    assert next(c for c in result['checks'] if c['name'] == 'validation_label_alignment')['status'] == 'fail'


def test_audit_plots_need_only_saved_measurements(saved_audit_run, monkeypatch):
    pytest.importorskip('matplotlib')
    import mne
    from agfl.visualization.eeg_audit import plot_eeg_audit

    run, _, _, output = saved_audit_run
    audit_eeg_run(run, output)
    def forbidden(*args, **kwargs):
        raise AssertionError('Plotting must not reopen recordings')
    monkeypatch.setattr(mne.io, 'read_raw_gdf', forbidden)
    manifest = plot_eeg_audit(output)
    assert not manifest['skipped']
    assert len(manifest['figures']) == 8
    assert 'data:image/png;base64,' in (output/'report.html').read_text()
    assert (output/'figures/index.md').is_file()
    for figure in manifest['figures']:
        for name in figure['files']:
            assert (output/'figures'/name).stat().st_size > 0
