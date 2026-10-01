"""Target-only checks of cue alignment, artifact labels and trial isolation."""
from types import SimpleNamespace
import numpy as np
from agfl.datasets import load_dataset
from agfl.datasets.base import bandpass_finite_spans


def fixture_recording(monkeypatch, tmp_path, session='T'):
    path = tmp_path / f'A01{session}.gdf'
    path.write_bytes(b'fake source for a mocked GDF reader')
    time = np.arange(17000) / 250
    signal = np.stack([np.sin(2 * np.pi * (8 + i / 10) * time) for i in range(25)])
    rows = [(0, 0, 32766)]
    for i in range(8):
        rows.extend([(2000 * i, 0, 768), (2000 * i + 500, 0, 783 if session == 'E' else 769 + i % 4)])
    rows.append((2000, 0, 1023))
    events = np.asarray(sorted(rows))
    raw = SimpleNamespace(info={'sfreq': 250}, first_samp=0, ch_names=[str(i) for i in range(25)],
                          get_data=lambda: signal)
    import mne
    monkeypatch.setattr(mne.io, 'read_raw_gdf', lambda *a, **kw: raw)
    monkeypatch.setattr(mne, 'events_from_annotations', lambda *a, **kw: (events, {str(c): c for c in set(events[:, 2])}))
    if session == 'E':
        from scipy.io import savemat
        savemat(tmp_path / 'A01E.mat', {'classlabel': np.tile(np.arange(1, 5), 2)[:, None]})
    # These checks exercise the explicit trial-local/exclusion settings; the
    # project defaults are run filtering with every cued trial retained.
    return signal, {'data_dir': str(tmp_path), 'subjects': [1], 'sessions': [session],
                    'artifact_policy': 'exclude', 'filter_scope': 'trial'}


def test_cue_window_and_artifact_exclusion_preserve_labels(monkeypatch, tmp_path):
    signal, config = fixture_recording(monkeypatch, tmp_path)
    bundle = load_dataset('eeg', config)
    assert bundle.x.shape == (7, 22, 1000)
    assert bundle.y.tolist() == [0, 2, 3, 0, 1, 2, 3]
    assert bundle.metadata['skipped']['artifact'] == 1
    assert set(bundle.groups) == {'A01'}
    expected, _ = bandpass_finite_spans(signal[:22, 500:1500], 250, 2, 30, gdf_missing=True)
    np.testing.assert_array_equal(bundle.x[0], expected.astype(np.float32))


def test_neighbor_trial_cannot_change_other_trial_preprocessing(monkeypatch, tmp_path):
    signal, config = fixture_recording(monkeypatch, tmp_path)
    original = load_dataset('eeg', config)
    signal[:, 12500:13500] += 100 * np.sin(np.arange(1000))[None]
    changed = load_dataset('eeg', config)
    np.testing.assert_array_equal(original.x[:5], changed.x[:5])
    assert not np.array_equal(original.x[5], changed.x[5])


def test_e_session_labels_remain_aligned_after_excluding_artifacts(monkeypatch, tmp_path):
    _, config = fixture_recording(monkeypatch, tmp_path, 'E')
    bundle = load_dataset('eeg', config)
    assert bundle.y.tolist() == [0, 2, 3, 0, 1, 2, 3]
    assert bundle.sample_ids[1].startswith('A01E:cue:002:')


def test_default_run_filtering_keeps_marked_trials_and_counts_them(monkeypatch, tmp_path):
    signal, config = fixture_recording(monkeypatch, tmp_path)
    config = {k: v for k, v in config.items() if k not in {'artifact_policy', 'filter_scope'}}
    bundle = load_dataset('eeg', config)
    assert bundle.metadata['preprocessing']['artifact_policy'] == 'include'
    assert bundle.metadata['preprocessing']['filter_scope'] == 'run'
    # The expert-marked trial is retained and counted, not dropped.
    assert bundle.x.shape == (8, 22, 1000)
    assert bundle.y.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]
    assert bundle.metadata['skipped']['artifact'] == 0
    assert bundle.metadata['artifact_trials_seen'] == 1
    # The whole run is filtered once; windows are then cut from the filtered run.
    filtered, _ = bandpass_finite_spans(signal[:22], 250, 2, 30, [0], gdf_missing=True)
    np.testing.assert_array_equal(bundle.x[0], filtered[:, 500:1500].astype(np.float32))
