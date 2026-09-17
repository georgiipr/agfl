"""Portable report/discovery invariants; run on the cluster, not locally."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from agfl.analysis import analyze_results
from agfl.config import comparison_identity, experiment_identity
from agfl.metrics import classification_metrics
from agfl.result_paths import diagnostic_manifest_paths, resolve_artifact_run, result_paths
from agfl.visualization.diagnostics import diagnose_run
from agfl.visualization.results import generate_result_plots
from tests.test_analysis import save_run


def _copy_report_run(session, directory):
    target = session / 'report' / 'runs' / directory.relative_to(session / 'artifacts')
    shutil.copytree(directory, target)
    return target


def test_full_session_uses_canonical_results_even_when_report_is_incomplete(tmp_path):
    session = tmp_path / 'session'
    save_run(session / 'artifacts', 'agfl', 0, .7)
    first = next((session / 'artifacts').rglob('result.json')).parent
    _copy_report_run(session, first)
    save_run(session / 'artifacts', 'agfl', 1, .8)

    result = analyze_results(session, session / 'report' / 'analysis')

    assert result['n_runs'] == 2
    assert all(Path(source).is_relative_to(session / 'artifacts') for source in result['sources'])
    assert len(result_paths(session / 'report')) == 1
    assert len(result_paths(session / 'artifacts')) == 2


def test_discovery_handles_parent_sessions_legacy_trees_and_report_only_downloads(tmp_path):
    expected = []
    for name in ('first', 'second'):
        session = tmp_path / name
        saved = session / 'artifacts' / 'eeg' / 'seed_0' / 'result.json'
        saved.parent.mkdir(parents=True)
        saved.write_text('{}')
        _copy_report_run(session, saved.parent)
        expected.append(saved)
    legacy = tmp_path / 'old-layout' / 'seed_1' / 'result.json'
    legacy.parent.mkdir(parents=True)
    legacy.write_text('{}')
    expected.append(legacy)
    portable = tmp_path / 'download' / 'report' / 'runs' / 'seed_2' / 'result.json'
    portable.parent.mkdir(parents=True)
    portable.write_text('{}')
    expected.append(portable)

    assert result_paths(tmp_path) == sorted(expected)


def test_layout_marker_recognizes_session_before_first_report_is_published(tmp_path):
    artifacts = tmp_path / 'artifacts'
    artifacts.mkdir()
    (artifacts / 'session_layout.json').write_text('{"layout_version": 1}')
    (artifacts / 'result.json').write_text('{}')
    # A stale pre-migration copy must not become a second experiment.
    (tmp_path / 'old').mkdir()
    (tmp_path / 'old' / 'result.json').write_text('{}')

    assert result_paths(tmp_path) == [artifacts / 'result.json']


def test_diagnostic_manifest_discovery_prefers_published_copy(tmp_path):
    artifact_run = tmp_path / 'artifacts' / 'eeg' / 'seed_0'
    artifact_run.mkdir(parents=True)
    (artifact_run / 'result.json').write_text('{}')
    _copy_report_run(tmp_path, artifact_run)
    for section in ('artifacts', 'report'):
        path = tmp_path / section / 'diagnostics' / 'seed_0' / 'manifest.json'
        path.parent.mkdir(parents=True)
        path.write_text('{}')
    expected = tmp_path / 'report' / 'diagnostics' / 'seed_0' / 'manifest.json'

    assert diagnostic_manifest_paths(tmp_path) == [expected]
    assert diagnostic_manifest_paths(tmp_path / 'artifacts') == [
        tmp_path / 'artifacts' / 'diagnostics' / 'seed_0' / 'manifest.json']


def test_report_checkpoint_lookup_resolves_same_session_artifacts(tmp_path):
    canonical = tmp_path / 'artifacts' / 'selected' / 'eeg' / 'subject_A03' / 'seed_0'
    canonical.mkdir(parents=True)
    (canonical / 'checkpoint.pt').write_bytes(b'checkpoint lookup only')
    report = tmp_path / 'report' / 'runs' / canonical.relative_to(tmp_path / 'artifacts')
    report.mkdir(parents=True)

    assert resolve_artifact_run(canonical) == canonical
    assert resolve_artifact_run(report) == canonical


def test_report_only_checkpoint_diagnostics_explain_required_artifacts(tmp_path):
    report_run = tmp_path / 'report' / 'runs' / 'eeg' / 'seed_0'
    report_run.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match='downloadable report excludes checkpoints'):
        diagnose_run(report_run, tmp_path / 'output')


def test_report_alone_regenerates_result_figures_without_checkpoint_or_data(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    session = tmp_path / 'cluster-session'
    run = save_run(session / 'artifacts', 'agfl', 0, 1.)
    directory = next((session / 'artifacts').rglob('result.json')).parent
    targets = np.array([0, 1, 0, 1])
    probabilities = np.array([[.8, .2], [.2, .8], [.7, .3], [.4, .6]])
    metrics = {**classification_metrics(targets, probabilities), 'loss': .3}
    run['config']['resolved_metadata'] = {'num_classes': 2, 'label_names': ['First', 'Second']}
    run.update(validation=metrics, test=metrics, best_checkpoint_epoch=1,
               comparison_id=comparison_identity(run['config']),
               experiment_id=experiment_identity(run['config']))
    (directory / 'result.json').write_text(json.dumps(run))
    (directory / 'config.json').write_text(json.dumps(run['config']))
    identifiers = {part: [f'{part}-{index}' for index in range(4)] for part in ('validation', 'test')}
    (directory / 'split.json').write_text(json.dumps({
        'split_id': run['split_id'], 'fingerprint': run['dataset_fingerprint'], 'sample_ids': identifiers}))
    (directory / 'history.json').write_text(json.dumps([
        {'epoch': 1, 'train_loss': .4, 'validation': metrics, 'learning_rate': .001}]))
    predictions = {}
    for partition, sample_ids in identifiers.items():
        predictions.update({f'{partition}_targets': targets, f'{partition}_probabilities': probabilities,
                            f'{partition}_ids': np.array(sample_ids)})
    np.savez_compressed(directory / 'predictions.npz', **predictions)
    _copy_report_run(session, directory)
    portable = tmp_path / 'downloaded-report'
    shutil.copytree(session / 'report', portable)

    import agfl.datasets
    import agfl.models

    def forbidden(*args, **kwargs):
        raise AssertionError('Portable result plotting must not load data or construct a model')

    monkeypatch.setattr(agfl.datasets, 'load_dataset', forbidden)
    monkeypatch.setattr(agfl.models, 'get_model_spec', forbidden)
    aggregation = analyze_results(portable, portable / 'analysis')
    manifest = generate_result_plots(aggregation, portable / 'analysis' / 'figures')

    assert aggregation['n_runs'] == 1
    assert all(Path(source).is_relative_to(portable) for source in aggregation['sources'])
    assert not list(portable.rglob('checkpoint.pt'))
    assert manifest['figures']
    assert not manifest['skipped']
    for figure in manifest['figures']:
        for filename in figure['files']:
            assert (portable / 'analysis' / 'figures' / filename).is_file()


def test_genuine_duplicates_are_still_rejected_inside_a_report(tmp_path):
    directory = tmp_path / 'report' / 'runs'
    save_run(directory, 'agfl', 0, .7)
    source = next(directory.rglob('result.json'))
    duplicate = directory / 'accidental-copy'
    duplicate.mkdir()
    shutil.copy2(source, duplicate / source.name)

    with pytest.raises(ValueError, match='pseudoreplication'):
        analyze_results(tmp_path / 'report', tmp_path / 'analysis')


def test_failed_run_cannot_be_counted_as_a_completed_report_result(tmp_path):
    runs = tmp_path / 'report' / 'runs'
    save_run(runs, 'agfl', 0, .7)
    failed = runs / 'failed-seed'
    failed.mkdir()
    (failed / 'result.json').write_text('interrupted output')
    (failed / 'failure.json').write_text('{"type":"KeyboardInterrupt"}')

    result = analyze_results(tmp_path / 'report', tmp_path / 'report' / 'analysis')

    assert result['n_runs'] == 1
    assert all(Path(source).parent != failed for source in result['sources'])
