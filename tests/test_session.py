"""Filesystem-only coverage for the report/artifact session boundary."""
import json
from pathlib import Path

import pytest

from agfl.session import (
    find_session, mirror_run, organize_session, publish_session_metadata,
    session_activity, session_paths, sync_report,
)


def _run(directory, *, completed=True):
    directory.mkdir(parents=True, exist_ok=True)
    records = {'config.json': {'output_dir': 'unchanged/original'},
               'split.json': {'split_id': 'same-original-split'},
               'history.json': [{'epoch': 1, 'train_loss': 1.5}]}
    if completed:
        records['result.json'] = {'status': 'completed', 'test': {'accuracy': .8}}
    for name, record in records.items():
        (directory / name).write_text(json.dumps(record))
    (directory / 'checkpoint.pt').write_bytes(b'checkpoint-preserved-byte-for-byte')
    (directory / 'predictions.npz').write_bytes(b'compact-predictions')
    (directory / 'raw_signals.npy').write_bytes(b'do-not-publish')
    return directory


def test_new_session_requires_explicit_creation(tmp_path):
    root = tmp_path / 'session'
    paths = session_paths(root)
    assert not root.exists()
    assert paths.artifacts == root / 'artifacts'
    paths = session_paths(root, create=True)
    assert paths.artifacts.is_dir() and paths.report.is_dir()
    assert (paths.report / 'README.md').is_file()
    assert find_session(paths.artifacts / 'eeg' / 'seed_0') == paths


def test_previous_layout_requires_organization(tmp_path):
    root = tmp_path / 'old session'
    run = _run(root / 'eeg' / 'seed_0')
    original = (run / 'checkpoint.pt').read_bytes()
    with pytest.raises(ValueError, match='organize-results'):
        session_paths(root, create=True)
    assert (run / 'checkpoint.pt').read_bytes() == original
    assert not (root / 'artifacts').exists()


def test_run_mirror_is_small_and_refresh_removes_stale_completion(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    run = _run(paths.artifacts / 'candidates' / 'dense' / 'A03' / 'seed_0')
    destination = mirror_run(paths.root, run)
    assert (destination / 'predictions.npz').read_bytes() == b'compact-predictions'
    assert not (destination / 'checkpoint.pt').exists()
    assert not (destination / 'raw_signals.npy').exists()
    assert json.loads((destination / 'config.json').read_text())['output_dir'] == 'unchanged/original'
    (destination / 'my_notes.txt').write_text('keep user notes')
    (run / 'result.json').unlink()
    (run / 'predictions.npz').unlink()
    (run / 'failure.json').write_text('{"type":"RuntimeError","message":"interrupted"}')
    mirror_run(paths.root, run)
    assert not (destination / 'result.json').exists()
    assert not (destination / 'predictions.npz').exists()
    assert (destination / 'failure.json').is_file()
    assert (destination / 'my_notes.txt').read_text() == 'keep user notes'


def test_mirror_rejects_run_outside_session(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    run = _run(tmp_path / 'other')
    with pytest.raises(ValueError, match='outside'):
        mirror_run(paths.root, run)


def test_history_only_mirror_preserves_other_records(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    run = _run(paths.artifacts / 'eeg' / 'seed_0')
    destination = mirror_run(paths.root, run)
    (run / 'result.json').unlink()
    (run / 'history.json').write_text('[{"epoch":2}]')
    mirror_run(paths.root, run, names=('history.json',))
    assert (destination / 'result.json').is_file()
    assert json.loads((destination / 'history.json').read_text()) == [{'epoch': 2}]
    with pytest.raises(ValueError, match='compact'):
        mirror_run(paths.root, run, names=('checkpoint.pt',))


def test_search_summary_refresh_removes_stale_completion(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    (paths.artifacts / 'search_plan.json').write_text('{"candidate_order":["dense"]}')
    (paths.artifacts / 'search_result.json').write_text('{"mean_test_accuracy":0.8}')
    (paths.artifacts / 'search_report.md').write_text('# Search result\n')
    publish_session_metadata(paths.root)
    assert (paths.report / 'search_result.json').is_file()
    (paths.artifacts / 'search_result.json').unlink()
    (paths.artifacts / 'search_report.md').unlink()
    publish_session_metadata(paths.root)
    assert (paths.report / 'search_plan.json').is_file()
    assert not (paths.report / 'search_result.json').exists()
    assert not (paths.report / 'search_report.md').exists()


def test_organize_preserves_originals_and_publishes_review_assets(tmp_path):
    root = tmp_path / 'session'
    run = _run(root / 'selected' / 'eeg' / 'subject_A03' / 'seed_0')
    config_bytes = (run / 'config.json').read_bytes()
    checkpoint_bytes = (run / 'checkpoint.pt').read_bytes()
    (root / 'search_report.md').write_text('# Existing report\n')
    diagnostics = root / 'analysis' / 'diagnostics' / 'seed_0'
    (diagnostics / 'mixers').mkdir(parents=True)
    for name in ('plot.png', 'plot.pdf', 'manifest.json', 'embedding.npz', 'mixers/graph.npz'):
        (diagnostics / name).write_bytes(b'preserved-review-data')
    (diagnostics / 'raw_signals.npz').write_bytes(b'large-raw-data')
    audit = root / 'raw-audit'
    audit.mkdir()
    (audit / 'audit.json').write_text('{}')
    (audit / 'trials.csv').write_text('trial,class\n0,1\n')
    (run / '.run.lock').touch()  # An unlocked/stale file must not block migration.
    paths = organize_session(root)
    moved = paths.artifacts / 'selected' / 'eeg' / 'subject_A03' / 'seed_0'
    assert not run.exists()
    assert (moved / 'config.json').read_bytes() == config_bytes
    assert (moved / 'checkpoint.pt').read_bytes() == checkpoint_bytes
    assert (paths.report / 'runs' / moved.relative_to(paths.artifacts) / 'result.json').is_file()
    report_diag = paths.report / 'analysis' / 'diagnostics' / 'seed_0'
    assert (report_diag / 'embedding.npz').read_bytes() == b'preserved-review-data'
    assert (report_diag / 'mixers' / 'graph.npz').is_file()
    assert not (report_diag / 'raw_signals.npz').exists()
    assert (paths.artifacts / 'analysis' / 'diagnostics' / 'seed_0' / 'raw_signals.npz').is_file()
    assert (paths.report / 'raw-audit' / 'trials.csv').is_file()
    # An idempotent organization must preserve plots regenerated after migration.
    (report_diag / 'plot.png').write_bytes(b'newer-report-plot')
    organize_session(root)
    assert (report_diag / 'plot.png').read_bytes() == b'newer-report-plot'


def test_organization_preflights_collisions_without_moving_other_files(tmp_path):
    root = tmp_path / 'session'
    _run(root / 'eeg' / 'seed_0')
    (root / 'artifacts' / 'eeg').mkdir(parents=True)
    (root / 'search_plan.json').write_text('{}')
    with pytest.raises(ValueError, match='overwrite'):
        organize_session(root)
    assert (root / 'search_plan.json').is_file()
    assert (root / 'eeg' / 'seed_0' / 'checkpoint.pt').is_file()


def test_organization_publishes_standalone_reports_figures_and_early_failures(tmp_path):
    root = tmp_path / 'session'
    root.mkdir()
    report_files = {'report.html': '<html>Existing report</html>',
                    'report.md': '# Original experiment results\n',
                    'scores.csv': 'seed,accuracy\n0,0.8\n',
                    'overview.png': 'image-bytes',
                    'figures/index.md': '# Existing figures\n',
                    'figures/accuracy.pdf': 'pdf-bytes',
                    'plots/matrix.png': 'plot-bytes'}
    for name, content in report_files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    (root / 'README.md').write_text('# Original session notes\n')
    (root / 'figures' / 'raw_signals.npz').write_bytes(b'keep-on-cluster')
    early = root / 'eeg' / 'seed_0'
    early.mkdir(parents=True)
    (early / 'failure.json').write_text('{"message":"failed before config and split"}')
    configured = root / 'eeg' / 'seed_1'
    configured.mkdir()
    (configured / 'config.json').write_text('{"seed":1}')
    history_only = root / 'eeg' / 'seed_2'
    history_only.mkdir()
    (history_only / 'history.json').write_text('[{"epoch":1}]')
    result_only = root / 'eeg' / 'seed_3'
    result_only.mkdir()
    (result_only / 'result.json').write_text('{"status":"completed","seed":3}')
    paths = organize_session(root)
    for name, content in report_files.items():
        assert (paths.report / name).read_text() == content
        assert (paths.artifacts / name).read_text() == content
    assert (paths.report / 'original_README.md').read_text() == '# Original session notes\n'
    assert not (paths.report / 'figures' / 'raw_signals.npz').exists()
    assert (paths.report / 'runs' / 'eeg' / 'seed_0' / 'failure.json').is_file()
    assert (paths.report / 'runs' / 'eeg' / 'seed_1' / 'config.json').is_file()
    assert json.loads((paths.report / 'runs' / 'eeg' / 'seed_2' / 'history.json').read_text()) == [{'epoch': 1}]
    assert (paths.report / 'runs' / 'eeg' / 'seed_3' / 'result.json').is_file()
    (paths.report / 'report.html').write_text('<html>Regenerated report</html>')
    (paths.report / 'figures' / 'accuracy.pdf').write_text('regenerated-pdf')
    sync_report(root)
    assert (paths.report / 'report.html').read_text() == '<html>Regenerated report</html>'
    assert (paths.report / 'figures' / 'accuracy.pdf').read_text() == 'regenerated-pdf'


def test_organizing_active_run_fails_without_moving_anything(tmp_path):
    import fcntl

    root = tmp_path / 'session'
    run = _run(root / 'eeg' / 'seed_0')
    with (run / '.run.lock').open('a') as active:
        fcntl.flock(active, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match='active session'):
            organize_session(root)
    assert (run / 'checkpoint.pt').is_file()
    assert not (root / 'artifacts').exists()


def test_shared_session_activity_blocks_organization_but_allows_other_runs(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    with session_activity(paths.root):
        with session_activity(paths.root):
            with pytest.raises(RuntimeError, match='active session'):
                organize_session(paths.root)
            with pytest.raises(RuntimeError, match='active session'):
                sync_report(paths.root)
    organize_session(paths.root)


def test_sync_recovers_partial_runs_and_removes_invalidated_result(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    run = _run(paths.artifacts / 'eeg' / 'seed_0')
    destination = mirror_run(paths.root, run)
    (run / 'result.json').unlink()
    (run / 'failure.json').write_text('{"message":"stopped"}')
    sync_report(paths.root)
    assert not (destination / 'result.json').exists()
    assert (destination / 'failure.json').is_file()
    for file in run.iterdir():
        file.unlink()
    sync_report(paths.root)
    assert not (destination / 'config.json').exists()
    assert not (destination / 'failure.json').exists()


def test_symlink_migration_and_mirror_are_refused(tmp_path):
    root = tmp_path / 'old-session'
    root.mkdir()
    external = tmp_path / 'private-checkpoint.pt'
    external.write_bytes(b'unrelated')
    (root / 'checkpoint.pt').symlink_to(external)
    with pytest.raises(ValueError, match='symlinks'):
        organize_session(root)
    paths = session_paths(tmp_path / 'new-session', create=True)
    run = _run(paths.artifacts / 'eeg' / 'seed_0')
    (run / 'predictions.npz').unlink()
    (run / 'predictions.npz').symlink_to(external)
    with pytest.raises(ValueError, match='symlink'):
        mirror_run(paths.root, run)
    assert external.read_bytes() == b'unrelated'


def test_report_only_download_is_found_but_not_initialized_as_training(tmp_path):
    downloaded = tmp_path / 'downloaded-report'
    downloaded.mkdir()
    (downloaded / 'session_layout.json').write_text('{"kind":"agfl_report","layout_version":1}')
    paths = find_session(downloaded / 'runs' / 'eeg' / 'seed_0')
    assert paths.report == downloaded
    assert paths.root == downloaded  # Do not misidentify Downloads as a session.
    root = tmp_path / 'report-only-session'
    (root / 'report').mkdir(parents=True)
    with pytest.raises(ValueError, match='downloaded report'):
        session_paths(root, create=True)


@pytest.mark.parametrize('subfolder', ('report', 'artifacts'))
def test_subfolders_cannot_be_organized_or_used_as_training_session(tmp_path, subfolder):
    paths = session_paths(tmp_path / 'session', create=True)
    target = paths.root / subfolder
    (target / 'notes.md').write_text('must remain byte-identical')
    before = {file.relative_to(target): file.read_bytes()
              for file in target.rglob('*') if file.is_file()}
    with pytest.raises(ValueError, match='parent session'):
        organize_session(target)
    with pytest.raises(ValueError, match='parent session'):
        session_paths(target, create=True)
    with pytest.raises(ValueError, match='parent session'):
        sync_report(target)
    after = {file.relative_to(target): file.read_bytes()
             for file in target.rglob('*') if file.is_file()}
    assert before == after
    assert not (target / '.session-layout.lock').exists()
    assert not (target / 'artifacts').exists()
    assert not (target / 'report').exists()


def test_organization_resumes_partial_top_level_migration(tmp_path):
    paths = session_paths(tmp_path / 'session', create=True)
    moved = _run(paths.artifacts / 'eeg' / 'seed_0')
    remaining = _run(paths.root / 'candidates' / 'dense' / 'A03' / 'seed_1')
    original_checkpoint = (moved / 'checkpoint.pt').read_bytes()
    remaining_config = (remaining / 'config.json').read_bytes()
    organized = organize_session(paths.root)
    assert (moved / 'checkpoint.pt').read_bytes() == original_checkpoint
    relocated = paths.artifacts / 'candidates' / 'dense' / 'A03' / 'seed_1'
    assert (relocated / 'config.json').read_bytes() == remaining_config
    assert not remaining.exists()
    assert (organized.report / 'runs' / 'eeg' / 'seed_0' / 'result.json').is_file()
    assert (organized.report / 'runs' / 'candidates' / 'dense' / 'A03' / 'seed_1' / 'result.json').is_file()
