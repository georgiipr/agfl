"""Separate portable reports from checkpoints and other working artifacts.

This module deliberately uses only the standard library: organizing completed
results does not load a model, checkpoint, dataset, or training dependency.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import shutil
import tempfile


LAYOUT_VERSION = 1
LAYOUT_FILE = 'session_layout.json'
RUN_REPORT_FILES = (
    'config.json', 'split.json', 'history.json', 'selection.json',
    'predictions.npz', 'result.json', 'failure.json',
)
SESSION_REPORT_FILES = (
    'search_plan.json', 'selection_report.json', 'search_result.json', 'search_report.md',
)
_ROOT_LOCKS = {'.run.lock', '.search.lock', '.session-layout.lock'}
_REVIEW_SUFFIXES = {'.json', '.csv', '.tsv', '.txt', '.md', '.html', '.png', '.pdf', '.svg'}


@dataclass(frozen=True)
class SessionPaths:
    root: Path
    artifacts: Path
    report: Path


def _paths(root):
    root = Path(root).expanduser()
    return SessionPaths(root, root / 'artifacts', root / 'report')


def _validate_session_root(paths):
    """A layout marker belongs inside report/artifacts, never their parent."""
    marker = paths.root / LAYOUT_FILE
    if not marker.exists() and not marker.is_symlink():
        return
    if marker.is_symlink():
        raise ValueError(f'Refusing a symlinked session marker: {marker}')
    try:
        metadata = json.loads(marker.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f'This folder contains a reserved layout marker; pass the parent session: {marker}') from error
    kind = metadata.get('kind') if isinstance(metadata, dict) else None
    if kind == 'agfl_report':
        raise ValueError('This is a report-only download or report subfolder. '
                         'Pass the complete parent session on the cluster, not report/.')
    if kind == 'agfl_session':
        raise ValueError('This is the artifacts subfolder. Pass its parent session, not artifacts/.')
    raise ValueError(f'This folder contains a reserved layout marker; pass the parent session: {marker}')


def _legacy_entries(paths):
    if not paths.root.exists():
        return []
    return sorted((entry for entry in paths.root.iterdir()
                   if entry.name not in {'artifacts', 'report', *_ROOT_LOCKS}),
                  key=lambda entry: entry.name)


def _check_destination(path, base):
    """Do not follow an output symlink into unrelated user files."""
    path, base = Path(path), Path(base)
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'Refusing to replace or write through a symlink: {part}')
        if part == base:
            break


def _copy_file(source, destination, base):
    if source.is_symlink():
        raise ValueError(f'Refusing to publish a symlink: {source}')
    _check_destination(destination, base)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix='.' + destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    descriptor, temporary_name = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w') as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write('\n')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _initialize(paths):
    for directory in (paths.artifacts, paths.report):
        _check_destination(directory, paths.root)
        directory.mkdir(parents=True, exist_ok=True)
    marker = paths.artifacts / LAYOUT_FILE
    _check_destination(marker, paths.root)
    if marker.exists():
        metadata = json.loads(marker.read_text())
        if metadata.get('layout_version') != LAYOUT_VERSION or metadata.get('kind') != 'agfl_session':
            raise ValueError(f'Unsupported session layout: {marker}')
    else:
        _write_json(marker, {'layout_version': LAYOUT_VERSION, 'kind': 'agfl_session'})
    report_marker = paths.report / LAYOUT_FILE
    _check_destination(report_marker, paths.root)
    if not report_marker.exists():
        _write_json(report_marker, {'layout_version': LAYOUT_VERSION, 'kind': 'agfl_report'})
    readme = paths.report / 'README.md'
    _check_destination(readme, paths.root)
    if not readme.exists():
        readme.write_text(
            '# AGFL report bundle\n\n'
            'Download this `report/` folder for result review and issue diagnosis. '
            'It contains saved metrics, configurations and provenance, splits, histories, '
            'predictions, and any plots already generated.\n\n'
            '- `runs/`: compact records of every run.\n'
            '- `analysis/`: result tables, statistical comparisons, and plots.\n'
            '- `diagnostics/`: checkpoint diagnostic figures and their small numeric summaries.\n'
            '\n'
            'Plots appear after the plotting commands finish; a missing plot is not generated '
            'by downloading this folder. Older organized sessions may keep checkpoint plots '
            'inside `analysis/diagnostics/`.\n\n'
            'The sibling `artifacts/` folder stays on the cluster. It holds checkpoints and '
            'working files. Regenerating checkpoint diagnostics requires those checkpoints '
            'and the original datasets; a report download is not a full experiment backup.\n\n'
            'A report can include partial histories and failure details from interrupted runs. '
            'Only completed result records should be aggregated.\n')
    activity_lock = paths.artifacts / '.session.lock'
    _check_destination(activity_lock, paths.root)
    activity_lock.touch(exist_ok=True)


@contextmanager
def _layout_lock(root):
    """Serialize layout creation/migration without deleting the lock inode."""
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    path = root / '.session-layout.lock'
    _check_destination(path, root)
    with path.open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'This session layout is being organized by another process: {root}') from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def _inactive_writers(root):
    """Stale lock files are safe; an actively held OS lock blocks organization."""
    import fcntl

    with ExitStack() as handles:
        for path in sorted(root.rglob('*.lock')):
            if path == root / '.session-layout.lock':
                continue
            if path.is_symlink():
                raise ValueError(f'Refusing to organize a symlinked lock: {path}')
            handle = handles.enter_context(path.open('a'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'Cannot organize an active session; stop its writer first: {path}') from error
        yield


def session_paths(root, *, create=False):
    """Resolve a session; initialize new output only when explicitly requested.

    Existing unsplit sessions must first pass through ``organize_session`` so
    a relaunch cannot silently ignore their checkpoints or completed results.
    """
    paths = _paths(root)
    if create:
        _validate_session_root(paths)
        with _layout_lock(paths.root):
            if _legacy_entries(paths):
                command = 'python -m agfl_speech organize-results ' + shlex.quote(str(paths.root))
                raise ValueError(f'This output folder uses the previous layout. Organize it first: {command}')
            if paths.report.exists() and not paths.artifacts.exists():
                raise ValueError('This looks like a downloaded report without its artifacts. '
                                 'Use the complete cluster session or a fresh --output-dir.')
            _initialize(paths)
    return paths


def find_session(path):
    """Find the containing session, or a portable report with no checkpoints."""
    path = Path(path).expanduser().absolute()
    if path.is_file():
        path = path.parent
    for parent in (path, *path.parents):
        paths = _paths(parent)
        if (paths.artifacts / LAYOUT_FILE).is_file():
            return paths
        if paths.artifacts.is_dir() and paths.report.is_dir():
            return paths
        marker = parent / LAYOUT_FILE
        if marker.is_file():
            try:
                metadata = json.loads(marker.read_text())
            except (OSError, ValueError):
                continue
            if metadata.get('kind') == 'agfl_report':
                # A copied/renamed report is its own portable root. Never infer
                # that an unrelated Downloads parent is an experiment session.
                sibling = _paths(parent.parent)
                if parent.name == 'report' and sibling.artifacts.is_dir():
                    return sibling
                return SessionPaths(parent, parent / 'artifacts', parent)
    return None


@contextmanager
def session_activity(root):
    """Permit concurrent runs while excluding session organization/sync."""
    import fcntl

    paths = session_paths(root, create=True)
    with ExitStack() as handles:
        with _layout_lock(paths.root):
            lock = handles.enter_context((paths.artifacts / '.session.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'This session is being organized: {paths.root}') from error
        yield paths


def _mirror_files(source, destination, names, base):
    for name in names:
        target = destination / name
        _check_destination(target, base)
        if not (source / name).is_file():
            target.unlink(missing_ok=True)
    for name in names:
        source_file = source / name
        if source_file.is_file():
            _copy_file(source_file, destination / name, base)


def mirror_run(session_root, run_dir, *, names=None):
    """Refresh small run records; never copy checkpoint or raw signal files."""
    paths = _paths(session_root)
    names = RUN_REPORT_FILES if names is None else tuple(names)
    if any(name not in RUN_REPORT_FILES for name in names):
        raise ValueError('Run reports accept only the documented compact artifact files')
    source = Path(run_dir)
    try:
        relative = source.resolve().relative_to(paths.artifacts.resolve())
    except ValueError as error:
        raise ValueError(f'Run is outside this session artifacts folder: {source}') from error
    destination = paths.report / 'runs' / relative
    _mirror_files(source, destination, names, paths.root)
    return destination


def publish_session_metadata(session_root):
    paths = _paths(session_root)
    _mirror_files(paths.artifacts, paths.report, SESSION_REPORT_FILES, paths.root)
    return paths.report


def _review_asset(path, relative):
    if path.name.startswith('.') or any(part.startswith('.') for part in relative.parts):
        return False
    if path.suffix.lower() in _REVIEW_SUFFIXES:
        return True
    return (path.suffix.lower() == '.npz'
            and (path.name in {'embedding.npz', 'predictions.npz'} or 'mixers' in relative.parts))


def _publish_saved_reviews(paths):
    # Existing analysis/audit trees remain intact in artifacts for provenance.
    # Publish only review assets: never copy arbitrary NPZ/NPY signal dumps.
    for source in paths.artifacts.iterdir():
        if source.name in {'candidates', 'selected', LAYOUT_FILE,
                           *RUN_REPORT_FILES, *SESSION_REPORT_FILES}:
            continue
        review_tree = (source.name in {'analysis', 'diagnostics', 'figures', 'plots', 'tables', 'reports'}
                       or 'audit' in source.name.lower())
        if source.is_file():
            assets = [source]
        elif source.is_dir() and review_tree:
            assets = sorted(source.rglob('*'))
        else:
            continue
        for asset in assets:
            relative = asset.relative_to(paths.artifacts)
            if asset.is_file() and _review_asset(asset, relative):
                if relative == Path('README.md'):
                    # Preserve session-specific historical instructions beside
                    # the generated explanation of the portable bundle.
                    relative = Path('original_README.md')
                destination = paths.report / relative
                # An existing report asset may have been regenerated since
                # migration. Never overwrite it with the older archived copy.
                if not destination.exists():
                    _copy_file(asset, destination, paths.root)


def _sync_report(paths):
    # Revisit existing mirrors too: a killed restart must not leave a completed
    # report result for a canonical run whose result was already invalidated.
    # An early failure can precede split creation; a configuration written just
    # before a hard interruption also remains useful without any other record.
    # Partial historical downloads may contain only a result, history, or
    # prediction file. Keep these available for review without implying that
    # an incomplete record is valid for analysis or checkpoint reconstruction.
    run_dirs = {file.parent for name in RUN_REPORT_FILES
                for file in paths.artifacts.rglob(name)}
    mirrors = paths.report / 'runs'
    if mirrors.exists():
        for name in RUN_REPORT_FILES:
            run_dirs.update(paths.artifacts / file.parent.relative_to(mirrors)
                            for file in mirrors.rglob(name))
    for run_dir in sorted(run_dirs):
        mirror_run(paths.root, run_dir)
    publish_session_metadata(paths.root)
    _publish_saved_reviews(paths)
    return paths


def sync_report(root):
    """Recover compact records after interruption; preserve existing plots."""
    paths = _paths(root)
    _validate_session_root(paths)
    if not paths.artifacts.is_dir():
        raise ValueError(f'The canonical artifacts folder is missing: {paths.artifacts}')
    with _layout_lock(paths.root), _inactive_writers(paths.root):
        if _legacy_entries(paths):
            raise ValueError('This session still contains previous-layout files; run organize-results first.')
        _initialize(paths)
        return _sync_report(paths)


def organize_session(root):
    """Conservatively migrate a completed old layout, or refresh a new one.

    Preflight checks every move before changing content. Existing files are
    renamed without rewriting their saved configuration or provenance. Stop
    launchers before organization; active advisory writer locks are rejected.
    """
    paths = _paths(root)
    _validate_session_root(paths)
    if not paths.root.is_dir():
        raise ValueError(f'Result session does not exist: {paths.root}')
    with _layout_lock(paths.root), _inactive_writers(paths.root):
        entries = _legacy_entries(paths)
        if paths.report.exists() and not paths.artifacts.exists() and not entries:
            raise ValueError('This is a report-only download; organize the full session on the cluster.')
        for entry in entries:
            destination = paths.artifacts / entry.name
            if destination.exists() or destination.is_symlink():
                raise ValueError(f'Migration would overwrite existing content: {destination}')
            if entry.is_symlink() or any(child.is_symlink() for child in entry.rglob('*')):
                raise ValueError(f'Refusing to organize a tree containing symlinks: {entry}')
        _initialize(paths)
        for entry in entries:
            entry.rename(paths.artifacts / entry.name)
        return _sync_report(paths)
