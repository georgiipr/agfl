"""Locate saved outputs without counting a session's downloadable copy twice."""
import os
from pathlib import Path


def _saved_paths(input_root, filename, *, report_first=False):
    root = Path(input_root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Saved-output directory does not exist: {root}')
    paths = []
    for current, directories, filenames in os.walk(root):
        directory = Path(current)
        artifacts, report = directory / 'artifacts', directory / 'report'
        is_session = ((artifacts / 'session_layout.json').is_file()
                      or (report / 'runs').is_dir())
        if is_session:
            # On the experiment machine, canonical runs can be newer than their
            # downloadable copies (for example after an interrupted export).
            # A copied report remains independently readable without artifacts.
            preferred = (report, artifacts) if report_first else (artifacts, report)
            selected = next((path for path in preferred if path.is_dir()), None)
            if selected is not None:
                directories[:] = [selected.name]
        else:
            directories.sort()
        if filename in filenames:
            paths.append(directory / filename)
    return sorted(paths)


def result_paths(input_root):
    """Find canonical results, or portable report copies when artifacts are absent.

    This also accepts an ordinary old-style run tree, an explicit artifacts
    subtree, a report directory, or a parent containing several sessions.
    Genuine duplicate experiment/seed records remain visible for analysis to
    reject; only the known report/artifacts mirror is pruned.
    """
    return _saved_paths(input_root, 'result.json')


def diagnostic_manifest_paths(input_root):
    """Prefer published diagnostic figures at each recognized session boundary."""
    return _saved_paths(input_root, 'manifest.json', report_first=True)


def resolve_artifact_run(run_dir):
    """Locate a checkpoint for a run path, including its portable report copy."""
    directory = Path(run_dir).expanduser().resolve()
    if (directory / 'checkpoint.pt').is_file():
        return directory
    for ancestor in (directory, *directory.parents):
        if ancestor.name != 'runs' or ancestor.parent.name != 'report':
            continue
        candidate = ancestor.parent.parent / 'artifacts' / directory.relative_to(ancestor)
        if (candidate / 'checkpoint.pt').is_file():
            return candidate
        raise FileNotFoundError(
            f'Checkpoint diagnostics require {candidate / "checkpoint.pt"}. '
            'The downloadable report excludes checkpoints. Generate diagnostics '
            'on the cluster, where the session artifacts are stored, or transfer '
            'the corresponding artifacts run. Saved-result plots can be generated '
            'from the report alone with analyze --plots.')
    raise FileNotFoundError(
        f'No checkpoint.pt in {directory}. Checkpoint diagnostics require one '
        'artifacts seed directory. A report-only download supports analyze --plots '
        'and already generated diagnostic figures, but cannot reconstruct a model.')
