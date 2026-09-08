"""Machine-readable output independent of model/training imports."""
import json
from contextlib import contextmanager
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def run_directory(path):
    """Serialize writers to one seed; process exit releases the OS lock."""
    import fcntl

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    # Keep the lock file in place: deleting it would let a second process lock
    # a different inode while this process still owns the original lock.
    with (path / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Another process is currently writing this seed: {path}') from error
        try:
            yield path
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def reset_run_artifacts(path):
    """Clear generated artifacts before restarting, preserving unrelated files."""
    path = Path(path)
    # Remove completion first so an aborted replacement cannot be aggregated as
    # a completed old run with a new checkpoint or history.
    for name in ('result.json', 'failure.json', 'history.json', 'checkpoint.pt',
                 'predictions.npz', 'config.json', 'split.json'):
        (path / name).unlink(missing_ok=True)
        (path / (name + '.tmp')).unlink(missing_ok=True)
