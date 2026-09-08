"""Randomness and provenance used by every experiment implementation."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import random
import subprocess

import numpy as np
import torch


def seed_everything(seed, deterministic=True, threads=1):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def provenance():
    root = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted((root / 'agfl').rglob('*.py')):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    versions = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "mne", "wfdb"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "git_revision": revision, "source_sha256": h.hexdigest(),
            "environment": {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']},
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
