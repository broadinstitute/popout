"""Atomic file writers.

Vestigial module: previously housed the WorkDir/manifest stage-resume
machinery, which was retired in favor of the train/infer scatter shape
(see ``popout.orchestrate``). The atomic NPZ/NPY writers are kept here
because they are the right primitive for any future model serializer
that needs crash-safe writes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def atomic_write_npz(dest: Path | str, save_dict: dict[str, Any]) -> None:
    """Write a compressed .npz atomically (tmpfile + rename).

    Uses a ``.tmp.npz`` suffix so ``np.savez_compressed`` does not append
    an extra ``.npz``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp.npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp, **save_dict)
        os.rename(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_npy(dest: Path | str, array) -> None:
    """Write a single array as .npy atomically.

    Uses a ``.tmp.npy`` suffix so ``np.save`` does not append an extra
    ``.npy``.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".tmp.npy")
    os.close(fd)
    try:
        np.save(tmp, array)
        os.rename(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
