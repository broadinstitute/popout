"""Warn when numpy operations produce large transient copies.

Enable by setting POPOUT_WARN_LARGE_COPY=<threshold_gb> (e.g. ``1``).
When enabled, ``check_no_copy`` logs a warning if an output array
occupies a different buffer than its input and exceeds the threshold.
Zero cost when disabled (default).
"""
from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger("popout.memcheck")

_THRESHOLD_GB = float(os.environ.get("POPOUT_WARN_LARGE_COPY", "0"))
_ENABLED = _THRESHOLD_GB > 0


def check_no_copy(name: str, before: np.ndarray, after: np.ndarray) -> None:
    """Log a warning if *after* is a different buffer than *before* and large."""
    if not _ENABLED:
        return
    if before is after:
        return
    if before.ctypes.data == after.ctypes.data:
        return
    nbytes = after.nbytes
    if nbytes / 1e9 >= _THRESHOLD_GB:
        log.warning(
            "%s: %s array copied (%.1f GB). Likely a redundant "
            "np.asarray/astype/np.array call on an already-conforming input.",
            name, after.shape, nbytes / 1e9,
        )
