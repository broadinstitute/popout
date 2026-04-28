"""Device memory budget helpers."""

import jax

_HEADROOM_BYTES = 12 * 1024**3  # 12 GB reserved for compute workspace

# Snapshot of free device memory taken at first call. Reusing this
# across the whole run keeps fits_on_device decisions stable even if
# other processes allocate/free GPU memory mid-run, which otherwise
# flips host/device dispatch branches and breaks reproducibility under
# a fixed --seed.
_cached_budget: int | None = None


def _query_device_free_bytes() -> int:
    try:
        stats = jax.devices()[0].memory_stats()
        return int(stats["bytes_limit"]) - int(stats["bytes_in_use"])
    except Exception:
        return 0


def device_free_bytes() -> int:
    """Snapshot of free device bytes from the first call.

    Subsequent calls return the same value so dispatch decisions are
    deterministic across runs.
    """
    global _cached_budget
    if _cached_budget is None:
        _cached_budget = _query_device_free_bytes()
    return _cached_budget


def fits_on_device(nbytes: int) -> bool:
    """True if an array of *nbytes* can live on device with headroom."""
    return nbytes + _HEADROOM_BYTES < device_free_bytes()
