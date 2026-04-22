"""Device memory budget helpers."""

import jax

_HEADROOM_BYTES = 12 * 1024**3  # 12 GB reserved for compute workspace


def device_free_bytes() -> int:
    """Bytes free on the default JAX device, or a pessimistic estimate."""
    try:
        stats = jax.devices()[0].memory_stats()
        return int(stats["bytes_limit"]) - int(stats["bytes_in_use"])
    except Exception:
        return 0  # assume nothing free → force host path


def fits_on_device(nbytes: int) -> bool:
    """True if an array of *nbytes* can live on device with headroom."""
    return nbytes + _HEADROOM_BYTES < device_free_bytes()
