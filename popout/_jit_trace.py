"""Opt-in tracing for JIT-compiled function input sizes.

Enable by setting POPOUT_TRACE_JIT=1. When enabled, ``traced_jit``
wraps ``jax.jit`` so that every call logs the total byte size of its
input arguments at INFO level. This surfaces accidental biobank-scale
tensors entering a JIT region.

When disabled (default), ``traced_jit`` is a zero-overhead alias for
``jax.jit``.
"""
from __future__ import annotations

import functools
import logging
import os

import jax

log = logging.getLogger("popout.jit_trace")

_ENABLED = os.environ.get("POPOUT_TRACE_JIT", "0") == "1"


def traced_jit(fn=None, **jit_kwargs):
    """Drop-in replacement for ``jax.jit`` that logs input sizes.

    Usage::

        @traced_jit
        def my_func(x, y): ...

        @traced_jit(static_argnums=(1,))
        def my_func(x, n): ...
    """
    if not _ENABLED:
        if fn is not None:
            return jax.jit(fn, **jit_kwargs)
        return lambda f: jax.jit(f, **jit_kwargs)

    def _decorator(f):
        jitted = jax.jit(f, **jit_kwargs)

        @functools.wraps(f)
        def _wrapper(*args, **kwargs):
            total = 0
            for a in args:
                if hasattr(a, "nbytes"):
                    total += int(a.nbytes)
            for v in kwargs.values():
                if hasattr(v, "nbytes"):
                    total += int(v.nbytes)
            log.info("jit %s: input bytes=%.3f GB",
                     getattr(f, "__name__", "<anon>"), total / 1e9)
            return jitted(*args, **kwargs)

        return _wrapper

    if fn is not None:
        return _decorator(fn)
    return _decorator
