"""Small TSV / formatting helpers used by chart functions and sections."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any, Iterable


def read_tsv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    """Return ``(header, rows)`` from a TSV (or gzipped TSV).

    Empty / missing file returns ``([], [])`` — sections gate on this.
    """
    p = Path(path)
    if not p.exists():
        return [], []
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt") as f:
        text = f.read().rstrip("\n")
    if not text:
        return [], []
    lines = text.split("\n")
    return lines[0].split("\t"), [ln.split("\t") for ln in lines[1:]]


def to_float(s: Any) -> float | None:
    if s is None:
        return None
    if isinstance(s, (int, float)):
        f = float(s)
        return None if f != f else f
    s = str(s).strip()
    if s in ("", "NA", "nan", "NaN"):
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return None if f != f else f


def fmt_num(v: Any, places: int = 3) -> str:
    f = to_float(v)
    if f is None:
        return "—"
    return f"{f:.{places}f}"


def fmt_int(v: Any) -> str:
    f = to_float(v)
    if f is None:
        return "—"
    return f"{int(f):,}"


def fmt_pct(v: Any, places: int = 1, *, scale: float = 1.0) -> str:
    """Render *v* as a percentage. ``scale=1.0`` treats *v* as a 0-1
    fraction; ``scale=0.01`` treats it as already-in-percent."""
    f = to_float(v)
    if f is None:
        return "—"
    return f"{(f / scale):.{places}f}%"


def md_escape(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("_", "\\_")
