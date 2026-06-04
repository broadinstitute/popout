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


def n_weighted_mean(pairs: Iterable[tuple[float, float]]) -> float | None:
    """``[(weight, value), ...]`` → n-weighted mean. Drops NaN values / zero
    weights. Returns None if no usable data."""
    num = 0.0
    den = 0.0
    for w, v in pairs:
        if v != v or w != w or w <= 0:
            continue
        num += w * v
        den += w
    return num / den if den > 0 else None


def overlay_ticks(
    ax, y_center: float, values: Iterable[float], *,
    color: str = "#222", tick_height: float = 0.32,
    alpha: float = 0.95, lw: float = 1.5,
) -> None:
    """Draw short vertical tick marks at each value on a horizontal-bar
    chart — used to overlay per-cluster values on top of a cohort-pooled
    bar."""
    import numpy as np

    arr = np.array([v for v in values if v == v], dtype=float)
    if arr.size == 0:
        return
    y0 = y_center - tick_height / 2
    y1 = y_center + tick_height / 2
    ax.vlines(arr, y0, y1, colors=color, alpha=alpha, linewidth=lw, zorder=5)


def topn(rows: Iterable[tuple[str, float]], n: int = 3,
         *, reverse: bool = True) -> list[tuple[str, float]]:
    """Top-n (lab, value) by descending value (ascending when reverse=False)."""
    clean = [(lab, v) for lab, v in rows if v is not None and v == v]
    return sorted(clean, key=lambda r: r[1], reverse=reverse)[:n]
