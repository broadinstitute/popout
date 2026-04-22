"""Ancestry name resolution for output headers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def parse_ancestry_names(spec: Optional[str], K: int) -> list[str]:
    """Parse --ancestry-names into a list of K strings.

    Parameters
    ----------
    spec : None, comma-separated string, or path to a single-column TSV
    K : expected number of ancestries

    Returns
    -------
    List of K ancestry name strings.
    """
    if spec is None:
        return [f"anc_{i}" for i in range(K)]
    p = Path(spec)
    if p.exists() and p.is_file():
        names = [line.strip() for line in p.read_text().splitlines() if line.strip()]
    else:
        names = [n.strip() for n in spec.split(",")]
    if len(names) != K:
        raise ValueError(f"--ancestry-names has {len(names)} entries, need {K}")
    return names
