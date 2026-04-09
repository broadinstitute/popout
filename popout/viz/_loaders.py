"""Data loaders for popout output files.

All loaders are pure-numpy (no pandas dependency). The tract reader streams
records to handle files with tens of millions of rows without loading into
memory.
"""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Global ancestry TSV
# ---------------------------------------------------------------------------

class GlobalAncestry:
    """Per-sample global ancestry proportions."""

    __slots__ = ("sample_names", "proportions", "n_ancestries")

    def __init__(self, sample_names: list[str], proportions: np.ndarray):
        self.sample_names = sample_names
        self.proportions = proportions  # (n_samples, n_ancestries)
        self.n_ancestries = proportions.shape[1]


def read_global_tsv(path: str | Path) -> GlobalAncestry:
    """Read ``{prefix}.global.tsv`` into sample names + numpy array."""
    path = Path(path)
    sample_names: list[str] = []
    rows: list[list[float]] = []
    with open(path) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            sample_names.append(parts[0])
            rows.append([float(x) for x in parts[1:]])
    return GlobalAncestry(sample_names, np.array(rows, dtype=np.float32))


# ---------------------------------------------------------------------------
# Tracts TSV (streaming)
# ---------------------------------------------------------------------------

class Tract(NamedTuple):
    chrom: str
    start_bp: int
    end_bp: int
    sample: str
    haplotype: int
    ancestry: int
    n_sites: int
    mean_posterior: float  # NaN if not available


def read_tracts(
    path: str | Path,
    *,
    sample: str | None = None,
    chrom: str | None = None,
) -> Iterator[Tract]:
    """Stream tracts from ``{prefix}.tracts.tsv.gz``.

    Optionally filter to a specific sample and/or chromosome during read.
    """
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open

    with opener(path, "rt") as f:
        header = f.readline()
        has_posterior = "mean_posterior" in header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            t_chrom = parts[0]
            t_sample = parts[3]
            if sample is not None and t_sample != sample:
                continue
            if chrom is not None and t_chrom != chrom:
                continue
            post = float(parts[7]) if has_posterior and len(parts) > 7 else float("nan")
            yield Tract(
                chrom=t_chrom,
                start_bp=int(parts[1]),
                end_bp=int(parts[2]),
                sample=t_sample,
                haplotype=int(parts[4]),
                ancestry=int(parts[5]),
                n_sites=int(parts[6]),
                mean_posterior=post,
            )


def collect_tract_lengths_by_ancestry(
    path: str | Path,
    *,
    chrom: str | None = None,
) -> dict[int, list[float]]:
    """Collect tract lengths (Mb) grouped by ancestry via streaming read."""
    lengths: dict[int, list[float]] = {}
    for t in read_tracts(path, chrom=chrom):
        mb = (t.end_bp - t.start_bp) / 1e6
        lengths.setdefault(t.ancestry, []).append(mb)
    return lengths


def collect_sample_names_from_tracts(path: str | Path) -> list[str]:
    """Collect unique sample names from tracts file (preserving order)."""
    seen: set[str] = set()
    names: list[str] = []
    for t in read_tracts(path):
        if t.sample not in seen:
            seen.add(t.sample)
            names.append(t.sample)
    return names


# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------

def read_model_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``{prefix}.model.npz``."""
    return dict(np.load(path, allow_pickle=False))


def read_model_text(path: str | Path) -> dict[str, str | float | int]:
    """Read ``{prefix}.model`` human-readable file."""
    result: dict[str, str | float | int] = {}
    with open(path) as f:
        for line in f:
            key, val = line.strip().split("\t", 1)
            if key == "n_ancestries":
                result[key] = int(val)
            elif key == "gen_since_admix":
                result[key] = float(val)
            elif key in ("mu", "mismatch"):
                result[key] = [float(x) for x in val.split(",")]
            else:
                result[key] = val
    return result


# ---------------------------------------------------------------------------
# Summary JSON / stats JSONL
# ---------------------------------------------------------------------------

def read_summary(path: str | Path) -> dict:
    """Read ``{prefix}.summary.json``."""
    with open(path) as f:
        return json.load(f)


def read_stats_jsonl(path: str | Path) -> list[dict]:
    """Read ``{prefix}.stats.jsonl`` event log."""
    records: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Spectral NPZ (Phase 3 — optional)
# ---------------------------------------------------------------------------

def read_spectral_npz(path: str | Path) -> dict[str, np.ndarray] | None:
    """Read ``{prefix}.spectral.npz`` if it exists, else None."""
    path = Path(path)
    if not path.exists():
        return None
    return dict(np.load(path, allow_pickle=False))


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_files(prefix: str | Path) -> dict[str, Path]:
    """Discover which output files exist for a given prefix.

    Returns a dict mapping file type names to paths.
    """
    prefix = Path(prefix)
    candidates = {
        "global_tsv": prefix.with_name(prefix.name + ".global.tsv"),
        "tracts": prefix.with_name(prefix.name + ".tracts.tsv.gz"),
        "model": prefix.with_name(prefix.name + ".model"),
        "model_npz": prefix.with_name(prefix.name + ".model.npz"),
        "summary": prefix.with_name(prefix.name + ".summary.json"),
        "stats_jsonl": prefix.with_name(prefix.name + ".stats.jsonl"),
        "spectral_npz": prefix.with_name(prefix.name + ".spectral.npz"),
    }
    return {k: v for k, v in candidates.items() if v.exists()}
