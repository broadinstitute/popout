"""Shared helpers for the AIM panel build pipeline.

Owns the candidate-row dataclass and TSV I/O helpers that are reused
across the scan and build stages. The 1KG superpop frequency TSV
schema (``#chrom pos ref alt EUR EAS AMR AFR SAS``) is the single
source of truth — this module does not introduce any other reference.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# Order matches the 1KG TSV header columns 5..9 ; do not reorder.
SUPERPOPS_1KG: tuple[str, ...] = ("EUR", "EAS", "AMR", "AFR", "SAS")

# AoU superpopulation labels; MID is included even though 1KG does not
# cover it because the priors framework targets MID via AIM panel only.
SUPERPOPS_AOU: tuple[str, ...] = ("AFR", "EUR", "EAS", "AMR", "SAS", "MID")


@dataclass(frozen=True)
class FreqRow:
    """One site from the 1KG superpop TSV, as parsed values."""

    chrom: str
    pos: int
    ref: str
    alt: str
    eur: float
    eas: float
    amr: float
    afr: float
    sas: float

    def freq(self, pop: str) -> float:
        return {
            "EUR": self.eur, "EAS": self.eas, "AMR": self.amr,
            "AFR": self.afr, "SAS": self.sas,
        }[pop]

    @classmethod
    def from_row(cls, row: list[str]) -> "FreqRow":
        return cls(
            chrom=row[0],
            pos=int(row[1]),
            ref=row[2],
            alt=row[3],
            eur=float(row[4]),
            eas=float(row[5]),
            amr=float(row[6]),
            afr=float(row[7]),
            sas=float(row[8]),
        )


def iter_1kg_rows(ref_path: Path) -> Iterator[FreqRow]:
    """Stream parsed rows from the 1KG TSV (gzipped or plain)."""
    opener = gzip.open if ref_path.suffix == ".gz" else open
    with opener(ref_path, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            yield FreqRow.from_row(row)


@dataclass
class Candidate:
    """One AIM candidate awaiting downstream phases."""

    chrom: str
    pos: int
    ref: str
    alt: str
    expected_freq: float           # alt-allele freq in the target pop
    target_pop: str                # AFR/EUR/EAS/AMR/SAS/MID
    others_max_freq: float         # max alt-allele freq in any other AoU pop
    separation: float              # |expected_freq - others_max_freq|
    direction: str                 # "+" if target > others, "-" if target < others
    pops_freqs: dict[str, float] = field(default_factory=dict)
    rationale: str = ""
    source: str = "1KG_Phase3"
    locus: str = ""                # gene/locus annotation when known


_CANDIDATE_COLS = [
    "chrom", "pos_bp", "ref", "alt", "target_pop", "expected_freq",
    "others_max_freq", "separation", "direction",
    "freq_eur", "freq_eas", "freq_amr", "freq_afr", "freq_sas",
    "locus", "source", "rationale",
]


def write_candidates(path: Path, candidates: list[Candidate]) -> None:
    """Write a candidate TSV (one row per candidate)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(_CANDIDATE_COLS) + "\n")
        for c in candidates:
            row = [
                c.chrom, str(c.pos), c.ref, c.alt, c.target_pop,
                f"{c.expected_freq:.4f}", f"{c.others_max_freq:.4f}",
                f"{c.separation:.4f}", c.direction,
                f"{c.pops_freqs.get('EUR', float('nan')):.4f}",
                f"{c.pops_freqs.get('EAS', float('nan')):.4f}",
                f"{c.pops_freqs.get('AMR', float('nan')):.4f}",
                f"{c.pops_freqs.get('AFR', float('nan')):.4f}",
                f"{c.pops_freqs.get('SAS', float('nan')):.4f}",
                c.locus, c.source, c.rationale,
            ]
            f.write("\t".join(row) + "\n")


def read_candidates(path: Path) -> list[Candidate]:
    """Read a candidate TSV back into Candidate objects."""
    out: list[Candidate] = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {h: i for i, h in enumerate(header)}
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts or not parts[0]:
                continue
            pops_freqs = {}
            for pop, key in (
                ("EUR", "freq_eur"), ("EAS", "freq_eas"),
                ("AMR", "freq_amr"), ("AFR", "freq_afr"),
                ("SAS", "freq_sas"),
            ):
                if key in idx:
                    try:
                        pops_freqs[pop] = float(parts[idx[key]])
                    except ValueError:
                        pass
            out.append(Candidate(
                chrom=parts[idx["chrom"]],
                pos=int(parts[idx["pos_bp"]]),
                ref=parts[idx["ref"]],
                alt=parts[idx["alt"]],
                expected_freq=float(parts[idx["expected_freq"]]),
                target_pop=parts[idx["target_pop"]],
                others_max_freq=float(parts[idx["others_max_freq"]]),
                separation=float(parts[idx["separation"]]),
                direction=parts[idx["direction"]],
                pops_freqs=pops_freqs,
                locus=parts[idx["locus"]] if "locus" in idx else "",
                source=parts[idx["source"]] if "source" in idx else "1KG_Phase3",
                rationale=parts[idx["rationale"]] if "rationale" in idx else "",
            ))
    return out


_PANEL_COLS = [
    "chrom", "pos_bp", "ref", "alt", "expected_freq", "weight",
    "source", "locus", "rationale",
]


def write_panel(path: Path, candidates: list[Candidate], pop: str) -> None:
    """Write the production panel TSV in AIMPanel.from_tsv schema.

    Required columns (consumed by the loader): chrom, pos_bp,
    expected_freq, weight. Extras (ref, alt, source, locus, rationale)
    are stored for documentation; the loader ignores them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(_PANEL_COLS) + "\n")
        for c in candidates:
            weight = _weight_for_separation(c.separation)
            f.write("\t".join([
                c.chrom, str(c.pos), c.ref, c.alt,
                f"{c.expected_freq:.4f}", f"{weight:.2f}",
                c.source, c.locus, c.rationale,
            ]) + "\n")


def _weight_for_separation(sep: float) -> float:
    """Map frequency separation to a per-marker reliability weight in [0, 1].

    The framework's variance-normalized scoring already de-weights
    low-information markers, so this map is conservative — anchor
    weight 1.0 for sep ≥ 0.7, scaling down to 0.4 at the 0.40 cutoff.
    """
    if sep >= 0.70:
        return 1.0
    if sep >= 0.55:
        return 0.85
    if sep >= 0.40:
        return 0.65
    return 0.40


@dataclass
class BuildLogEntry:
    """One row of the build log capturing a candidate's fate."""

    chrom: str
    pos: int
    ref: str
    alt: str
    target_pop: str
    expected_freq: float
    others_max_freq: float
    separation: float
    stage: str        # "selected" | "linkage_pruned" | "below_threshold"
    reason: str


_BUILD_LOG_COLS = [
    "chrom", "pos_bp", "ref", "alt", "target_pop", "expected_freq",
    "others_max_freq", "separation", "stage", "reason",
]


def write_build_log(path: Path, entries: list[BuildLogEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\t".join(_BUILD_LOG_COLS) + "\n")
        for e in entries:
            f.write("\t".join([
                e.chrom, str(e.pos), e.ref, e.alt, e.target_pop,
                f"{e.expected_freq:.4f}", f"{e.others_max_freq:.4f}",
                f"{e.separation:.4f}", e.stage, e.reason,
            ]) + "\n")
