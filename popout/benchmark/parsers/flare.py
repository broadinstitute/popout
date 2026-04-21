"""Parser for FLARE .anc.vcf.gz output."""

from __future__ import annotations

import gzip
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from popout.benchmark.common import MISSING_LABEL, TractSet, load_ancestry_header


def parse_flare(
    vcf_path: str | Path,
    global_path: Optional[str | Path] = None,
    chrom: Optional[str] = None,
) -> TractSet:
    """Parse a FLARE .anc.vcf.gz into a TractSet.

    If global_path is provided, cross-check the per-site calls against the
    per-sample globals as a sanity check (warn if they diverge by more than
    5pp on any ancestry).
    """
    vcf_path = Path(vcf_path)
    label_map: dict[int, str] = {}
    sample_ids: list[str] = []
    positions: list[int] = []
    rows: list[list[int]] = []  # each row is (2*n_samples,) AN1/AN2 values
    file_chrom: Optional[str] = None

    opener = gzip.open if vcf_path.suffix == ".gz" else open
    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("##ANCESTRY="):
                label_map = load_ancestry_header(line.strip())
                continue
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                parts = line.strip().split("\t")
                sample_ids = parts[9:]
                continue
            # Data line
            parts = line.strip().split("\t")
            row_chrom = parts[0]
            if chrom is not None and row_chrom != chrom:
                continue
            if file_chrom is None:
                file_chrom = row_chrom
            elif row_chrom != file_chrom and chrom is None:
                raise ValueError(
                    f"Multiple chromosomes in VCF ({file_chrom}, {row_chrom}). "
                    f"Specify --chrom to filter."
                )
            pos = int(parts[1])
            positions.append(pos)
            row_calls = []
            for sample_field in parts[9:]:
                fields = sample_field.split(":")
                an1_str = fields[1] if len(fields) > 1 else "."
                an2_str = fields[2] if len(fields) > 2 else "."
                an1 = MISSING_LABEL if an1_str == "." else int(an1_str)
                an2 = MISSING_LABEL if an2_str == "." else int(an2_str)
                row_calls.append(an1)
                row_calls.append(an2)
            rows.append(row_calls)

    if not label_map:
        raise ValueError("No ##ANCESTRY= header found in VCF")
    if not sample_ids:
        raise ValueError("No #CHROM header line found in VCF")
    if not rows:
        raise ValueError("No data rows found in VCF")

    n_samples = len(sample_ids)
    n_sites = len(rows)
    # calls shape: (2*n_samples, n_sites)
    calls = np.array(rows, dtype=np.uint16).T  # (2*n_samples, n_sites)
    # Reshape: rows interleave AN1, AN2 for each sample
    # Current layout: [S0_AN1, S0_AN2, S1_AN1, S1_AN2, ...]
    # This is already correct — sample i has haps at 2*i and 2*i+1

    hap_ids = np.array(
        [f"{s}:{h}" for s in sample_ids for h in (0, 1)],
        dtype=object,
    )
    site_positions = np.array(positions, dtype=np.int64)

    ts = TractSet(
        tool_name="flare",
        chrom=file_chrom or chrom or "unknown",
        hap_ids=hap_ids,
        site_positions=site_positions,
        calls=calls,
        label_map=label_map,
    )
    ts.validate()

    if global_path is not None:
        _crosscheck_global(ts, Path(global_path), sample_ids)

    return ts


def _crosscheck_global(
    ts: TractSet, global_path: Path, sample_ids: list[str]
) -> None:
    """Warn if per-site calls diverge from global ancestry file."""
    opener = gzip.open if global_path.suffix == ".gz" else open
    with opener(global_path, "rt") as f:
        header = f.readline().strip().split("\t")
        anc_names = header[1:]
        global_fracs: dict[str, dict[str, float]] = {}
        for line in f:
            parts = line.strip().split("\t")
            sample = parts[0]
            fracs = {anc_names[i]: float(parts[i + 1]) for i in range(len(anc_names))}
            global_fracs[sample] = fracs

    # Compute per-sample fractions from VCF calls (average of both haplotypes)
    for idx, sample in enumerate(sample_ids):
        if sample not in global_fracs:
            continue
        hap0_calls = ts.calls[2 * idx]
        hap1_calls = ts.calls[2 * idx + 1]
        for code, name in ts.label_map.items():
            if code == MISSING_LABEL:
                continue
            vcf_frac = ((hap0_calls == code).mean() + (hap1_calls == code).mean()) / 2
            global_frac = global_fracs[sample].get(name, 0.0)
            if abs(vcf_frac - global_frac) > 0.05:
                warnings.warn(
                    f"FLARE global mismatch: sample={sample} ancestry={name} "
                    f"vcf_frac={vcf_frac:.3f} global_frac={global_frac:.3f}"
                )
