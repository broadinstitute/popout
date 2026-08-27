"""Synthetic VCF fixtures for the extract_tract_events tests.

Every VCF written by this helper is uncompressed plain text. ``bcftools
view -h`` and ``bcftools query -f`` both accept plain VCFs, so we skip
bgzip + tabix to keep fixtures fast and byte-transparent (a test failure
prints readable text, not a hex dump).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest


def _write_vcf_impl(
    path: Path,
    *,
    panel_line: str,
    contigs: Sequence[tuple[str, int]],
    samples: Sequence[str],
    records: Sequence[tuple[str, int, Sequence[tuple[str, str]]]],
) -> Path:
    lines = ["##fileformat=VCFv4.2", panel_line]
    for cid, cl in contigs:
        lines.append(f"##contig=<ID={cid},length={cl},assembly=GRCh38>")
    lines.append(
        '##FORMAT=<ID=AN1,Number=1,Type=Integer,'
        'Description="Ancestry of first haplotype">'
    )
    lines.append(
        '##FORMAT=<ID=AN2,Number=1,Type=Integer,'
        'Description="Ancestry of second haplotype">'
    )
    lines.append(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
        + "\t".join(samples)
    )
    for chrom, pos, per_sample in records:
        row = [chrom, str(pos), ".", "A", "T", ".", "PASS", ".", "AN1:AN2"]
        for an1, an2 in per_sample:
            row.append(f"{an1}:{an2}")
        lines.append("\t".join(row))
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def write_vcf():
    return _write_vcf_impl
