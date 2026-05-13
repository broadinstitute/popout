"""Unit tests for workflows/lai-tools/scripts/generate_partitions.py.

Builds a tiny bgzipped+tabix-indexed VCF fixture via pysam, then exercises
the byte-balanced partitioner against it. Runs in <2s, offline, no Docker.

    pytest workflows/lai-tools/tests/
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pysam
import pytest

# Load the script as a module without making the directory a package.
# Register in sys.modules so dataclass introspection (which looks up
# cls.__module__ in sys.modules) works.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_partitions.py"
_spec = importlib.util.spec_from_file_location("generate_partitions", _SCRIPT)
gp = importlib.util.module_from_spec(_spec)
sys.modules["generate_partitions"] = gp
_spec.loader.exec_module(gp)


CONTIG_LENGTH = 50_000_000


def _build_fixture_vcf(path: Path, *, chroms: dict[str, list[int]]) -> Path:
    """Write a bgzipped phased VCF with the given chrom->positions and tabix-index it.

    Returns the path to the .vcf.gz; the .tbi lands next to it.
    """
    header = pysam.VariantHeader()
    header.add_meta("FILTER", items=[("ID", "PASS"), ("Description", "All filters passed")])
    header.add_meta("FORMAT", items=[("ID", "GT"), ("Number", "1"), ("Type", "String"), ("Description", "Genotype")])
    for c in chroms:
        header.contigs.add(c, length=CONTIG_LENGTH)
    header.add_sample("S1")

    with pysam.VariantFile(str(path), "wz", header=header) as vcf:
        for chrom, positions in chroms.items():
            for pos in positions:
                rec = vcf.new_record()
                rec.contig = chrom
                rec.pos = pos
                rec.ref = "A"
                rec.alts = ("G",)
                rec.qual = 100
                rec.samples["S1"]["GT"] = (0, 1)
                rec.samples["S1"].phased = True
                vcf.write(rec)

    pysam.tabix_index(str(path), preset="vcf", force=True)
    return path


@pytest.fixture
def uniform_fixture(tmp_path: Path) -> Path:
    """1000 records evenly spread across 1..50_000_000 on chr_uniform."""
    chroms = {
        "chr_uniform": [1 + i * 50_000 for i in range(1000)],
    }
    return _build_fixture_vcf(tmp_path / "uniform.vcf.gz", chroms=chroms)


@pytest.fixture
def two_chrom_fixture(tmp_path: Path) -> Path:
    """Two chroms, one denser than the other."""
    chroms = {
        "chr_dense":  [1 + i * 25_000 for i in range(2000)],   # 1 record per 25 kb
        "chr_sparse": [1 + i * 200_000 for i in range(250)],   # 1 record per 200 kb
    }
    return _build_fixture_vcf(tmp_path / "two.vcf.gz", chroms=chroms)


@pytest.fixture
def spike_fixture(tmp_path: Path) -> Path:
    """One chrom, mostly sparse but with a high-density "HLA-style" spike."""
    positions = []
    # Sparse stretch: 1 record per 50 kb across the first 20 Mb.
    positions += [1 + i * 50_000 for i in range(400)]
    # Spike: 500 records crammed into a 200 kb window in the middle.
    spike_start = 25_000_000
    positions += [spike_start + i * 400 for i in range(500)]
    # Sparse tail: same density as head.
    positions += [30_000_000 + i * 50_000 for i in range(400)]
    return _build_fixture_vcf(tmp_path / "spike.vcf.gz", chroms={"chr_spike": positions})


def test_parse_tbi_round_trips(uniform_fixture: Path):
    """parse_tbi recovers the contig list from a freshly written .tbi."""
    indexes = gp.parse_tbi(uniform_fixture.with_suffix(uniform_fixture.suffix + ".tbi"))
    assert list(indexes.keys()) == ["chr_uniform"]
    # 50 Mb of records with 16 kb linear bins → ~3050 entries (last entry may
    # be present even if empty; sparse trailing entries fold via copy-forward).
    assert 3000 <= len(indexes["chr_uniform"]) <= 3200


def test_uniform_density_yields_multiple_partitions(uniform_fixture: Path):
    tbi = uniform_fixture.with_suffix(uniform_fixture.suffix + ".tbi")
    indexes = gp.parse_tbi(tbi)
    total_bytes = indexes["chr_uniform"][-1] - indexes["chr_uniform"][0]
    target = max(100, total_bytes // 5)

    partitions = gp.generate_partitions(tbi_path=tbi, target_bytes=target, max_bytes=target * 5)

    # We expect roughly 5 partitions; allow a wide tolerance because trailing
    # bins can absorb into the last partition.
    assert 3 <= len(partitions) <= 7
    # All on the expected contig.
    assert {p.chrom for p in partitions} == {"chr_uniform"}
    # Coverage is monotone in BP within the contig: starts are non-overlapping
    # and end >= start for each partition.
    for p in partitions:
        assert p.start >= 1
        assert p.end >= p.start


def test_largest_first_ordering(two_chrom_fixture: Path):
    tbi = two_chrom_fixture.with_suffix(two_chrom_fixture.suffix + ".tbi")
    partitions = gp.generate_partitions(
        tbi_path=tbi,
        target_bytes=500,        # small target → multiple partitions
        max_bytes=500 * 100,
    )
    assert len(partitions) >= 2
    # Sorted descending by estimated_bytes.
    sizes = [p.estimated_bytes for p in partitions]
    assert sizes == sorted(sizes, reverse=True)


def test_chromosome_filter(two_chrom_fixture: Path):
    tbi = two_chrom_fixture.with_suffix(two_chrom_fixture.suffix + ".tbi")
    only_sparse = gp.generate_partitions(
        tbi_path=tbi,
        target_bytes=500,
        max_bytes=500 * 100,
        chromosomes=["chr_sparse"],
    )
    assert {p.chrom for p in only_sparse} == {"chr_sparse"}


def test_unknown_chromosome_raises(uniform_fixture: Path):
    tbi = uniform_fixture.with_suffix(uniform_fixture.suffix + ".tbi")
    with pytest.raises(ValueError, match="not present"):
        gp.generate_partitions(
            tbi_path=tbi,
            target_bytes=1000,
            max_bytes=10000,
            chromosomes=["chr_does_not_exist"],
        )


def test_density_spike_gets_smaller_bp_partition(spike_fixture: Path):
    """HLA-style density spike → its partition should be much smaller in bp
    than uniform-density partitions but carry similar bytes."""
    tbi = spike_fixture.with_suffix(spike_fixture.suffix + ".tbi")
    indexes = gp.parse_tbi(tbi)
    total_bytes = indexes["chr_spike"][-1] - indexes["chr_spike"][0]

    # Aim for ~6 partitions so the spike isn't lost in noise.
    partitions = gp.generate_partitions(
        tbi_path=tbi,
        target_bytes=total_bytes // 6,
        max_bytes=total_bytes,  # don't trigger subdivision
    )
    assert len(partitions) >= 3

    spans_bp = [p.end - p.start + 1 for p in partitions]
    median_span = sorted(spans_bp)[len(spans_bp) // 2]
    smallest_span = min(spans_bp)
    # Spike partition should be at least 4x narrower in bp than the median.
    assert smallest_span * 4 <= median_span, (
        f"smallest bp span {smallest_span} not noticeably narrower than median {median_span}; "
        f"density-aware partitioning may be broken"
    )


def test_max_bytes_safety_valve_subdivides(uniform_fixture: Path):
    """Force max_bytes < target_bytes-equivalent via a pathological cap;
    the algorithm should still respect max_bytes by subdividing."""
    tbi = uniform_fixture.with_suffix(uniform_fixture.suffix + ".tbi")
    indexes = gp.parse_tbi(tbi)
    total_bytes = indexes["chr_uniform"][-1] - indexes["chr_uniform"][0]

    # Target is the whole contig; max cap is a tenth of that.
    target = total_bytes
    max_cap = total_bytes // 10
    partitions = gp.generate_partitions(tbi_path=tbi, target_bytes=target, max_bytes=max_cap)

    # All partitions must be <= max_cap.
    for p in partitions:
        assert p.estimated_bytes <= max_cap, (
            f"{p.partition_id} estimated_bytes={p.estimated_bytes} exceeds max_cap={max_cap}"
        )


def test_target_larger_than_total_yields_one_partition(uniform_fixture: Path):
    tbi = uniform_fixture.with_suffix(uniform_fixture.suffix + ".tbi")
    indexes = gp.parse_tbi(tbi)
    total_bytes = indexes["chr_uniform"][-1] - indexes["chr_uniform"][0]

    partitions = gp.generate_partitions(
        tbi_path=tbi,
        target_bytes=total_bytes * 100,
        max_bytes=total_bytes * 1000,
    )
    assert len(partitions) == 1
    assert partitions[0].chrom == "chr_uniform"


def test_manifest_write_format(uniform_fixture: Path, tmp_path: Path):
    tbi = uniform_fixture.with_suffix(uniform_fixture.suffix + ".tbi")
    partitions = gp.generate_partitions(tbi_path=tbi, target_bytes=1000, max_bytes=100000)

    manifest = tmp_path / "partitions.tsv"
    regions = tmp_path / "regions.txt"
    region_ids = tmp_path / "region_ids.txt"
    gp.write_outputs(partitions, manifest, regions, region_ids)

    manifest_lines = manifest.read_text().splitlines()
    assert manifest_lines[0] == "partition_id\tchrom\tstart\tend\testimated_bytes"
    assert len(manifest_lines) == 1 + len(partitions)
    # regions / region_ids match manifest body line count.
    assert len(regions.read_text().splitlines()) == len(partitions)
    assert len(region_ids.read_text().splitlines()) == len(partitions)
    # First region line corresponds to the largest partition.
    first_region = regions.read_text().splitlines()[0]
    assert first_region.startswith(partitions[0].chrom + ":")
    assert first_region.endswith(str(partitions[0].end))
