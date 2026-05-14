"""Unit tests for workflows/lai-tools/scripts/generate_partitions.py.

Two strategies:
  - One test builds a real .tbi via pysam to verify parse_tbi round-trips.
  - The rest construct synthetic linear-index arrays directly so we can
    test the partitioning algorithm against precise voff values without
    being at the mercy of bgzf block boundaries in tiny fixture files.

Runs in <2s, offline, no Docker.

    pytest workflows/lai-tools/tests/
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pysam
import pytest

# Load the script as a module. Register in sys.modules so dataclass
# introspection works.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_partitions.py"
_spec = importlib.util.spec_from_file_location("generate_partitions", _SCRIPT)
gp = importlib.util.module_from_spec(_spec)
sys.modules["generate_partitions"] = gp
_spec.loader.exec_module(gp)


# ---------------------------------------------------------------------
# Helpers to build synthetic linear indexes.
#
# A voff packs (file_offset << 16) | within_block_offset. For partitioning
# the only signal that matters at biobank scale is the file_offset diff,
# so these helpers build linear indexes with controllable file_offset
# spacing per bin.
# ---------------------------------------------------------------------

def _voff(file_offset: int, within_block: int = 0) -> int:
    return (file_offset << 16) | within_block


def _uniform_index(*, n_bins: int, bytes_per_bin: int, file_start: int = 0) -> list[int]:
    """Linear index with uniform bytes_per_bin across n_bins."""
    return [_voff(file_start + i * bytes_per_bin) for i in range(n_bins + 1)][:n_bins]


def _spike_index(*, n_bins: int, normal_bytes: int, dense_bin: int, dense_bytes: int) -> list[int]:
    """Linear index with uniform spacing except one bin that has a larger byte span
    (HLA-style hotspot — small bp, large bytes).

    `dense_bin` is the bin that contains many records (its byte span is
    `dense_bytes`; the byte span of any other bin is `normal_bytes`).
    Per the tabix linear index, bin K's byte span is foffs[K+1] - foffs[K],
    so the dense_bin's delta is applied to the transition i = dense_bin + 1.
    """
    foffs = [0]
    for i in range(1, n_bins + 1):  # need n_bins+1 entries to have spans for all n_bins bins
        delta = dense_bytes if (i - 1) == dense_bin else normal_bytes
        foffs.append(foffs[-1] + delta)
    return [_voff(f) for f in foffs[:n_bins]]


# ---------------------------------------------------------------------
# parse_tbi round-trip (this still needs a real fixture)
# ---------------------------------------------------------------------

@pytest.fixture
def small_tbi(tmp_path: Path) -> Path:
    """1000 records on a single contig; we just want parse_tbi to read its .tbi."""
    header = pysam.VariantHeader()
    header.add_meta("FILTER", items=[("ID", "PASS"), ("Description", "ok")])
    header.add_meta("FORMAT", items=[("ID", "GT"), ("Number", "1"), ("Type", "String"), ("Description", "Genotype")])
    header.contigs.add("chr_tiny", length=50_000_000)
    header.add_sample("S1")

    vcf_path = tmp_path / "tiny.vcf.gz"
    with pysam.VariantFile(str(vcf_path), "wz", header=header) as vcf:
        for i in range(1000):
            rec = vcf.new_record()
            rec.contig = "chr_tiny"
            rec.pos = 1 + i * 50_000
            rec.ref = "A"
            rec.alts = ("G",)
            rec.samples["S1"]["GT"] = (0, 1)
            rec.samples["S1"].phased = True
            vcf.write(rec)

    pysam.tabix_index(str(vcf_path), preset="vcf", force=True)
    return Path(str(vcf_path) + ".tbi")


def test_parse_tbi_round_trips(small_tbi: Path):
    indexes = gp.parse_tbi(small_tbi)
    assert list(indexes.keys()) == ["chr_tiny"]
    assert len(indexes["chr_tiny"]) > 0


# ---------------------------------------------------------------------
# Partition algorithm — tested with synthetic linear indexes
# ---------------------------------------------------------------------

def test_uniform_density_yields_target_partition_count():
    # 1000 bins × 1 MB each = 1 GB total. target = 100 MB → 10 partitions.
    idx = _uniform_index(n_bins=1000, bytes_per_bin=1_000_000)
    parts, _ = gp.partition_chrom("chr1", idx, target_bytes=100_000_000, max_bytes=10**12)
    assert 9 <= len(parts) <= 11


def test_largest_first_ordering():
    # Three contigs of different sizes — confirm cross-contig partitions are
    # sorted by estimated_bytes desc.
    big   = _uniform_index(n_bins=200, bytes_per_bin=1_000_000)  # 200 MB
    med   = _uniform_index(n_bins=50,  bytes_per_bin=1_000_000)  # 50 MB
    small = _uniform_index(n_bins=10,  bytes_per_bin=1_000_000)  # 10 MB
    parts = []
    next_idx = 1
    for chrom, vidx in [("chr_big", big), ("chr_med", med), ("chr_small", small)]:
        p, next_idx = gp.partition_chrom(chrom, vidx, target_bytes=20_000_000, max_bytes=10**12, starting_partition_idx=next_idx)
        parts.extend(p)
    parts.sort(key=lambda p: -p.estimated_bytes)
    sizes = [p.estimated_bytes for p in parts]
    assert sizes == sorted(sizes, reverse=True)


def test_density_spike_gets_smaller_bp_partition():
    # 100 bins with 1 MB each, except bin 50 has a 100 MB spike — this
    # mimics HLA: same bp width but much higher record density. With
    # target=5 MB normal partitions span 5 bins; the spike bin alone
    # exceeds target so it becomes its own 1-bin partition.
    idx = _spike_index(n_bins=100, normal_bytes=1_000_000, dense_bin=50, dense_bytes=100_000_000)
    parts, _ = gp.partition_chrom("chr_spike", idx, target_bytes=5_000_000, max_bytes=10**12)
    assert len(parts) >= 10
    spans_bp = [p.end - p.start + 1 for p in parts]
    median = sorted(spans_bp)[len(spans_bp) // 2]
    smallest = min(spans_bp)
    # Spike partition should be at least 4× narrower in bp than the median.
    assert smallest * 4 <= median, (
        f"smallest bp span {smallest} not noticeably narrower than median {median}; "
        f"density-aware partitioning may be broken"
    )


def test_max_bytes_safety_valve_subdivides():
    # 100 bins × 10 MB each = 1 GB total. target=100 MB, max=50 MB →
    # whenever a 100 MB partition would form, it exceeds max and gets
    # subdivided BP-uniformly. Each sub-partition must respect max_bytes.
    idx = _uniform_index(n_bins=100, bytes_per_bin=10_000_000)
    parts, _ = gp.partition_chrom("chr_sub", idx, target_bytes=100_000_000, max_bytes=50_000_000)
    assert len(parts) > 0
    for p in parts:
        assert p.estimated_bytes <= 50_000_000, (
            f"{p.partition_id} estimated_bytes={p.estimated_bytes} exceeds max"
        )


def test_target_larger_than_total_yields_one_partition():
    idx = _uniform_index(n_bins=100, bytes_per_bin=1_000_000)  # 100 MB total
    parts, _ = gp.partition_chrom("chr_one", idx, target_bytes=10_000_000_000, max_bytes=10**13)
    assert len(parts) == 1


def test_single_block_emits_one_partition():
    """All bins point to the same file offset (file fits in one bgzf block).
    There's nothing to scatter — algorithm should emit one partition for the
    whole contig, not zero and not 1-per-bin."""
    n_bins = 100
    idx = [_voff(0, within) for within in range(n_bins)]  # all in block 0
    parts, _ = gp.partition_chrom("chr_block", idx, target_bytes=1_000_000, max_bytes=10**9)
    assert len(parts) == 1
    assert parts[0].chrom == "chr_block"


def test_empty_index_emits_nothing():
    parts, _ = gp.partition_chrom("chr_empty", [], target_bytes=1000, max_bytes=10000)
    assert parts == []


# ---------------------------------------------------------------------
# generate_partitions top-level + manifest writer
# ---------------------------------------------------------------------

def test_unknown_chromosome_raises(small_tbi: Path):
    with pytest.raises(ValueError, match="not present"):
        gp.generate_partitions(
            tbi_path=small_tbi,
            target_bytes=1000,
            max_bytes=10000,
            chromosomes=["chr_does_not_exist"],
        )


def test_manifest_write_format(tmp_path: Path):
    idx = _uniform_index(n_bins=50, bytes_per_bin=1_000_000)
    parts, _ = gp.partition_chrom("chr1", idx, target_bytes=10_000_000, max_bytes=10**12)
    assert len(parts) >= 2

    manifest = tmp_path / "partitions.tsv"
    regions = tmp_path / "regions.txt"
    region_ids = tmp_path / "region_ids.txt"
    gp.write_outputs(parts, manifest, regions, region_ids)

    manifest_lines = manifest.read_text().splitlines()
    assert manifest_lines[0] == "partition_id\tchrom\tstart\tend\testimated_bytes"
    assert len(manifest_lines) == 1 + len(parts)
    assert len(regions.read_text().splitlines()) == len(parts)
    assert len(region_ids.read_text().splitlines()) == len(parts)
    first_region = regions.read_text().splitlines()[0]
    assert first_region.startswith(parts[0].chrom + ":")
    assert first_region.endswith(str(parts[0].end))
