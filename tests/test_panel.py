"""Tests for reference panel extraction (popout.panel)."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import jax.numpy as jnp

from popout.datatypes import AncestryModel, AncestryResult, ChromData
from popout.panel import (
    PanelConfig,
    WholeHapExtraction,
    Segment,
    extract_whole_haplotypes,
    extract_segments,
    write_panel_haplotypes,
    write_panel_segments,
    write_allele_frequencies,
    write_haplotype_proportions,
    export_panel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(posteriors: np.ndarray, chrom: str = "1") -> AncestryResult:
    """Build an AncestryResult from a numpy posteriors array."""
    H, T, A = posteriors.shape
    gamma = jnp.array(posteriors, dtype=jnp.float32)
    calls = np.array(jnp.argmax(gamma, axis=2), dtype=np.int8)
    model = AncestryModel(
        n_ancestries=A,
        mu=jnp.ones(A) / A,
        gen_since_admix=20.0,
        allele_freq=jnp.full((A, T), 0.5),
    )
    return AncestryResult(posteriors=gamma, calls=calls, model=model, chrom=chrom)


def _make_chromdata(n_haps: int, n_sites: int, chrom: str = "1") -> ChromData:
    """Build a ChromData with uniform spacing."""
    return ChromData(
        geno=np.zeros((n_haps, n_sites), dtype=np.uint8),
        pos_bp=np.arange(n_sites, dtype=np.int64) * 1000 + 1_000_000,
        pos_cm=np.linspace(0.0, 100.0, n_sites),
        chrom=chrom,
    )


# ---------------------------------------------------------------------------
# Whole-haplotype extraction
# ---------------------------------------------------------------------------

class TestWholeHapExtraction:

    def test_unadmixed_all_pass(self):
        """All haplotypes pass when posteriors are near 1.0."""
        H, T, A = 10, 100, 3
        # Haplotype i assigned to ancestry i % A with high confidence
        posteriors = np.full((H, T, A), 0.001)
        for h in range(H):
            a = h % A
            posteriors[h, :, a] = 0.99
        # Normalize
        posteriors /= posteriors.sum(axis=2, keepdims=True)

        result = _make_result(posteriors)
        ext = extract_whole_haplotypes([result], threshold=0.95)

        assert len(ext.hap_indices) == H
        for h in range(H):
            idx = np.where(ext.hap_indices == h)[0][0]
            assert ext.ancestry_labels[idx] == h % A

    def test_admixed_filtered(self):
        """Haplotypes with mixed ancestry are filtered at high threshold."""
        H, T, A = 4, 100, 2
        posteriors = np.zeros((H, T, A))

        # Haplotypes 0-1: pure ancestry 0
        posteriors[0, :, 0] = 0.99
        posteriors[0, :, 1] = 0.01
        posteriors[1, :, 0] = 0.99
        posteriors[1, :, 1] = 0.01

        # Haplotype 2: admixed — first half ancestry 0, second half ancestry 1
        posteriors[2, :50, 0] = 0.99
        posteriors[2, :50, 1] = 0.01
        posteriors[2, 50:, 0] = 0.01
        posteriors[2, 50:, 1] = 0.99

        # Haplotype 3: low confidence everywhere
        posteriors[3, :, 0] = 0.55
        posteriors[3, :, 1] = 0.45

        result = _make_result(posteriors)
        ext = extract_whole_haplotypes([result], threshold=0.95)

        # Only haplotypes 0 and 1 should pass
        assert set(ext.hap_indices.tolist()) == {0, 1}
        assert all(ext.ancestry_labels == 0)

    def test_multi_chromosome(self):
        """Genome-wide minimum computed across chromosomes."""
        H, T, A = 4, 50, 2

        # Chromosome 1: all haplotypes confident
        post1 = np.full((H, T, A), 0.01)
        for h in range(H):
            post1[h, :, h % A] = 0.99

        # Chromosome 2: haplotype 0 has one low-confidence site
        post2 = np.full((H, T, A), 0.01)
        for h in range(H):
            post2[h, :, h % A] = 0.99
        # Make hap 0 uncertain at one site
        post2[0, 25, :] = [0.6, 0.4]

        r1 = _make_result(post1, chrom="1")
        r2 = _make_result(post2, chrom="2")
        ext = extract_whole_haplotypes([r1, r2], threshold=0.95)

        # Haplotype 0 should be filtered (min posterior = 0.6)
        assert 0 not in ext.hap_indices.tolist()
        # Haplotypes 1, 2, 3 should pass
        assert set(ext.hap_indices.tolist()) == {1, 2, 3}

    def test_empty_results(self):
        """Empty results list returns empty extraction."""
        ext = extract_whole_haplotypes([], threshold=0.95)
        assert len(ext.hap_indices) == 0


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

class TestSegmentExtraction:

    def test_finds_confident_segments(self):
        """Extracts contiguous high-confidence regions."""
        H, T, A = 2, 200, 2
        posteriors = np.full((H, T, A), 0.5)

        # Haplotype 0: confident ancestry-0 from site 10-109, ancestry-1 from 120-189
        posteriors[0, 10:110, :] = [0.995, 0.005]
        posteriors[0, 120:190, :] = [0.005, 0.995]

        result = _make_result(posteriors)
        cdata = _make_chromdata(H, T)

        segs = extract_segments(result, cdata, threshold=0.99, min_cm=0.0, min_sites=10)

        # Should find at least the two confident regions for haplotype 0
        hap0_segs = [s for s in segs if s.hap_index == 0]
        assert len(hap0_segs) == 2
        anc0_seg = [s for s in hap0_segs if s.ancestry == 0][0]
        anc1_seg = [s for s in hap0_segs if s.ancestry == 1][0]
        assert anc0_seg.n_sites == 100
        assert anc1_seg.n_sites == 70

    def test_min_sites_filtering(self):
        """Short segments are filtered by min_sites."""
        H, T, A = 1, 100, 2
        posteriors = np.full((H, T, A), 0.5)
        posteriors[0, 10:30, :] = [0.995, 0.005]  # 20 sites

        result = _make_result(posteriors)
        cdata = _make_chromdata(H, T)

        segs_pass = extract_segments(result, cdata, threshold=0.99, min_cm=0.0, min_sites=10)
        segs_fail = extract_segments(result, cdata, threshold=0.99, min_cm=0.0, min_sites=30)

        assert len(segs_pass) == 1
        assert len(segs_fail) == 0

    def test_min_cm_filtering(self):
        """Short segments are filtered by min_cm."""
        H, T, A = 1, 200, 2
        posteriors = np.full((H, T, A), 0.5)
        # Short segment: 5 sites spanning a small genetic distance
        posteriors[0, 0:5, :] = [0.995, 0.005]
        # Long segment: 100 sites
        posteriors[0, 50:150, :] = [0.995, 0.005]

        result = _make_result(posteriors)
        cdata = _make_chromdata(H, T)
        # pos_cm spans 0-100 over 200 sites, so 5 sites ~ 2.5 cM, 100 sites ~ 50 cM

        segs = extract_segments(result, cdata, threshold=0.99, min_cm=10.0, min_sites=1)
        assert len(segs) == 1
        assert segs[0].n_sites == 100

    def test_no_confident_sites(self):
        """Returns empty list when nothing is confident."""
        H, T, A = 2, 50, 2
        posteriors = np.full((H, T, A), 0.5)  # all uncertain

        result = _make_result(posteriors)
        cdata = _make_chromdata(H, T)

        segs = extract_segments(result, cdata, threshold=0.99, min_cm=0.0, min_sites=1)
        assert len(segs) == 0

    def test_ancestry_boundary_splits_segments(self):
        """Ancestry change creates a segment boundary even if both sides are confident."""
        H, T, A = 1, 100, 2
        posteriors = np.full((H, T, A), 0.01)
        posteriors[0, :50, 0] = 0.995
        posteriors[0, :50, 1] = 0.005
        posteriors[0, 50:, 0] = 0.005
        posteriors[0, 50:, 1] = 0.995

        result = _make_result(posteriors)
        cdata = _make_chromdata(H, T)

        segs = extract_segments(result, cdata, threshold=0.99, min_cm=0.0, min_sites=1)
        assert len(segs) == 2
        assert segs[0].ancestry != segs[1].ancestry


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

class TestOutputWriters:

    def test_write_panel_haplotypes(self, tmp_path):
        """Panel haplotypes TSV is parseable and correct."""
        ext = WholeHapExtraction(
            hap_indices=np.array([0, 3, 4]),
            ancestry_labels=np.array([0, 1, 0], dtype=np.int8),
            min_posteriors=np.array([0.97, 0.96, 0.98], dtype=np.float32),
            mean_posteriors=np.array([0.99, 0.98, 0.995], dtype=np.float32),
        )
        out = str(tmp_path / "haps.tsv")
        write_panel_haplotypes(["sample_A", "sample_B", "sample_C"], ext, out)

        lines = Path(out).read_text().strip().split("\n")
        assert lines[0].startswith("sample_id")
        assert len(lines) == 4  # header + 3 rows
        # Hap 0 = sample_A hap 0, hap 3 = sample_B hap 1, hap 4 = sample_C hap 0
        assert lines[1].startswith("sample_A\t0")
        assert lines[2].startswith("sample_B\t1")
        assert lines[3].startswith("sample_C\t0")

    def test_write_panel_segments(self, tmp_path):
        """Panel segments TSV.gz is parseable."""
        segs = [
            Segment(hap_index=0, chrom="1", start_site=10, end_site=109,
                    start_bp=10000, end_bp=109000, start_cm=1.0, end_cm=10.9,
                    ancestry=0, mean_posterior=0.998, n_sites=100),
        ]
        out = str(tmp_path / "segs.tsv.gz")
        write_panel_segments(segs, ["sample_A"], out)

        with gzip.open(out, "rt") as f:
            lines = f.read().strip().split("\n")
        assert lines[0].startswith("sample_id")
        assert len(lines) == 2
        assert "sample_A" in lines[1]

    def test_write_allele_frequencies(self, tmp_path):
        """Allele frequency table matches model.allele_freq values."""
        H, T, A = 10, 20, 3
        freq = np.random.default_rng(42).uniform(0.01, 0.99, (A, T))
        model = AncestryModel(
            n_ancestries=A,
            mu=jnp.ones(A) / A,
            gen_since_admix=20.0,
            allele_freq=jnp.array(freq, dtype=jnp.float32),
        )
        posteriors = jnp.full((H, T, A), 1.0 / A)
        result = AncestryResult(
            posteriors=posteriors,
            calls=np.zeros((H, T), dtype=np.int8),
            model=model,
            chrom="1",
        )
        cdata = _make_chromdata(H, T)
        out = str(tmp_path / "freq.tsv.gz")
        write_allele_frequencies([result], [cdata], out)

        # Read back and verify
        with gzip.open(out, "rt") as f:
            lines = f.read().strip().split("\n")
        assert len(lines) == T + 1  # header + T sites
        # Check first data row matches model frequencies
        cols = lines[1].split("\t")
        for a in range(A):
            written = float(cols[3 + a])
            np.testing.assert_almost_equal(written, freq[a, 0], decimal=5)

    def test_write_haplotype_proportions(self, tmp_path):
        """Per-haplotype proportions sum to ~1.0."""
        H, T, A = 6, 50, 3
        n_samples = H // 2
        # Generate random posteriors that sum to 1 along ancestry axis
        rng = np.random.default_rng(42)
        raw = rng.dirichlet(np.ones(A), size=(H, T))
        result = _make_result(raw.astype(np.float32))
        cdata = _make_chromdata(H, T)

        out = str(tmp_path / "props.tsv")
        write_haplotype_proportions(
            [result], n_samples,
            ["s1", "s2", "s3"], out,
        )

        lines = Path(out).read_text().strip().split("\n")
        assert len(lines) == H + 1  # header + H haplotype rows
        for line in lines[1:]:
            cols = line.split("\t")
            vals = [float(c) for c in cols[2:]]
            np.testing.assert_almost_equal(sum(vals), 1.0, decimal=3)


# ---------------------------------------------------------------------------
# Integration: export_panel entry point
# ---------------------------------------------------------------------------

class TestExportPanel:

    def test_export_panel_produces_all_files(self, tmp_path):
        """export_panel creates all expected output files."""
        H, T, A = 10, 100, 2
        posteriors = np.full((H, T, A), 0.01)
        for h in range(H):
            posteriors[h, :, h % A] = 0.99

        result = _make_result(posteriors)
        cdata = _make_chromdata(H, T)

        prefix = str(tmp_path / "test")
        config = PanelConfig(
            whole_hap_threshold=0.95,
            segment_threshold=0.98,
            min_segment_cm=0.0,
            min_segment_sites=10,
        )
        export_panel(
            [result], [cdata], H // 2,
            [f"s{i}" for i in range(H // 2)],
            prefix, config,
        )

        assert (tmp_path / "test.panel.haplotypes.tsv").exists()
        assert (tmp_path / "test.panel.segments.tsv.gz").exists()
        assert (tmp_path / "test.panel.frequencies.tsv.gz").exists()
        assert (tmp_path / "test.panel.proportions.tsv").exists()
