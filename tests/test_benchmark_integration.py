"""End-to-end integration test for the benchmark pipeline."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest

from popout.benchmark.align import align_haps, align_sites, apply_label_map, match_labels
from popout.benchmark.common import TractSet
from popout.benchmark.metrics import compute_all_metrics
from popout.benchmark.parsers.truth import tractset_from_arrays
from popout.benchmark.report import build_report


def _simulate_small(n_samples=50, n_sites=500, n_ancestries=3, seed=42):
    """Minimal simulation without JAX dependency."""
    rng = np.random.default_rng(seed)
    n_haps = 2 * n_samples
    A = n_ancestries

    # Generate per-site ancestry with realistic tract structure
    pos_bp = np.linspace(1000, 100_000_000, n_sites, dtype=np.int64)
    mu = rng.dirichlet(np.ones(A) * 2)
    true_ancestry = np.zeros((n_haps, n_sites), dtype=np.int8)

    for h in range(n_haps):
        anc = rng.choice(A, p=mu)
        for t in range(n_sites):
            if t > 0 and rng.random() < 0.02:
                anc = rng.choice(A, p=mu)
            true_ancestry[h, t] = anc

    return true_ancestry, pos_bp, mu


def _make_popout_output(true_ancestry, pos_bp, rng, error_rate=0.05):
    """Create popout-like output with permuted labels and some errors."""
    n_haps, n_sites = true_ancestry.shape
    A = int(true_ancestry.max()) + 1
    # Random permutation of labels
    perm = rng.permutation(A)
    calls = perm[true_ancestry].astype(np.uint16)
    # Introduce errors
    n_errors = int(n_haps * n_sites * error_rate)
    error_h = rng.integers(0, n_haps, size=n_errors)
    error_t = rng.integers(0, n_sites, size=n_errors)
    calls[error_h, error_t] = rng.integers(0, A, size=n_errors).astype(np.uint16)
    return calls, perm


def _make_flare_output(true_ancestry, pos_bp, rng, error_rate=0.03):
    """Create FLARE-like output (labels match truth, some errors)."""
    n_haps, n_sites = true_ancestry.shape
    A = int(true_ancestry.max()) + 1
    calls = true_ancestry.astype(np.uint16).copy()
    n_errors = int(n_haps * n_sites * error_rate)
    error_h = rng.integers(0, n_haps, size=n_errors)
    error_t = rng.integers(0, n_sites, size=n_errors)
    calls[error_h, error_t] = rng.integers(0, A, size=n_errors).astype(np.uint16)
    return calls


class TestIntegration:
    def test_full_pipeline(self, tmp_path):
        """Run the full benchmark pipeline end-to-end."""
        rng = np.random.default_rng(42)
        true_ancestry, pos_bp, mu = _simulate_small()
        n_haps, n_sites = true_ancestry.shape
        n_samples = n_haps // 2
        A = int(true_ancestry.max()) + 1

        sample_names = [f"S{i:04d}" for i in range(n_samples)]
        hap_ids = np.array(
            [f"{sample_names[i // 2]}:{i % 2}" for i in range(n_haps)],
            dtype=object,
        )

        # Build truth TractSet
        truth = TractSet(
            tool_name="truth",
            chrom="chr1",
            hap_ids=hap_ids.copy(),
            site_positions=pos_bp.copy(),
            calls=true_ancestry.astype(np.uint16),
            label_map={i: f"anc_{i}" for i in range(A)},
        )

        # Build popout TractSet (permuted labels + errors)
        popout_calls, perm = _make_popout_output(true_ancestry, pos_bp, rng)
        popout_ts = TractSet(
            tool_name="popout",
            chrom="chr1",
            hap_ids=hap_ids.copy(),
            site_positions=pos_bp.copy(),
            calls=popout_calls,
            label_map={i: str(i) for i in range(A)},
        )

        # Build FLARE TractSet (aligned labels + fewer errors)
        flare_calls = _make_flare_output(true_ancestry, pos_bp, rng)
        flare_ts = TractSet(
            tool_name="flare",
            chrom="chr1",
            hap_ids=hap_ids.copy(),
            site_positions=pos_bp.copy(),
            calls=flare_calls,
            label_map={i: f"anc_{i}" for i in range(A)},
        )

        # --- Test alignment ---
        pop_aligned, truth_aligned = align_sites(popout_ts, truth)
        pop_aligned, truth_aligned = align_haps(pop_aligned, truth_aligned)
        assert pop_aligned.n_sites == truth_aligned.n_sites
        assert pop_aligned.n_haps == truth_aligned.n_haps

        # --- Test Hungarian matching ---
        mapping = match_labels(pop_aligned, truth_aligned)
        # The mapping should recover the permutation
        for src_label, ref_label in mapping.items():
            assert ref_label == int(perm[src_label]), (
                f"Mapping {src_label}->{ref_label} != expected {src_label}->{perm[src_label]}"
            )

        # --- Test metrics after remapping ---
        pop_remapped = apply_label_map(pop_aligned, mapping)
        metrics = compute_all_metrics(pop_remapped, truth_aligned, b_is_truth=True)
        assert "per_site_accuracy" in metrics
        assert "per_ancestry_r2" in metrics
        assert "mean_r2" in metrics
        # With 5% error rate, accuracy should be around 0.95
        assert metrics["per_site_accuracy"] > 0.5
        # r² should be high
        for k, r2 in metrics["per_ancestry_r2"].items():
            assert r2 > 0.5, f"r² for ancestry {k} is {r2}"

        # --- Test report generation ---
        report_path = build_report(
            tracts={"popout": popout_ts, "flare": flare_ts},
            truth=truth,
            output_dir=tmp_path / "report",
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "LAI Benchmark Report" in content
        assert "Per-ancestry" in content
        assert len(content) > 100

    def test_no_truth(self, tmp_path):
        """Pipeline works without ground truth (tool-to-tool only)."""
        rng = np.random.default_rng(99)
        n_haps, n_sites, A = 20, 50, 2
        pos_bp = np.arange(100, 100 + n_sites * 100, 100, dtype=np.int64)
        hap_ids = np.array([f"S{i // 2:02d}:{i % 2}" for i in range(n_haps)], dtype=object)

        calls_a = rng.integers(0, A, size=(n_haps, n_sites)).astype(np.uint16)
        calls_b = rng.integers(0, A, size=(n_haps, n_sites)).astype(np.uint16)

        ts_a = TractSet("toolA", "chr1", hap_ids.copy(), pos_bp.copy(), calls_a,
                        {i: f"anc_{i}" for i in range(A)})
        ts_b = TractSet("toolB", "chr1", hap_ids.copy(), pos_bp.copy(), calls_b,
                        {i: f"anc_{i}" for i in range(A)})

        report_path = build_report(
            tracts={"toolA": ts_a, "toolB": ts_b},
            truth=None,
            output_dir=tmp_path / "report_notru",
        )
        assert report_path.exists()
        content = report_path.read_text()
        assert "Tool-to-tool agreement" in content
