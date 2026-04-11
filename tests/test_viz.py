"""Integration tests for popout.viz visualization module.

Generates synthetic output files and verifies each plot function
produces a valid matplotlib Figure without error.
"""

from __future__ import annotations

import gzip
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture
def synthetic_run(tmp_path):
    """Create synthetic popout output files for testing."""
    prefix = tmp_path / "test_run"
    n_samples = 50
    n_haps = 2 * n_samples
    n_anc = 4
    n_sites = 200
    rng = np.random.default_rng(42)

    # ---- global.tsv ----
    props = rng.dirichlet(np.ones(n_anc), size=n_samples).astype(np.float32)
    with open(f"{prefix}.global.tsv", "w") as f:
        f.write("sample\t" + "\t".join(f"ancestry_{a}" for a in range(n_anc)) + "\n")
        for i in range(n_samples):
            vals = "\t".join(f"{v:.4f}" for v in props[i])
            f.write(f"SAMPLE_{i}\t{vals}\n")

    # ---- tracts.tsv.gz ----
    with gzip.open(f"{prefix}.tracts.tsv.gz", "wt") as f:
        f.write("#chrom\tstart_bp\tend_bp\tsample\thaplotype\tancestry\tn_sites\tmean_posterior\n")
        for chrom_num in [1, 2, 20]:
            chrom = f"chr{chrom_num}"
            for si in range(n_samples):
                for hap in [0, 1]:
                    pos = 0
                    while pos < 100_000_000:
                        anc = rng.integers(0, n_anc)
                        length = int(rng.exponential(5_000_000))
                        length = max(length, 100_000)
                        end = min(pos + length, 100_000_000)
                        n_s = max(1, (end - pos) // 50_000)
                        post = float(rng.uniform(0.7, 1.0))
                        f.write(f"{chrom}\t{pos}\t{end}\tSAMPLE_{si}\t{hap}\t{anc}\t{n_s}\t{post:.4f}\n")
                        pos = end

    # ---- model ----
    mu = props.mean(axis=0)
    mu /= mu.sum()
    with open(f"{prefix}.model", "w") as f:
        f.write(f"n_ancestries\t{n_anc}\n")
        f.write(f"gen_since_admix\t25.00\n")
        f.write(f"mu\t{','.join(f'{x:.4f}' for x in mu)}\n")
        f.write(f"mismatch\t{','.join('0.001000' for _ in range(n_anc))}\n")

    # ---- model.npz ----
    allele_freq = rng.uniform(0.01, 0.99, size=(n_anc, n_sites)).astype(np.float32)
    gen_per_hap = rng.exponential(25, size=n_haps).astype(np.float32)
    bucket_centers = np.geomspace(1, 1000, 20).astype(np.float32)
    np.savez_compressed(
        f"{prefix}.model.npz",
        allele_freq=allele_freq,
        mu=mu,
        mismatch=np.full(n_anc, 0.001, dtype=np.float32),
        n_ancestries=np.array(n_anc),
        gen_since_admix=np.array(25.0),
        gen_per_hap=gen_per_hap,
        bucket_centers=bucket_centers,
    )

    # ---- summary.json ----
    summary = {
        "em_convergence": [
            {"iteration": i, "max_delta_freq": 0.5 * (0.8 ** i),
             "mean_delta_freq": 0.01 * (0.7 ** i)}
            for i in range(10)
        ],
        "spectral": {
            "singular_values": [100 - i * 8 for i in range(10)],
            "gap_ratios": [1.5, 1.3, 1.2, 2.0, 1.1, 1.05, 1.02, 1.01, 1.0],
            "n_ancestries": n_anc,
        },
        "output": {
            "genome_wide_ancestry_proportions": mu.tolist(),
            "mean_posterior_confidence": 0.92,
        },
    }
    with open(f"{prefix}.summary.json", "w") as f:
        json.dump(summary, f)

    # ---- stats.jsonl ----
    with open(f"{prefix}.stats.jsonl", "w") as f:
        for i in range(10):
            f.write(json.dumps({"key": "em/max_delta_freq", "value": 0.5 * (0.8 ** i),
                                "context": {"iteration": i}}) + "\n")
            f.write(json.dumps({"key": "em/mean_delta_freq", "value": 0.01 * (0.7 ** i),
                                "context": {"iteration": i}}) + "\n")
            f.write(json.dumps({"key": "em/mu", "value": mu.tolist(),
                                "context": {"iteration": i}}) + "\n")
            f.write(json.dumps({"key": "em/T", "value": 20.0 + i * 0.5,
                                "context": {"iteration": i}}) + "\n")

    return prefix


class TestVizPlots:
    """Test each plot function produces a valid Figure."""

    def test_admixture(self, synthetic_run):
        from popout.viz import plot_admixture
        fig = plot_admixture(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ancestry_density(self, synthetic_run):
        from popout.viz import plot_ancestry_density
        fig = plot_ancestry_density(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_karyogram(self, synthetic_run):
        from popout.viz import plot_karyogram
        fig = plot_karyogram(synthetic_run, "SAMPLE_0")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_tract_lengths(self, synthetic_run):
        from popout.viz import plot_tract_lengths
        fig = plot_tract_lengths(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_switch_rate(self, synthetic_run):
        from popout.viz import plot_switch_rate
        fig = plot_switch_rate(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ancestry_along_genome(self, synthetic_run):
        from popout.viz import plot_ancestry_along_genome
        fig = plot_ancestry_along_genome(synthetic_run, window_mb=10.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multi_individual(self, synthetic_run):
        from popout.viz import plot_multi_individual
        fig = plot_multi_individual(synthetic_run, chrom="chr1",
                                    max_individuals=20, window_mb=5.0)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_convergence(self, synthetic_run):
        from popout.viz import plot_convergence
        fig = plot_convergence(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_posterior_confidence(self, synthetic_run):
        from popout.viz import plot_posterior_confidence
        fig = plot_posterior_confidence(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_chromosome_boxplots(self, synthetic_run):
        from popout.viz import plot_chromosome_boxplots
        fig = plot_chromosome_boxplots(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_freq_divergence(self, synthetic_run):
        from popout.viz import plot_freq_divergence
        fig = plot_freq_divergence(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_per_hap_t(self, synthetic_run):
        from popout.viz import plot_per_hap_t
        fig = plot_per_hap_t(synthetic_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_ternary_skips_if_not_k3(self, synthetic_run):
        """Ternary plot should raise ValueError when K != 3."""
        from popout.viz import plot_ternary
        with pytest.raises(ValueError, match="exactly 3"):
            plot_ternary(synthetic_run)

    def test_gallery(self, synthetic_run, tmp_path):
        """Gallery generates multiple plots from synthetic data."""
        from popout.viz import generate_gallery
        out_dir = tmp_path / "gallery"
        generated = generate_gallery(
            synthetic_run, out_dir,
            sample="SAMPLE_0",
        )
        assert len(generated) > 5
        for path in generated:
            assert path.exists()
            assert path.stat().st_size > 0

    def test_gallery_parallel(self, synthetic_run, tmp_path):
        """Parallel gallery produces the same plots as sequential."""
        from popout.viz import generate_gallery
        out_seq = tmp_path / "gallery_seq"
        out_par = tmp_path / "gallery_par"
        seq = generate_gallery(synthetic_run, out_seq, sample="SAMPLE_0", workers=1)
        par = generate_gallery(synthetic_run, out_par, sample="SAMPLE_0", workers=2)
        assert sorted(p.name for p in seq) == sorted(p.name for p in par)
        for path in par:
            assert path.exists()
            assert path.stat().st_size > 0


@pytest.fixture
def synthetic_k3_run(tmp_path):
    """Create synthetic popout output with K=3 for ternary plot test."""
    prefix = tmp_path / "test_k3"
    n_samples = 30
    n_anc = 3
    rng = np.random.default_rng(123)

    props = rng.dirichlet(np.ones(n_anc), size=n_samples).astype(np.float32)
    with open(f"{prefix}.global.tsv", "w") as f:
        f.write("sample\t" + "\t".join(f"ancestry_{a}" for a in range(n_anc)) + "\n")
        for i in range(n_samples):
            vals = "\t".join(f"{v:.4f}" for v in props[i])
            f.write(f"SAMPLE_{i}\t{vals}\n")
    return prefix


class TestTernary:
    def test_ternary_k3(self, synthetic_k3_run):
        from popout.viz import plot_ternary
        fig = plot_ternary(synthetic_k3_run)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
