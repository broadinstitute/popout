"""Tests for ancestry labeling module."""

import csv
import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest

from popout.label import (
    LabelResult,
    _assign_labels,
    _correlation_matrix,
    label_ancestries,
    rewrite_global_tsv,
    rewrite_tracts_tsv,
)


def _make_model_npz(tmp_path: Path, freq: np.ndarray, chrom: str = "chr20") -> Path:
    """Create a fake model.npz with given frequencies and uniform positions."""
    K, T = freq.shape
    path = tmp_path / "test.model.npz"
    np.savez_compressed(
        path,
        allele_freq=freq,
        pos_bp=np.arange(T, dtype=np.int64) * 1000 + 1_000_000,
        pos_cm=np.linspace(0, 50, T),
        chrom=np.array(chrom),
        mu=np.ones(K) / K,
        mismatch=np.zeros(K),
        n_ancestries=K,
        gen_since_admix=20.0,
    )
    return path


def _make_ref_tsv(
    tmp_path: Path,
    freq: np.ndarray,
    pos_bp: np.ndarray | None = None,
    pop_names: list[str] | None = None,
    chrom: str = "chr20",
) -> Path:
    """Create a fake reference frequency TSV."""
    K_ref, T = freq.shape
    if pos_bp is None:
        pos_bp = np.arange(T, dtype=np.int64) * 1000 + 1_000_000
    if pop_names is None:
        pop_names = ["EUR", "EAS", "AMR", "AFR", "SAS"][:K_ref]

    path = tmp_path / "ref.tsv.gz"
    with gzip.open(path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["#chrom", "pos", "ref", "alt"] + pop_names)
        for i in range(T):
            row = [chrom, int(pos_bp[i]), "A", "T"] + [f"{freq[k, i]:.4f}" for k in range(K_ref)]
            writer.writerow(row)
    return path


class TestCorrelationMatrix:
    def test_perfect_correlation(self):
        """Identical rows give correlation of 1."""
        a = np.array([[0.1, 0.5, 0.9, 0.3]])
        corr = _correlation_matrix(a, a)
        np.testing.assert_allclose(corr[0, 0], 1.0, atol=1e-6)

    def test_shape(self):
        """Output shape is (M, N)."""
        a = np.random.default_rng(42).uniform(0, 1, (3, 100))
        b = np.random.default_rng(43).uniform(0, 1, (5, 100))
        corr = _correlation_matrix(a, b)
        assert corr.shape == (3, 5)


class TestAssignLabels:
    def test_one_to_one(self):
        """K_inf == K_ref gives 1-to-1 matching."""
        # Correlation matrix where ancestry i matches ref i
        corr = np.array([
            [0.95, 0.1, 0.2],
            [0.1, 0.90, 0.15],
            [0.2, 0.1, 0.85],
        ])
        ref_names = ["EUR", "AFR", "EAS"]
        label_map, merge_map = _assign_labels(corr, ref_names)

        assert label_map == {0: "EUR", 1: "AFR", 2: "EAS"}
        assert all(len(v) == 1 for v in merge_map.values())

    def test_fewer_inferred_than_ref(self):
        """K_inf < K_ref leaves some refs unmatched."""
        corr = np.array([
            [0.9, 0.1, 0.2, 0.1, 0.05],
            [0.1, 0.05, 0.1, 0.85, 0.1],
        ])
        ref_names = ["EUR", "EAS", "AMR", "AFR", "SAS"]
        label_map, merge_map = _assign_labels(corr, ref_names)

        assert len(label_map) == 2
        assert label_map[0] == "EUR"
        assert label_map[1] == "AFR"

    def test_merge_when_more_inferred(self):
        """K_inf > K_ref merges multiple inferred to same ref."""
        # 4 inferred, 3 ref: two inferred map to same ref
        corr = np.array([
            [0.9, 0.1, 0.2],   # -> EUR
            [0.1, 0.85, 0.1],  # -> EAS
            [0.8, 0.1, 0.15],  # -> EUR (merge)
            [0.1, 0.1, 0.9],   # -> AFR
        ])
        ref_names = ["EUR", "EAS", "AFR"]
        label_map, merge_map = _assign_labels(corr, ref_names)

        assert label_map[0] == "EUR"
        assert label_map[2] == "EUR"
        assert label_map[1] == "EAS"
        assert label_map[3] == "AFR"
        assert len(merge_map["EUR"]) == 2
        # Strongest correlation first
        assert merge_map["EUR"][0] == 0  # r=0.9 > r=0.8


class TestLabelAncestries:
    def test_perfect_match(self, tmp_path):
        """When inferred frequencies match reference exactly, labels are correct."""
        rng = np.random.default_rng(42)
        T = 500
        # Create distinct frequency profiles for 3 populations
        ref_freq = np.array([
            rng.uniform(0.0, 0.3, T),  # EUR-like: low freq
            rng.uniform(0.3, 0.7, T),  # AFR-like: mid freq
            rng.uniform(0.7, 1.0, T),  # EAS-like: high freq
        ], dtype=np.float32)

        # Inferred matches reference but in different order: [EAS, EUR, AFR]
        inf_freq = np.array([ref_freq[2], ref_freq[0], ref_freq[1]], dtype=np.float32)

        model_path = _make_model_npz(tmp_path, inf_freq)
        ref_path = _make_ref_tsv(tmp_path, ref_freq, pop_names=["EUR", "AFR", "EAS"])

        result = label_ancestries(model_path, ref_path)

        assert result.label_map[0] == "EAS"
        assert result.label_map[1] == "EUR"
        assert result.label_map[2] == "AFR"
        assert result.n_overlapping_sites == T

    def test_partial_site_overlap(self, tmp_path):
        """Works with only partial site overlap."""
        rng = np.random.default_rng(42)
        T = 200

        # Model sites: positions 1M, 1001K, 1002K, ...
        model_freq = rng.uniform(0.1, 0.9, (3, T)).astype(np.float32)
        model_path = _make_model_npz(tmp_path, model_freq)

        # Reference has every other site overlapping + extra sites
        ref_pos = np.arange(T, dtype=np.int64) * 2000 + 1_000_000  # every 2nd site overlaps
        ref_freq = rng.uniform(0.1, 0.9, (3, T)).astype(np.float32)
        ref_path = _make_ref_tsv(tmp_path, ref_freq, pos_bp=ref_pos, pop_names=["EUR", "AFR", "EAS"])

        result = label_ancestries(model_path, ref_path)
        assert result.n_overlapping_sites == T // 2

    def test_too_few_sites_raises(self, tmp_path):
        """Raises when fewer than 10 sites overlap."""
        model_freq = np.random.default_rng(42).uniform(0.1, 0.9, (3, 100)).astype(np.float32)
        model_path = _make_model_npz(tmp_path, model_freq)

        # Reference at completely different positions
        ref_pos = np.arange(100, dtype=np.int64) * 1000 + 999_000_000
        ref_freq = np.random.default_rng(43).uniform(0.1, 0.9, (3, 100)).astype(np.float32)
        ref_path = _make_ref_tsv(tmp_path, ref_freq, pos_bp=ref_pos, pop_names=["EUR", "AFR", "EAS"])

        with pytest.raises(ValueError, match="overlapping sites"):
            label_ancestries(model_path, ref_path)


class TestRewriteGlobalTsv:
    def test_relabel(self, tmp_path):
        """Columns are renamed to population labels."""
        in_path = tmp_path / "input.global.tsv"
        with open(in_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sample", "ancestry_0", "ancestry_1", "ancestry_2"])
            writer.writerow(["S1", "0.500000", "0.300000", "0.200000"])
            writer.writerow(["S2", "0.100000", "0.800000", "0.100000"])

        label_result = LabelResult(
            label_map={0: "EUR", 1: "AFR", 2: "EAS"},
            merge_map={"EUR": [0], "AFR": [1], "EAS": [2]},
            correlations=np.eye(3),
            ref_names=["EUR", "AFR", "EAS"],
            n_overlapping_sites=100,
        )

        out_path = tmp_path / "output.global.tsv"
        rewrite_global_tsv(in_path, out_path, label_result)

        with open(out_path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)

        assert header == ["sample", "EUR", "AFR", "EAS"]
        assert rows[0] == ["S1", "0.500000", "0.300000", "0.200000"]

    def test_merge_columns(self, tmp_path):
        """Merged ancestries have their proportions summed."""
        in_path = tmp_path / "input.global.tsv"
        with open(in_path, "w") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["sample", "ancestry_0", "ancestry_1", "ancestry_2"])
            writer.writerow(["S1", "0.400000", "0.100000", "0.500000"])

        # ancestry_0 and ancestry_2 both map to EUR
        label_result = LabelResult(
            label_map={0: "EUR", 1: "AFR", 2: "EUR"},
            merge_map={"EUR": [0, 2], "AFR": [1]},
            correlations=np.eye(3),
            ref_names=["EUR", "AFR"],
            n_overlapping_sites=100,
        )

        out_path = tmp_path / "output.global.tsv"
        rewrite_global_tsv(in_path, out_path, label_result)

        with open(out_path) as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)

        assert header == ["sample", "EUR", "AFR"]
        # EUR = 0.4 + 0.5 = 0.9
        np.testing.assert_almost_equal(float(rows[0][1]), 0.9, decimal=5)
        np.testing.assert_almost_equal(float(rows[0][2]), 0.1, decimal=5)


class TestRewriteTractsTsv:
    def test_relabel(self, tmp_path):
        """Ancestry integers are replaced with population names."""
        in_path = tmp_path / "input.tracts.tsv.gz"
        with gzip.open(in_path, "wt") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["#chrom", "start_bp", "end_bp", "sample", "haplotype", "ancestry", "n_sites"])
            writer.writerow(["chr20", "1000", "5000", "S1", "0", "0", "10"])
            writer.writerow(["chr20", "5000", "9000", "S1", "0", "1", "8"])

        label_result = LabelResult(
            label_map={0: "EUR", 1: "AFR"},
            merge_map={"EUR": [0], "AFR": [1]},
            correlations=np.eye(2),
            ref_names=["EUR", "AFR"],
            n_overlapping_sites=100,
        )

        out_path = tmp_path / "output.tracts.tsv.gz"
        rewrite_tracts_tsv(in_path, out_path, label_result)

        with gzip.open(out_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)

        assert rows[0][5] == "EUR"
        assert rows[1][5] == "AFR"

    def test_merge_adjacent(self, tmp_path):
        """Adjacent tracts with same remapped label are merged."""
        in_path = tmp_path / "input.tracts.tsv.gz"
        with gzip.open(in_path, "wt") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["#chrom", "start_bp", "end_bp", "sample", "haplotype", "ancestry", "n_sites"])
            # ancestry 0 and 2 both map to EUR
            writer.writerow(["chr20", "1000", "3000", "S1", "0", "0", "5"])
            writer.writerow(["chr20", "3000", "6000", "S1", "0", "2", "7"])
            writer.writerow(["chr20", "6000", "9000", "S1", "0", "1", "4"])

        label_result = LabelResult(
            label_map={0: "EUR", 1: "AFR", 2: "EUR"},
            merge_map={"EUR": [0, 2], "AFR": [1]},
            correlations=np.eye(3),
            ref_names=["EUR", "AFR"],
            n_overlapping_sites=100,
        )

        out_path = tmp_path / "output.tracts.tsv.gz"
        rewrite_tracts_tsv(in_path, out_path, label_result)

        with gzip.open(out_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            rows = list(reader)

        # Two EUR tracts merged into one, then one AFR tract
        assert len(rows) == 2
        assert rows[0][5] == "EUR"
        assert rows[0][1] == "1000"
        assert rows[0][2] == "6000"
        assert rows[0][6] == "12"  # 5 + 7
        assert rows[1][5] == "AFR"

    def test_no_merge_different_haplotypes(self, tmp_path):
        """Tracts on different haplotypes are not merged."""
        in_path = tmp_path / "input.tracts.tsv.gz"
        with gzip.open(in_path, "wt") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["#chrom", "start_bp", "end_bp", "sample", "haplotype", "ancestry", "n_sites"])
            writer.writerow(["chr20", "1000", "5000", "S1", "0", "0", "10"])
            writer.writerow(["chr20", "1000", "5000", "S1", "1", "0", "10"])

        label_result = LabelResult(
            label_map={0: "EUR"},
            merge_map={"EUR": [0]},
            correlations=np.array([[1.0]]),
            ref_names=["EUR"],
            n_overlapping_sites=100,
        )

        out_path = tmp_path / "output.tracts.tsv.gz"
        rewrite_tracts_tsv(in_path, out_path, label_result)

        with gzip.open(out_path, "rt") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            rows = list(reader)

        assert len(rows) == 2
