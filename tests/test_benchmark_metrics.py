"""Tests for benchmark metrics."""

import numpy as np

from popout.benchmark.common import TractSet
from popout.benchmark.metrics import (
    compute_all_metrics,
    global_fraction_error,
    per_ancestry_precision_recall,
    per_ancestry_r2,
    per_haplotype_accuracy,
    per_site_accuracy,
    tract_length_stats,
)


def _make_ts(calls, label_map=None, tool_name="test"):
    n_haps, n_sites = calls.shape
    return TractSet(
        tool_name=tool_name,
        chrom="chr1",
        hap_ids=np.array([f"S{i // 2:02d}:{i % 2}" for i in range(n_haps)], dtype=object),
        site_positions=np.arange(100, 100 + n_sites * 100, 100, dtype=np.int64),
        calls=np.array(calls, dtype=np.uint16),
        label_map=label_map or {0: "eur", 1: "afr"},
    )


class TestPerAncestryR2:
    def test_perfect_agreement(self):
        # Ensure varying per-hap fractions so std != 0
        calls = np.array([[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=np.uint16)
        a = _make_ts(calls)
        b = _make_ts(calls)
        r2 = per_ancestry_r2(a, b)
        assert abs(r2[0] - 1.0) < 1e-10
        assert abs(r2[1] - 1.0) < 1e-10

    def test_random_gives_low_r2(self):
        np.random.seed(123)
        calls_a = np.random.choice([0, 1], size=(20, 100)).astype(np.uint16)
        calls_b = np.random.choice([0, 1], size=(20, 100)).astype(np.uint16)
        a = _make_ts(calls_a)
        b = _make_ts(calls_b)
        r2 = per_ancestry_r2(a, b)
        # Random should give low r2
        assert r2[0] < 0.3
        assert r2[1] < 0.3


class TestPerSiteAccuracy:
    def test_perfect(self):
        calls = np.array([[0, 1, 0, 1]], dtype=np.uint16)
        a = _make_ts(calls)
        b = _make_ts(calls)
        assert per_site_accuracy(a, b) == 1.0

    def test_half(self):
        calls_a = np.array([[0, 0, 1, 1]], dtype=np.uint16)
        calls_b = np.array([[0, 1, 0, 1]], dtype=np.uint16)
        a = _make_ts(calls_a)
        b = _make_ts(calls_b)
        assert per_site_accuracy(a, b) == 0.5


class TestPerHaplotypeAccuracy:
    def test_per_hap(self):
        calls_a = np.array([[0, 0, 0, 0], [1, 1, 0, 0]], dtype=np.uint16)
        calls_b = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.uint16)
        a = _make_ts(calls_a)
        b = _make_ts(calls_b)
        accs = per_haplotype_accuracy(a, b)
        assert accs[0] == 1.0
        assert accs[1] == 0.75


class TestPrecisionRecall:
    def test_known_values(self):
        # a calls: all 0. b calls: half 0, half 1
        calls_a = np.array([[0, 0, 0, 0]], dtype=np.uint16)
        calls_b = np.array([[0, 0, 1, 1]], dtype=np.uint16)
        a = _make_ts(calls_a)
        b = _make_ts(calls_b)
        pr = per_ancestry_precision_recall(a, b)
        # For label 0: a called 0 everywhere (4 sites), b called 0 at 2 sites
        # precision = 2/4 = 0.5, recall = 2/2 = 1.0
        assert abs(pr[0]["precision"] - 0.5) < 1e-10
        assert abs(pr[0]["recall"] - 1.0) < 1e-10


class TestGlobalFractionError:
    def test_zero_error(self):
        calls = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint16)
        a = _make_ts(calls)
        b = _make_ts(calls)
        errors = global_fraction_error(a, b)
        assert errors[0] == 0.0

    def test_known_error(self):
        # Sample has haps [0,0,0,0] and [0,0,0,0] in a -> eur=1.0
        # In b: haps [1,1,1,1] and [1,1,1,1] -> afr=1.0
        # L1 = |1-0| + |0-1| = 2.0
        calls_a = np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint16)
        calls_b = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.uint16)
        a = _make_ts(calls_a)
        b = _make_ts(calls_b)
        errors = global_fraction_error(a, b)
        assert errors[0] == 2.0


class TestTractLengthStats:
    def test_basic(self):
        # Single hap: [0,0,1,1,1] -> 2 tracts: (2 sites, 3 sites)
        calls = np.array([[0, 0, 1, 1, 1]], dtype=np.uint16)
        ts = _make_ts(calls)
        stats = tract_length_stats(ts)
        assert stats["count"] == 2
        assert stats["sites"]["min"] == 2
        assert stats["sites"]["max"] == 3


class TestComputeAllMetrics:
    def test_returns_all_keys(self):
        calls = np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.uint16)
        a = _make_ts(calls, tool_name="popout")
        b = _make_ts(calls, tool_name="truth")
        result = compute_all_metrics(a, b, b_is_truth=True)
        assert "per_site_accuracy" in result
        assert "per_ancestry_r2" in result
        assert "mean_r2" in result
        assert "global_fraction_error_mean" in result
        assert "tract_stats_a" in result
        assert result["per_site_accuracy"] == 1.0
