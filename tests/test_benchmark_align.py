"""Tests for benchmark alignment and label matching."""

import numpy as np
import pytest

from popout.benchmark.align import (
    align_haps,
    align_sites,
    apply_label_map,
    match_labels,
)
from popout.benchmark.common import TractSet


def _make_ts(tool_name, hap_ids, positions, calls, label_map):
    return TractSet(
        tool_name=tool_name,
        chrom="chr1",
        hap_ids=np.array(hap_ids, dtype=object),
        site_positions=np.array(positions, dtype=np.int64),
        calls=np.array(calls, dtype=np.uint16),
        label_map=label_map,
    )


class TestAlignSites:
    def test_intersect(self):
        a = _make_ts("a", ["h0", "h1"], [100, 200, 300, 400],
                     [[0, 1, 0, 1], [1, 0, 1, 0]], {0: "x", 1: "y"})
        b = _make_ts("b", ["h0", "h1"], [200, 300, 500],
                     [[1, 0, 1], [0, 1, 0]], {0: "x", 1: "y"})
        a_out, b_out = align_sites(a, b, strategy="intersect")
        # Common: [200, 300]
        np.testing.assert_array_equal(a_out.site_positions, [200, 300])
        np.testing.assert_array_equal(b_out.site_positions, [200, 300])
        np.testing.assert_array_equal(a_out.calls, [[1, 0], [0, 1]])
        np.testing.assert_array_equal(b_out.calls, [[1, 0], [0, 1]])

    def test_no_common(self):
        a = _make_ts("a", ["h0"], [100, 200], [[0, 1]], {0: "x", 1: "y"})
        b = _make_ts("b", ["h0"], [300, 400], [[0, 1]], {0: "x", 1: "y"})
        with pytest.raises(ValueError, match="No common site positions"):
            align_sites(a, b)


class TestAlignHaps:
    def test_common_haps(self):
        a = _make_ts("a", ["h0", "h1", "h2"], [100, 200],
                     [[0, 1], [1, 0], [0, 0]], {0: "x", 1: "y"})
        b = _make_ts("b", ["h1", "h2", "h3"], [100, 200],
                     [[1, 0], [0, 0], [1, 1]], {0: "x", 1: "y"})
        a_out, b_out = align_haps(a, b)
        # Common: h1, h2
        assert list(a_out.hap_ids) == ["h1", "h2"]
        assert list(b_out.hap_ids) == ["h1", "h2"]
        np.testing.assert_array_equal(a_out.calls, [[1, 0], [0, 0]])
        np.testing.assert_array_equal(b_out.calls, [[1, 0], [0, 0]])

    def test_no_common(self):
        a = _make_ts("a", ["h0"], [100], [[0]], {0: "x"})
        b = _make_ts("b", ["h1"], [100], [[0]], {0: "x"})
        with pytest.raises(ValueError, match="No common haplotype"):
            align_haps(a, b)


class TestMatchLabels:
    def test_permuted_labels(self):
        """Construct two TractSets where the true mapping is {0:2, 1:0, 2:1}."""
        np.random.seed(42)
        n_haps, n_sites = 10, 100
        # Ground truth with labels 0, 1, 2
        ref_calls = np.random.choice([0, 1, 2], size=(n_haps, n_sites)).astype(np.uint16)
        # Source: permuted — src label 0 = ref label 2, etc.
        perm = {0: 2, 1: 0, 2: 1}
        inv_perm = {v: k for k, v in perm.items()}
        src_calls = np.vectorize(inv_perm.get)(ref_calls).astype(np.uint16)

        positions = np.arange(100, 100 + n_sites * 10, 10, dtype=np.int64)
        hap_ids = np.array([f"h{i}" for i in range(n_haps)], dtype=object)

        src = TractSet("popout", "chr1", hap_ids, positions, src_calls,
                       {0: "0", 1: "1", 2: "2"})
        ref = TractSet("truth", "chr1", hap_ids, positions, ref_calls,
                       {0: "anc_0", 1: "anc_1", 2: "anc_2"})

        mapping = match_labels(src, ref)
        # The mapping should recover the permutation
        assert mapping == perm

    def test_unequal_k(self):
        """K_src < K_ref should still work."""
        hap_ids = np.array(["h0", "h1"], dtype=object)
        positions = np.array([100, 200, 300], dtype=np.int64)
        # src has 2 labels, ref has 3
        src_calls = np.array([[0, 0, 1], [1, 0, 0]], dtype=np.uint16)
        ref_calls = np.array([[2, 2, 0], [0, 2, 2]], dtype=np.uint16)
        src = TractSet("popout", "chr1", hap_ids, positions, src_calls,
                       {0: "0", 1: "1"})
        ref = TractSet("truth", "chr1", hap_ids, positions, ref_calls,
                       {0: "anc_0", 1: "anc_1", 2: "anc_2"})
        mapping = match_labels(src, ref)
        # src 0 -> ref 2, src 1 -> ref 0
        assert mapping[0] == 2
        assert mapping[1] == 0


class TestApplyLabelMap:
    def test_remap(self):
        hap_ids = np.array(["h0"], dtype=object)
        positions = np.array([100, 200, 300], dtype=np.int64)
        calls = np.array([[0, 1, 2]], dtype=np.uint16)
        ts = TractSet("popout", "chr1", hap_ids, positions, calls,
                      {0: "0", 1: "1", 2: "2"})
        remapped = apply_label_map(ts, {0: 2, 1: 0, 2: 1})
        np.testing.assert_array_equal(remapped.calls[0], [2, 0, 1])
