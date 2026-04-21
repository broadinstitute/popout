"""Tests for benchmark parsers."""

import gzip
import tempfile
from pathlib import Path

import numpy as np
import pytest

from popout.benchmark.common import MISSING_LABEL, TractSet, load_ancestry_header
from popout.benchmark.parsers.flare import parse_flare
from popout.benchmark.parsers.popout import parse_popout
from popout.benchmark.parsers.truth import parse_truth, tractset_from_arrays


# ---------- common.py tests ----------


class TestTractSet:
    def _make_ts(self):
        """3 haps, 5 sites, 2 ancestries."""
        calls = np.array(
            [
                [0, 0, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
            ],
            dtype=np.uint16,
        )
        return TractSet(
            tool_name="test",
            chrom="chr1",
            hap_ids=np.array(["S01:0", "S01:1", "S02:0"], dtype=object),
            site_positions=np.array([100, 200, 300, 400, 500], dtype=np.int64),
            calls=calls,
            label_map={0: "eur", 1: "afr"},
        )

    def test_n_haps_n_sites(self):
        ts = self._make_ts()
        assert ts.n_haps == 3
        assert ts.n_sites == 5

    def test_to_tracts(self):
        ts = self._make_ts()
        tracts = ts.to_tracts()
        # Hap 0: [0,0,1,1,1] -> (0, 0, 2, 0), (0, 2, 5, 1)
        # Hap 1: [1,1,1,0,0] -> (1, 0, 3, 1), (1, 3, 5, 0)
        # Hap 2: [0,0,0,0,1] -> (2, 0, 4, 0), (2, 4, 5, 1)
        assert (0, 0, 2, 0) in tracts
        assert (0, 2, 5, 1) in tracts
        assert (1, 0, 3, 1) in tracts
        assert (1, 3, 5, 0) in tracts
        assert (2, 0, 4, 0) in tracts
        assert (2, 4, 5, 1) in tracts
        assert len(tracts) == 6

    def test_global_fractions(self):
        ts = self._make_ts()
        fracs = ts.global_fractions()
        assert fracs.shape == (3, 2)
        # Hap 0: 2/5 eur, 3/5 afr
        np.testing.assert_allclose(fracs[0], [0.4, 0.6], atol=1e-10)
        # Each row sums to 1
        np.testing.assert_allclose(fracs.sum(axis=1), [1.0, 1.0, 1.0], atol=1e-10)

    def test_validate_ok(self):
        ts = self._make_ts()
        ts.validate()  # Should not raise

    def test_validate_shape_mismatch(self):
        ts = self._make_ts()
        ts.hap_ids = np.array(["S01:0", "S01:1"], dtype=object)  # 2 != 3
        with pytest.raises(ValueError, match="hap_ids length"):
            ts.validate()

    def test_validate_nonmonotonic(self):
        ts = self._make_ts()
        ts.site_positions = np.array([100, 200, 150, 400, 500], dtype=np.int64)
        with pytest.raises(ValueError, match="monotonically increasing"):
            ts.validate()

    def test_validate_invalid_labels(self):
        ts = self._make_ts()
        ts.calls[0, 0] = 99  # Not in label_map
        with pytest.raises(ValueError, match="labels not in label_map"):
            ts.validate()


class TestLoadAncestryHeader:
    def test_basic(self):
        line = "##ANCESTRY=<nwe=0,see=1,afr=2,eas=3>"
        result = load_ancestry_header(line)
        assert result == {0: "nwe", 1: "see", 2: "afr", 3: "eas"}

    def test_with_spaces(self):
        line = "##ANCESTRY=< eur = 0 , afr = 1 >"
        result = load_ancestry_header(line)
        assert result == {0: "eur", 1: "afr"}

    def test_invalid(self):
        with pytest.raises(ValueError):
            load_ancestry_header("##ANCESTRY=bad_format")


# ---------- FLARE parser tests ----------


class TestFlareParser:
    def _write_flare_vcf(self, path: Path):
        """Write a tiny FLARE VCF: 3 samples, 5 sites, 4 ancestries."""
        lines = [
            "##fileformat=VCFv4.3\n",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n',
            '##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="Ancestry hap1">\n',
            '##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="Ancestry hap2">\n',
            "##ANCESTRY=<eur=0,afr=1,eas=2,amr=3>\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS01\tS02\tS03\n",
            "chr1\t1000\t.\tA\tT\t.\t.\t.\tGT:AN1:AN2\t0|1:0:1\t1|0:2:0\t0|0:3:3\n",
            "chr1\t2000\t.\tC\tG\t.\t.\t.\tGT:AN1:AN2\t0|0:0:1\t1|1:2:2\t0|1:3:1\n",
            "chr1\t3000\t.\tG\tA\t.\t.\t.\tGT:AN1:AN2\t1|0:1:1\t0|1:0:0\t1|1:2:2\n",
            "chr1\t4000\t.\tT\tC\t.\t.\t.\tGT:AN1:AN2\t0|1:1:0\t1|0:2:1\t0|0:3:3\n",
            "chr1\t5000\t.\tA\tG\t.\t.\t.\tGT:AN1:AN2\t1|1:0:0\t0|0:2:2\t1|0:1:1\n",
        ]
        with gzip.open(path, "wt") as f:
            f.writelines(lines)

    def test_parse_flare(self, tmp_path):
        vcf_path = tmp_path / "flare_tiny.anc.vcf.gz"
        self._write_flare_vcf(vcf_path)
        ts = parse_flare(vcf_path)

        assert ts.tool_name == "flare"
        assert ts.chrom == "chr1"
        assert ts.label_map == {0: "eur", 1: "afr", 2: "eas", 3: "amr"}
        assert list(ts.hap_ids) == [
            "S01:0", "S01:1", "S02:0", "S02:1", "S03:0", "S03:1"
        ]
        np.testing.assert_array_equal(
            ts.site_positions, [1000, 2000, 3000, 4000, 5000]
        )
        assert ts.n_haps == 6
        assert ts.n_sites == 5

        # Spot-check: S01 hap0 (AN1) at sites: 0,0,1,1,0
        np.testing.assert_array_equal(ts.calls[0], [0, 0, 1, 1, 0])
        # S01 hap1 (AN2) at sites: 1,1,1,0,0
        np.testing.assert_array_equal(ts.calls[1], [1, 1, 1, 0, 0])
        # S02 hap0 (AN1): 2,2,0,2,2
        np.testing.assert_array_equal(ts.calls[2], [2, 2, 0, 2, 2])

    def test_parse_flare_missing(self, tmp_path):
        """Test that '.' is parsed as MISSING_LABEL."""
        vcf_path = tmp_path / "missing.anc.vcf.gz"
        lines = [
            "##ANCESTRY=<eur=0,afr=1>\n",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS01\n",
            "chr1\t100\t.\tA\tT\t.\t.\t.\tGT:AN1:AN2\t0|1:.:1\n",
        ]
        with gzip.open(vcf_path, "wt") as f:
            f.writelines(lines)
        ts = parse_flare(vcf_path)
        assert ts.calls[0, 0] == MISSING_LABEL
        assert ts.calls[1, 0] == 1

    def test_global_crosscheck(self, tmp_path):
        vcf_path = tmp_path / "flare.anc.vcf.gz"
        self._write_flare_vcf(vcf_path)
        # Write matching global file
        global_path = tmp_path / "flare.global.anc.gz"
        # Compute expected fracs from VCF for S01
        # hap0: [0,0,1,1,0] -> eur=3/5, afr=2/5
        # hap1: [1,1,1,0,0] -> eur=2/5, afr=3/5
        # sample avg: eur=(3/5+2/5)/2=0.5, afr=(2/5+3/5)/2=0.5
        lines = [
            "SAMPLE\teur\tafr\teas\tamr\n",
            "S01\t0.500\t0.500\t0.000\t0.000\n",
            "S02\t0.100\t0.100\t0.600\t0.200\n",
            "S03\t0.000\t0.200\t0.200\t0.600\n",
        ]
        with gzip.open(global_path, "wt") as f:
            f.writelines(lines)
        # Should not raise
        ts = parse_flare(vcf_path, global_path=global_path)
        assert ts is not None


# ---------- popout parser tests ----------


class TestPopoutParser:
    def _write_popout_tracts(self, path: Path):
        """Write a small popout tracts file."""
        lines = [
            "#chrom\tstart_bp\tend_bp\tsample\thaplotype\tancestry\tn_sites\n",
            "chr1\t1000\t3000\tS01\t0\t0\t3\n",
            "chr1\t3001\t5000\tS01\t0\t1\t2\n",
            "chr1\t1000\t5000\tS01\t1\t1\t5\n",
            "chr1\t1000\t2000\tS02\t0\t0\t2\n",
            "chr1\t2001\t5000\tS02\t0\t2\t3\n",
            "chr1\t1000\t5000\tS02\t1\t0\t5\n",
        ]
        with open(path, "w") as f:
            f.writelines(lines)

    def test_parse_popout(self, tmp_path):
        path = tmp_path / "popout.tracts.tsv"
        self._write_popout_tracts(path)
        # Provide specific site positions for deterministic test
        site_pos = np.array([1000, 2000, 3000, 4000, 5000], dtype=np.int64)
        ts = parse_popout(path, site_positions=site_pos)

        assert ts.tool_name == "popout"
        assert ts.chrom == "chr1"
        assert ts.n_sites == 5
        # 4 haplotypes: S01:0, S01:1, S02:0, S02:1
        assert ts.n_haps == 4
        assert "S01:0" in ts.hap_ids
        assert "S02:1" in ts.hap_ids

        # S01:0 tracts: [1000,3000] -> anc 0, [3001,5000] -> anc 1
        # sites 1000,2000,3000 = anc 0; sites 4000,5000 = anc 1
        idx = list(ts.hap_ids).index("S01:0")
        np.testing.assert_array_equal(ts.calls[idx], [0, 0, 0, 1, 1])

        # S02:1: [1000,5000] -> anc 0 (all sites)
        idx = list(ts.hap_ids).index("S02:1")
        np.testing.assert_array_equal(ts.calls[idx], [0, 0, 0, 0, 0])

    def test_parse_popout_auto_positions(self, tmp_path):
        """Without explicit site_positions, uses union of start/end."""
        path = tmp_path / "popout.tracts.tsv"
        self._write_popout_tracts(path)
        ts = parse_popout(path)
        # Should have positions from union of all start/end values
        assert ts.n_sites > 0
        assert ts.n_haps == 4


# ---------- truth parser tests ----------


class TestTruthParser:
    def test_parse_truth_npz(self, tmp_path):
        true_ancestry = np.array(
            [[0, 0, 1, 1, 2], [1, 1, 0, 0, 2], [2, 2, 2, 1, 0], [0, 1, 2, 0, 1]],
            dtype=np.int8,
        )
        pos_bp = np.array([100, 200, 300, 400, 500], dtype=np.int64)
        npz_path = tmp_path / "truth.npz"
        np.savez(npz_path, true_ancestry=true_ancestry, pos_bp=pos_bp, chrom="chr1", n_ancestries=3)

        ts = parse_truth(npz_path)
        assert ts.tool_name == "truth"
        assert ts.chrom == "chr1"
        assert ts.n_haps == 4
        assert ts.n_sites == 5
        assert ts.label_map == {0: "anc_0", 1: "anc_1", 2: "anc_2"}
        np.testing.assert_array_equal(ts.calls[0], [0, 0, 1, 1, 2])

    def test_tractset_from_arrays(self):
        true_anc = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int8)
        pos = np.array([10, 20, 30], dtype=np.int64)
        ts = tractset_from_arrays(true_anc, pos, sample_names=["A"])
        assert ts.n_haps == 2
        assert list(ts.hap_ids) == ["A:0", "A:1"]
        np.testing.assert_array_equal(ts.calls[0], [0, 1, 0])
