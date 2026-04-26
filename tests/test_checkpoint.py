"""Tests for popout.checkpoint — WorkDir, manifest I/O, invalidation."""

import json
import os

import numpy as np
import pytest

from popout.checkpoint import STAGE_ORDER, WorkDir


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def work_path(tmp_path):
    return tmp_path / "test.work"


def _fp():
    """Minimal input fingerprint dict."""
    return {
        "pgen_sha_prefix": "abcd1234abcd1234",
        "n_haps": 1000,
        "thin_cm": 0.02,
        "seeding_exclusion_sha_prefix": None,
    }


def _args():
    """Minimal args dict."""
    return {
        "seed_method": "recursive",
        "n_ancestries": None,
        "max_ancestries": 20,
        "ancestry_detection": "marchenko-pastur",
        "recursive_kwargs_hash": "deadbeef",
        "seed": 42,
        "gen_since_admix": 20.0,
        "n_em_iter": 5,
        "block_emissions": False,
        "block_size": 8,
        "freeze_anchors_iters": 0,
        "probs": False,
        "per_hap_T": False,
        "n_T_buckets": 20,
    }


def _open(wd, **overrides):
    """Open/create a work dir with default fingerprint and args."""
    fp = _fp()
    args = _args()
    fp.update(overrides.get("fp_overrides", {}))
    args.update(overrides.get("args_overrides", {}))
    wd.open_or_create(
        popout_version="0.3.1-test",
        input_fingerprint=fp,
        args=args,
        restart_stage=overrides.get("restart_stage"),
    )


# ---------------------------------------------------------------------------
# TestWorkDir — creation and manifest I/O
# ---------------------------------------------------------------------------

class TestWorkDir:
    def test_create_new_workdir(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        assert work_path.exists()
        assert (work_path / "manifest.json").exists()

    def test_manifest_roundtrip(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        raw = json.loads((work_path / "manifest.json").read_text())
        assert raw["popout_version"] == "0.3.1-test"
        assert raw["input_fingerprint"]["n_haps"] == 1000
        assert raw["args"]["seed_method"] == "recursive"
        assert raw["stages"]["seed"]["done"] is False

    def test_atomic_write_no_corruption(self, work_path):
        """Simulate a crash: .tmp file left behind, real file absent."""
        wd = WorkDir(work_path)
        _open(wd)

        # Create a fake .tmp file (simulating an incomplete write)
        tmp_file = work_path / "something.manifest.tmp"
        tmp_file.write_text("garbage")

        # The real manifest should still be valid
        wd2 = WorkDir(work_path)
        m = wd2._read_manifest()
        assert m is not None
        assert m["popout_version"] == "0.3.1-test"

    def test_stage_lifecycle(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        assert not wd.stage_done("seed")
        wd.mark_done("seed", wall_s=42.5)
        assert wd.stage_done("seed")
        # Verify wall_s persisted
        m = wd._read_manifest()
        assert m["stages"]["seed"]["wall_s"] == 42.5
        assert m["stages"]["seed"]["done"] is True

    def test_decode_per_chrom(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        assert not wd.stage_done("decode", chrom="1")
        assert not wd.stage_done("decode", chrom="2")
        wd.mark_done("decode", chrom="1", wall_s=10.0)
        assert wd.stage_done("decode", chrom="1")
        assert not wd.stage_done("decode", chrom="2")

    def test_pending_decode_chroms(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        all_chroms = ["1", "2", "3"]
        assert wd.pending_decode_chroms(all_chroms) == ["1", "2", "3"]
        wd.mark_done("decode", chrom="1")
        wd.mark_done("decode", chrom="3")
        assert wd.pending_decode_chroms(all_chroms) == ["2"]

    def test_all_decode_done(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        chroms = ["1", "2"]
        assert not wd.all_decode_done(chroms)
        wd.mark_done("decode", chrom="1")
        assert not wd.all_decode_done(chroms)
        wd.mark_done("decode", chrom="2")
        assert wd.all_decode_done(chroms)

    def test_stage_path(self, work_path):
        wd = WorkDir(work_path)
        assert wd.stage_path("seed") == work_path / "seed.npz"
        assert wd.stage_path("em") == work_path / "em.npz"
        assert wd.stage_path("decode", chrom="1") == (
            work_path / "decode" / "chr1.parquet"
        )
        assert wd.stage_path("tracts") == work_path / "tracts.tsv.gz"

    def test_stage_path_decode_requires_chrom(self, work_path):
        wd = WorkDir(work_path)
        with pytest.raises(ValueError, match="requires chrom"):
            wd.stage_path("decode")

    def test_stage_done_decode_requires_chrom(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        with pytest.raises(ValueError, match="requires chrom"):
            wd.stage_done("decode")

    def test_reopen_preserves_state(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)

        # Reopen with same fingerprint/args
        wd2 = WorkDir(work_path)
        _open(wd2)
        assert wd2.stage_done("seed")
        assert wd2.stage_done("em")
        assert wd2.stage_done("decode", chrom="1")
        assert not wd2.stage_done("tracts")


# ---------------------------------------------------------------------------
# TestInvalidation
# ---------------------------------------------------------------------------

class TestInvalidation:
    def test_fingerprint_change_invalidates_all(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)

        # Reopen with different pgen fingerprint
        wd2 = WorkDir(work_path)
        _open(wd2, fp_overrides={"pgen_sha_prefix": "different_hash!!"})
        assert not wd2.stage_done("seed")
        assert not wd2.stage_done("em")
        assert not wd2.stage_done("decode", chrom="1")

    def test_seed_args_change_invalidates_seed_and_later(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)

        # Change seed_method (a seed-stage arg)
        wd2 = WorkDir(work_path)
        _open(wd2, args_overrides={"seed_method": "gmm"})
        assert not wd2.stage_done("seed")
        assert not wd2.stage_done("em")

    def test_em_args_change_preserves_seed(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)

        # Change gen_since_admix (an em-stage arg)
        wd2 = WorkDir(work_path)
        _open(wd2, args_overrides={"gen_since_admix": 12.0})
        assert wd2.stage_done("seed")  # preserved
        assert not wd2.stage_done("em")  # invalidated
        assert not wd2.stage_done("decode", chrom="1")  # cascade

    def test_resume_invalidates_em_when_per_hap_T_changes(self, work_path):
        """Toggling --per-hap-T must invalidate em + decode but preserve
        seed: the EM iteration produces a different model (with vs without
        bucket_assignments) and decode follows a different code path."""
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)

        wd2 = WorkDir(work_path)
        _open(wd2, args_overrides={"per_hap_T": True})
        assert wd2.stage_done("seed")
        assert not wd2.stage_done("em")
        assert not wd2.stage_done("decode", chrom="1")

    def test_resume_invalidates_em_when_n_T_buckets_changes(self, work_path):
        """Changing --n-T-buckets changes the bucket centers used by the EM
        M-step and decode, so both stages must invalidate while seed stays."""
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)

        wd2 = WorkDir(work_path)
        _open(wd2, args_overrides={"n_T_buckets": 10})
        assert wd2.stage_done("seed")
        assert not wd2.stage_done("em")
        assert not wd2.stage_done("decode", chrom="1")

    def test_restart_stage_em(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)
        wd.mark_done("tracts", wall_s=5)

        # Restart from em
        wd2 = WorkDir(work_path)
        _open(wd2, restart_stage="em")
        assert wd2.stage_done("seed")  # preserved
        assert not wd2.stage_done("em")
        assert not wd2.stage_done("decode", chrom="1")
        assert not wd2.stage_done("tracts")

    def test_restart_stage_all(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed")
        wd.mark_done("em")

        wd2 = WorkDir(work_path)
        _open(wd2, restart_stage="all")
        assert not wd2.stage_done("seed")
        assert not wd2.stage_done("em")

    def test_matching_args_preserves_all(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)
        wd.mark_done("seed", wall_s=100)
        wd.mark_done("em", wall_s=200)
        wd.mark_done("decode", chrom="1", wall_s=30)
        wd.mark_done("tracts", wall_s=5)

        # Reopen with identical args
        wd2 = WorkDir(work_path)
        _open(wd2)
        assert wd2.stage_done("seed")
        assert wd2.stage_done("em")
        assert wd2.stage_done("decode", chrom="1")
        assert wd2.stage_done("tracts")

    def test_invalidation_removes_files(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)

        # Create fake stage files
        seed_path = wd.stage_path("seed")
        seed_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(str(seed_path), dummy=np.array([1]))
        assert seed_path.exists()

        em_path = wd.stage_path("em")
        np.savez_compressed(str(em_path), dummy=np.array([1]))
        assert em_path.exists()

        wd.mark_done("seed")
        wd.mark_done("em")

        # Invalidate from seed — should remove seed.npz and em.npz
        wd2 = WorkDir(work_path)
        _open(wd2, fp_overrides={"pgen_sha_prefix": "different_hash!!"})
        assert not seed_path.exists()
        assert not em_path.exists()

    def test_decode_invalidation_removes_parquets(self, work_path):
        wd = WorkDir(work_path)
        _open(wd)

        # Create fake decode files
        for c in ["1", "2"]:
            pq = wd.stage_path("decode", chrom=c)
            pq.parent.mkdir(parents=True, exist_ok=True)
            pq.write_text("fake")
            gs = pq.with_suffix(".global_sums.npy")
            gs.write_text("fake")
            wd.mark_done("decode", chrom=c)

        assert wd.stage_done("decode", chrom="1")

        # Invalidate decode
        wd._invalidate_from("decode")
        assert not wd.stage_done("decode", chrom="1")
        assert not wd.stage_done("decode", chrom="2")
        assert not (wd.stage_path("decode", chrom="1")).exists()
        assert not (wd.stage_path("decode", chrom="2")).exists()


# ---------------------------------------------------------------------------
# TestFingerprint
# ---------------------------------------------------------------------------

class TestFingerprint:
    def test_pgen_fingerprint_deterministic(self, tmp_path):
        pgen = tmp_path / "test.pgen"
        pgen.write_bytes(os.urandom(2048))
        fp1 = WorkDir.compute_pgen_fingerprint(pgen)
        fp2 = WorkDir.compute_pgen_fingerprint(pgen)
        assert fp1 == fp2
        assert len(fp1) == 16

    def test_pgen_fingerprint_differs_on_content(self, tmp_path):
        p1 = tmp_path / "a.pgen"
        p2 = tmp_path / "b.pgen"
        p1.write_bytes(os.urandom(2048))
        p2.write_bytes(os.urandom(2048))
        assert WorkDir.compute_pgen_fingerprint(p1) != \
            WorkDir.compute_pgen_fingerprint(p2)

    def test_pgen_fingerprint_directory(self, tmp_path):
        d = tmp_path / "pgens"
        d.mkdir()
        (d / "chr1.pgen").write_bytes(os.urandom(1024))
        (d / "chr2.pgen").write_bytes(os.urandom(1024))
        fp = WorkDir.compute_pgen_fingerprint(d)
        assert len(fp) == 16

    def test_hash_recursive_kwargs_deterministic(self):
        kw = {"max_leaves": 20, "merge_hellinger": 0.012, "min_leaf_size": 500}
        h1 = WorkDir.hash_recursive_kwargs(kw)
        h2 = WorkDir.hash_recursive_kwargs(kw)
        assert h1 == h2

    def test_hash_recursive_kwargs_none(self):
        assert WorkDir.hash_recursive_kwargs(None) is None

    def test_hash_recursive_kwargs_order_independent(self):
        kw1 = {"a": 1, "b": 2}
        kw2 = {"b": 2, "a": 1}
        assert WorkDir.hash_recursive_kwargs(kw1) == \
            WorkDir.hash_recursive_kwargs(kw2)

    def test_hash_exclusion_file(self, tmp_path):
        f = tmp_path / "exclude.tsv"
        f.write_text("sample_id\nS1\nS2\n")
        h = WorkDir.hash_exclusion_file(f)
        assert h is not None and len(h) == 16

    def test_hash_exclusion_file_none(self):
        assert WorkDir.hash_exclusion_file(None) is None


# ---------------------------------------------------------------------------
# TestAtomicWrite
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_atomic_write_npz(self, work_path):
        wd = WorkDir(work_path)
        dest = work_path / "test.npz"
        wd.atomic_write_npz(dest, {"arr": np.array([1, 2, 3])})
        assert dest.exists()
        data = np.load(str(dest))
        np.testing.assert_array_equal(data["arr"], [1, 2, 3])

    def test_atomic_write_npy(self, work_path):
        wd = WorkDir(work_path)
        dest = work_path / "test.npy"
        arr = np.array([4.0, 5.0, 6.0])
        wd.atomic_write_npy(dest, arr)
        assert dest.exists()
        loaded = np.load(str(dest))
        np.testing.assert_array_equal(loaded, arr)

    def test_atomic_write_npz_creates_parent_dirs(self, work_path):
        wd = WorkDir(work_path)
        dest = work_path / "subdir" / "nested" / "test.npz"
        wd.atomic_write_npz(dest, {"x": np.array([1])})
        assert dest.exists()
