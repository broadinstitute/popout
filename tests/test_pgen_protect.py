"""Tests for AIM panel position protection from cohort filtering.

Two layers of protection are validated here:

* :func:`popout.pgen_io._thin_sites` honors a ``must_keep_mask`` that
  forces flagged positions to survive cM thinning regardless of
  spacing.
* :func:`popout.pgen_io._apply_maf_mac_filter` honors
  ``must_keep_idxs`` so AIM positions sitting at cohort MAF below the
  threshold are still retained.

Plus a contract test for :func:`popout.prior_spec.panel_protect_positions`
that confirms it aggregates across all AIM signatures in a bundle, and
a round-trip test for the ``popout aim-panel-bed`` subcommand.
"""

from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from popout.aim_panel_bed import emit_bed
from popout.pgen_io import _apply_maf_mac_filter, _thin_sites
from popout.prior_spec import load_priors, panel_protect_positions


_REPO_ROOT = Path(__file__).resolve().parent.parent
_PRIORS_YAML = _REPO_ROOT / "configs" / "priors_v2.yaml"


# ---------------------------------------------------------------------------
# panel_protect_positions
# ---------------------------------------------------------------------------


def test_panel_protect_positions_aggregates_six_panels():
    """Loading the bundled priors_v2.yaml should yield ~83 positions
    spread across multiple chromosomes (matches the static
    all_panels.bed artifact)."""
    if not _PRIORS_YAML.exists():
        pytest.skip(f"{_PRIORS_YAML} is local-only and not present")
    priors = load_priors(_PRIORS_YAML)
    protect = panel_protect_positions(priors)
    assert len(protect) > 0, "expected at least one chrom with protected positions"
    n_total = sum(len(v) for v in protect.values())
    assert 50 < n_total < 500, (
        f"expected ~80–200 unique AIM positions in the bundled panels, got {n_total}"
    )
    for chrom, arr in protect.items():
        # chrom is normalized (no 'chr' prefix)
        assert not chrom.startswith("chr"), f"protect key {chrom!r} should be normalized"
        # positions sorted, unique, int64
        assert arr.dtype == np.int64
        assert (np.diff(arr) > 0).all(), f"positions on {chrom} not strictly sorted"


def test_panel_protect_positions_empty_when_no_aims():
    """Priors with only FST signatures (no AIMSignature) should
    produce an empty dict."""
    from popout.identity import FSTReferenceSignature
    from popout.prior_spec import (
        LinearAnnealingSchedule, Prior, Priors, prior_to_beta,
    )

    fst = FSTReferenceSignature(
        ref_freq=np.array([0.5, 0.6], dtype=np.float64),
        ref_pos_bp=np.array([100, 200], dtype=np.int64),
        ref_chrom=np.array(["1", "1"], dtype=object),
        ref_name="test",
        weight=1.0,
    )
    alpha, beta = prior_to_beta(7, 4, 12, 1e-4)
    p = Prior(
        name="X", identity_signatures=(fst,),
        gen_mean=7, gen_lo=4, gen_hi=12,
        alpha=alpha, beta=beta,
    )
    priors = Priors(
        priors=(p,), morgans_per_step=1e-4,
        annealing=LinearAnnealingSchedule(),
        fingerprint="0" * 64, source_path="/tmp/x.yaml",
    )
    assert panel_protect_positions(priors) == {}


# ---------------------------------------------------------------------------
# _thin_sites must_keep_mask
# ---------------------------------------------------------------------------


def test_thin_sites_must_keep_overrides_spacing():
    """A site within the spacing window should still be kept if its
    must_keep_mask entry is True."""
    pos_cm = np.array([0.0, 0.005, 0.01, 0.02, 0.03], dtype=float)
    must_keep = np.array([False, True, False, False, False])
    keep = _thin_sites(pos_cm, min_spacing_cm=0.02, must_keep_mask=must_keep)
    # site 0: kept (first); site 1: kept (must-keep, despite 0.005 < 0.02);
    # site 2: dropped (0.01 - 0.005 = 0.005 < 0.02);
    # site 3: dropped (0.02 - 0.005 = 0.015 < 0.02);
    # site 4: kept (last; also 0.03 - 0.005 = 0.025 >= 0.02).
    assert keep[1], "must-keep site at idx 1 was dropped"
    assert keep[0], "first site should always be kept"
    assert keep[-1], "last site should always be kept"


def test_thin_sites_no_must_keep_unchanged():
    """When must_keep_mask is None the behavior matches the
    pre-protection greedy pass."""
    pos_cm = np.array([0.0, 0.01, 0.025, 0.04, 0.06], dtype=float)
    keep = _thin_sites(pos_cm, min_spacing_cm=0.02)
    # 0 kept; 1 dropped (0.01 < 0.02); 2 kept (0.025 >= 0.02); 3 dropped
    # (0.04 - 0.025 = 0.015 < 0.02); 4 kept (last).
    assert keep[0] and keep[2] and keep[-1]
    assert not keep[1]


# ---------------------------------------------------------------------------
# _apply_maf_mac_filter must_keep_idxs
# ---------------------------------------------------------------------------


class _FakePgenReader:
    """Minimal stand-in for pgenlib.PgenReader.count() for unit testing.

    Maps variant index → (hom_ref, het, hom_alt, missing) tuple.
    """
    def __init__(self, counts: dict[int, tuple[int, int, int, int]]):
        self._counts = counts

    def count(self, vidx: int, buf: np.ndarray) -> None:
        buf[:] = self._counts[vidx]


def test_apply_maf_mac_filter_must_keep_bypasses_threshold():
    """A site with MAC=1 below min_mac=5 should still be kept when
    listed in must_keep_idxs."""
    n_samples = 100   # 200 alleles
    counts = {
        10: (199, 1, 0, 0),  # MAC = 1 → would fail min_mac=5
        11: (50, 100, 50, 0),  # MAC = 100 → passes anything reasonable
        12: (199, 0, 0, 1),    # MAC = 0 → always fails
    }
    reader = _FakePgenReader(counts)
    variant_idxs = np.array([10, 11, 12], dtype=np.uint32)

    # Without protection: only idx 11 survives.
    survived = _apply_maf_mac_filter(
        reader, variant_idxs, n_samples, min_maf=0.0, min_mac=5,
    )
    assert set(survived.tolist()) == {11}

    # With idx 10 protected: 10 and 11 both survive; 12 still dropped.
    survived = _apply_maf_mac_filter(
        reader, variant_idxs, n_samples, min_maf=0.0, min_mac=5,
        must_keep_idxs={10},
    )
    assert set(survived.tolist()) == {10, 11}


# ---------------------------------------------------------------------------
# popout aim-panel-bed subcommand
# ---------------------------------------------------------------------------


def test_emit_bed_round_trips_panel_positions(tmp_path):
    """emit_bed(priors_yaml) writes one row per AIM panel (chrom, pos)
    in 0-based half-open format."""
    if not _PRIORS_YAML.exists():
        pytest.skip(f"{_PRIORS_YAML} is local-only and not present")

    buf = io.StringIO()
    n = emit_bed(_PRIORS_YAML, buf)
    assert n > 0

    text = buf.getvalue()
    rows = [line.split("\t") for line in text.strip().split("\n")]
    assert all(len(r) == 3 for r in rows), "BED rows must have 3 columns"
    for chrom, start, end in rows:
        s, e = int(start), int(end)
        # 0-based half-open invariant: end == start + 1 for single-base sites
        assert e == s + 1, f"non-single-base BED row: {chrom} {start}-{end}"

    # Cross-check: BED row count matches panel_protect_positions total
    priors = load_priors(_PRIORS_YAML)
    protect = panel_protect_positions(priors)
    n_protect = sum(len(v) for v in protect.values())
    assert n == n_protect, (
        f"emit_bed wrote {n} rows but panel_protect_positions has "
        f"{n_protect} unique positions"
    )


def test_aim_panel_bed_subcommand_matches_static_artifact():
    """`popout aim-panel-bed` output equals the committed
    popout/data/aim_panels/all_panels.bed (no drift)."""
    if not _PRIORS_YAML.exists():
        pytest.skip(f"{_PRIORS_YAML} is local-only and not present")
    static_bed = _REPO_ROOT / "popout" / "data" / "aim_panels" / "all_panels.bed"
    if not static_bed.exists():
        pytest.skip("static all_panels.bed not committed yet")

    proc = subprocess.run(
        [sys.executable, "-m", "popout", "aim-panel-bed",
         "--priors", str(_PRIORS_YAML)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"aim-panel-bed exited {proc.returncode}: {proc.stderr}"

    # Subprocess output may have other log noise on stderr; stdout should be
    # the BED only.
    fresh = proc.stdout.strip().split("\n")
    committed = static_bed.read_text().strip().split("\n")
    assert fresh == committed, (
        "static all_panels.bed has drifted from `popout aim-panel-bed` output. "
        "Regenerate via `python -m popout aim-panel-bed --priors "
        "configs/priors_v2.yaml --out popout/data/aim_panels/all_panels.bed`."
    )
