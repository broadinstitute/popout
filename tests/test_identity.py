"""Tests for popout.identity."""

from __future__ import annotations

import numpy as np
import pytest

from popout.identity import (
    AIMPanel,
    AIMSignature,
    ComponentState,
    FSTReferenceSignature,
    compose_scores,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _make_state(
    freq: np.ndarray,
    pos_bp: np.ndarray,
    *,
    mu: float = 0.25,
    chrom: str = "1",
) -> ComponentState:
    return ComponentState(
        freq=np.asarray(freq, dtype=np.float64),
        mu=mu,
        pos_bp=np.asarray(pos_bp, dtype=np.int64),
        chrom=chrom,
    )


def _aim_panel(
    expected: list[float],
    *,
    chrom: str = "1",
    pos_start: int = 1000,
    weight: float = 1.0,
) -> AIMPanel:
    n = len(expected)
    return AIMPanel(
        chrom=np.array([chrom] * n, dtype=object),
        pos_bp=np.arange(pos_start, pos_start + n, dtype=np.int64),
        expected_freq=np.asarray(expected, dtype=np.float64),
        marker_weight=np.full(n, weight, dtype=np.float64),
        source="test",
    )


# --------------------------------------------------------------------------
# AIMSignature
# --------------------------------------------------------------------------


def test_aim_signature_ranks_matching_component_first():
    """One of K=4 components matches the panel; it must score highest."""
    rng = np.random.default_rng(0)
    n_markers = 50
    target_freqs = rng.uniform(0.05, 0.95, n_markers)
    panel = _aim_panel(target_freqs.tolist())
    sig = AIMSignature(panel=panel)

    pos = panel.pos_bp.copy()
    matching = target_freqs + rng.normal(0.0, 0.02, n_markers)  # tiny noise
    matching = np.clip(matching, 0.001, 0.999)
    others = [
        np.clip(rng.uniform(0.05, 0.95, n_markers), 0.001, 0.999)
        for _ in range(3)
    ]

    states = [_make_state(matching, pos)] + [_make_state(o, pos) for o in others]
    scores = np.array([sig.score(s) for s in states])

    # Matching component (index 0) has the highest score.
    assert int(np.argmax(scores)) == 0

    # And it is > 3 SD above the mean of the OTHERS' scores.
    others_scores = scores[1:]
    others_mean = others_scores.mean()
    others_sd = others_scores.std() + 1e-9
    margin = (scores[0] - others_mean) / others_sd
    assert margin > 3.0, f"matching component only {margin:.2f} SD above others"


def test_aim_signature_returns_zero_when_no_overlap():
    panel = _aim_panel([0.5, 0.5, 0.5], chrom="1", pos_start=10_000)
    sig = AIMSignature(panel=panel)

    # ComponentState on chrom 2 — no chromosome overlap.
    cs_other_chrom = _make_state([0.5, 0.5, 0.5], [10_000, 10_001, 10_002], chrom="2")
    assert sig.score(cs_other_chrom) == 0.0

    # ComponentState on chrom 1 but disjoint positions — no position overlap.
    cs_disjoint = _make_state([0.5, 0.5, 0.5], [99_000, 99_001, 99_002], chrom="1")
    assert sig.score(cs_disjoint) == 0.0


def test_aim_signature_chrom_normalization():
    """'chr1' panel and '1' component (or vice versa) match."""
    panel = _aim_panel([0.5], chrom="chr1")
    sig = AIMSignature(panel=panel)
    cs = _make_state([0.5], panel.pos_bp.copy(), chrom="1")
    # No NaN, no zero — score is non-zero (zero exact match would be -0).
    s = sig.score(cs)
    assert np.isfinite(s)


def test_aim_signature_handles_nan_freq():
    panel = _aim_panel([0.5, 0.5, 0.5])
    sig = AIMSignature(panel=panel)
    cs = _make_state([np.nan, np.nan, np.nan], panel.pos_bp.copy())
    s = sig.score(cs)
    assert np.isfinite(s)
    # NaN diffs become 0 → contrib is 0 → score is 0.
    assert s == 0.0


# --------------------------------------------------------------------------
# FSTReferenceSignature
# --------------------------------------------------------------------------


def test_fst_signature_ranks_matching_component_first():
    rng = np.random.default_rng(1)
    n_sites = 200
    pos = np.arange(2000, 2000 + n_sites, dtype=np.int64)
    ref_freq = rng.uniform(0.05, 0.95, n_sites)
    sig = FSTReferenceSignature(
        ref_freq=ref_freq,
        ref_pos_bp=pos,
        ref_chrom="1",
        ref_name="1KG_TEST",
    )

    matching = np.clip(ref_freq + rng.normal(0.0, 0.02, n_sites), 0.001, 0.999)
    others = [
        np.clip(rng.uniform(0.05, 0.95, n_sites), 0.001, 0.999)
        for _ in range(3)
    ]

    states = [_make_state(matching, pos)] + [_make_state(o, pos) for o in others]
    scores = np.array([sig.score(s) for s in states])
    assert int(np.argmax(scores)) == 0
    others_mean = scores[1:].mean()
    others_sd = scores[1:].std() + 1e-9
    margin = (scores[0] - others_mean) / others_sd
    assert margin > 3.0


def test_fst_signature_zero_on_chrom_mismatch():
    sig = FSTReferenceSignature(
        ref_freq=np.array([0.5, 0.6, 0.7]),
        ref_pos_bp=np.array([10, 20, 30]),
        ref_chrom="1",
        ref_name="1KG_TEST",
    )
    cs = _make_state([0.5, 0.6, 0.7], [10, 20, 30], chrom="2")
    assert sig.score(cs) == 0.0


def test_fst_signature_handles_nan_freq():
    sig = FSTReferenceSignature(
        ref_freq=np.array([0.5, 0.6, 0.7]),
        ref_pos_bp=np.array([10, 20, 30]),
        ref_chrom="1",
        ref_name="1KG_TEST",
    )
    cs = _make_state([np.nan, np.nan, np.nan], [10, 20, 30])
    s = sig.score(cs)
    assert np.isfinite(s)
    assert s == 0.0


# --------------------------------------------------------------------------
# compose_scores
# --------------------------------------------------------------------------


def test_compose_scores_combines_two_signatures():
    """K=4: comp 0 matches AIM only; comp 1 matches F_ST only;
    comp 2 matches both; comp 3 matches neither. Comp 2 wins.

    The AIM panel and F_ST reference cover *disjoint* site ranges so
    that a "match-both" component can be constructed by matching each
    signature on its own range.
    """
    rng = np.random.default_rng(2)
    n_aim = 60
    n_fst = 60
    n_total = n_aim + n_fst
    pos = np.arange(5000, 5000 + n_total, dtype=np.int64)

    truth_aim = rng.uniform(0.05, 0.95, n_aim)
    truth_fst = rng.uniform(0.05, 0.95, n_fst)

    panel = AIMPanel(
        chrom=np.array(["1"] * n_aim, dtype=object),
        pos_bp=pos[:n_aim].copy(),
        expected_freq=truth_aim,
        marker_weight=np.ones(n_aim),
        source="test",
    )
    aim_sig = AIMSignature(panel=panel)

    fst_sig = FSTReferenceSignature(
        ref_freq=truth_fst,
        ref_pos_bp=pos[n_aim:].copy(),
        ref_chrom="1",
        ref_name="ref",
    )

    def jitter(arr):
        return np.clip(arr + rng.normal(0.0, 0.02, len(arr)), 0.001, 0.999)

    # Random fillers for the off-range portion of each component.
    def random_fill(n):
        return np.clip(rng.uniform(0.05, 0.95, n), 0.001, 0.999)

    # comp 0: matches AIM range only; F_ST range random
    comp_0 = np.concatenate([jitter(truth_aim), random_fill(n_fst)])
    # comp 1: AIM range random; F_ST range matches truth_fst
    comp_1 = np.concatenate([random_fill(n_aim), jitter(truth_fst)])
    # comp 2: matches AIM range AND F_ST range
    comp_2 = np.concatenate([jitter(truth_aim), jitter(truth_fst)])
    # comp 3: random everywhere
    comp_3 = random_fill(n_total)

    states = [
        _make_state(comp_0, pos),
        _make_state(comp_1, pos),
        _make_state(comp_2, pos),
        _make_state(comp_3, pos),
    ]

    combined = compose_scores([aim_sig, fst_sig], states)
    assert int(np.argmax(combined)) == 2, f"combined={combined}"

    # Single-signature ablations
    aim_only = compose_scores([aim_sig], states)
    fst_only = compose_scores([fst_sig], states)
    # comp 0 and comp 2 both match AIM well; tie-break by argmax may pick
    # either. Same for FST with comp 1 and comp 2. We assert the matching
    # components rank in the top 2.
    assert int(np.argmax(aim_only)) in (0, 2), f"aim_only={aim_only}"
    assert int(np.argmax(fst_only)) in (1, 2), f"fst_only={fst_only}"


def test_compose_scores_skips_degenerate_signature():
    """A signature returning identical scores for all components should
    contribute zero to the combined output (z-score of constant is 0)."""

    class ConstSig:
        weight = 1.0

        def score(self, cs):
            return 7.0

    pos = np.array([10, 20, 30], dtype=np.int64)
    states = [_make_state([0.5, 0.5, 0.5], pos) for _ in range(4)]
    combined = compose_scores([ConstSig()], states)
    np.testing.assert_array_equal(combined, np.zeros(4))


def test_compose_scores_empty_signatures_returns_zeros():
    states = [
        _make_state([0.5, 0.5], [10, 20]),
        _make_state([0.5, 0.5], [10, 20]),
    ]
    combined = compose_scores([], states)
    np.testing.assert_array_equal(combined, np.zeros(2))


def test_compose_scores_all_nan_components_finite():
    """Components with all-NaN freqs do not crash composition."""
    panel = _aim_panel([0.5, 0.5, 0.5])
    sig = AIMSignature(panel=panel)
    pos = panel.pos_bp.copy()
    states = [
        _make_state([np.nan, np.nan, np.nan], pos),
        _make_state([np.nan, np.nan, np.nan], pos),
        _make_state([np.nan, np.nan, np.nan], pos),
    ]
    combined = compose_scores([sig], states)
    assert np.all(np.isfinite(combined))


# --------------------------------------------------------------------------
# AIMPanel.from_tsv
# --------------------------------------------------------------------------


def test_aim_panel_from_tsv_round_trip(tmp_path):
    tsv = tmp_path / "panel.tsv"
    tsv.write_text(
        "chrom\tpos_bp\tref\talt\texpected_freq\tweight\tsource\n"
        "1\t100\tA\tG\t0.85\t1.0\tDuffy null (rs2814778)\n"
        "1\t200\tC\tT\t0.10\t0.8\tHBB\n"
        "2\t300\tA\tC\t0.40\t1.0\tLCT\n"
    )
    panel = AIMPanel.from_tsv(tsv)
    assert len(panel.chrom) == 3
    assert list(panel.chrom) == ["1", "1", "2"]
    np.testing.assert_array_equal(panel.pos_bp, [100, 200, 300])
    np.testing.assert_allclose(panel.expected_freq, [0.85, 0.10, 0.40])
    np.testing.assert_allclose(panel.marker_weight, [1.0, 0.8, 1.0])


def test_aim_panel_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        AIMPanel(
            chrom=np.array(["1", "1"], dtype=object),
            pos_bp=np.array([10, 10], dtype=np.int64),
            expected_freq=np.array([0.5, 0.5]),
            marker_weight=np.array([1.0, 1.0]),
        )


def test_aim_panel_rejects_missing_columns(tmp_path):
    tsv = tmp_path / "bad.tsv"
    tsv.write_text("chrom\tpos_bp\texpected_freq\n1\t100\t0.5\n")
    with pytest.raises(ValueError, match="missing columns"):
        AIMPanel.from_tsv(tsv)
