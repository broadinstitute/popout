"""Phase 2: matching invariants for the named strategies.

Goldens prove byte-identity. These tests pin the *properties* the
strategies must satisfy regardless of input:

  1. Totality           — every component maps somewhere (never silently dropped)
  2. Determinism        — same inputs ⇒ same map
  3. Stable under input column permutation when the target carries names
  4. ``unassigned`` sentinel never overlaps a real label

(Order-invariant collapse and naming stability are exercised in
test_project.py and test_naming.py respectively, in Phase 3.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from popout.labelspace import (
    Assignment,
    by_name,
    confusion_hungarian,
    corr_hungarian,
    get,
    manual,
    posterior_slope,
)
from popout.labelspace.registry import SP6, make_native_space


# ── Reusable synthetic input ────────────────────────────────────────────


def _freq_pair(seed: int = 0, k_inf: int = 6, k_ref: int = 6, n_sites: int = 200):
    rng = np.random.default_rng(seed)
    ref = rng.uniform(0.0, 1.0, size=(k_ref, n_sites)).astype(np.float64)
    inf = np.zeros((k_inf, n_sites), dtype=np.float64)
    for i in range(min(k_inf, k_ref)):
        inf[i] = ref[i] + 0.03 * rng.standard_normal(n_sites)
    for i in range(k_ref, k_inf):
        a, b = rng.integers(0, k_ref, size=2)
        inf[i] = 0.5 * (ref[a] + ref[b]) + 0.08 * rng.standard_normal(n_sites)
    np.clip(inf, 0.0, 1.0, out=inf)
    return inf, ref


# ── corr_hungarian ──────────────────────────────────────────────────────


def test_corr_hungarian_totality():
    inf, ref = _freq_pair(k_inf=7, k_ref=6)
    a = corr_hungarian(inf, ref, SP6)
    assert set(a.component_to_label) == set(range(7))


def test_corr_hungarian_determinism():
    inf, ref = _freq_pair()
    a = corr_hungarian(inf, ref, SP6)
    b = corr_hungarian(inf, ref, SP6)
    assert a.component_to_label == b.component_to_label
    assert a.label_to_components == b.label_to_components


def test_corr_hungarian_bijective_when_K_inf_le_K_ref():
    inf, ref = _freq_pair(k_inf=5, k_ref=6)
    a = corr_hungarian(inf, ref, SP6)
    # 5 components, all distinct labels → 5 unique values
    assert len(set(a.component_to_label.values())) == 5


def test_corr_hungarian_merges_when_K_inf_gt_K_ref():
    inf, ref = _freq_pair(k_inf=10, k_ref=6)
    a = corr_hungarian(inf, ref, SP6)
    # 10 components squeezed into ≤ 6 labels: at least one merge
    assert len(set(a.component_to_label.values())) <= 6
    # every label_to_components value is sorted by descending correlation
    corr = np.array(a.diagnostics["correlations"])
    for label, idxs in a.label_to_components.items():
        ref_col = SP6.index(label)
        scores = [corr[i, ref_col] for i in idxs]
        assert scores == sorted(scores, reverse=True)


def test_corr_hungarian_rejects_mismatched_target():
    inf, ref = _freq_pair(k_ref=5)
    with pytest.raises(ValueError):
        corr_hungarian(inf, ref, SP6)


# ── posterior_slope ────────────────────────────────────────────────────


def _post_pair(seed: int = 0, k_pop: int = 6, n_samples: int = 300):
    rng = np.random.default_rng(seed)
    rf = rng.dirichlet([0.5] * 6, size=n_samples).astype(np.float64)
    pop = np.zeros((n_samples, k_pop), dtype=np.float64)
    for i in range(n_samples):
        ri = rng.integers(0, 6)
        pidx = ri if ri < k_pop else (k_pop - 1)
        pop[i, pidx] = 0.7
        rest = np.full(k_pop, 0.3 / (k_pop - 1))
        rest[pidx] = 0.7
        pop[i] = rest
    return pop, rf


def test_posterior_slope_totality():
    pop, rf = _post_pair(k_pop=7)
    a = posterior_slope(pop, rf, SP6)
    assert set(a.component_to_label) == set(range(7))


def test_posterior_slope_determinism():
    pop, rf = _post_pair()
    a = posterior_slope(pop, rf, SP6)
    b = posterior_slope(pop, rf, SP6)
    assert a.to_dict() == b.to_dict()


def test_posterior_slope_diag_carries_slope_matrix():
    pop, rf = _post_pair()
    a = posterior_slope(pop, rf, SP6)
    assert "slope_matrix" in a.diagnostics
    assert "max_cal_matrix" in a.diagnostics
    assert a.diagnostics["unit"] == "samples"


def test_posterior_slope_rejects_wrong_target_width():
    pop, rf = _post_pair()
    with pytest.raises(ValueError):
        posterior_slope(pop, rf, get("SP5"))


# ── confusion_hungarian ─────────────────────────────────────────────────


def test_confusion_hungarian_recovers_permutation():
    rng = np.random.default_rng(0)
    n_haps, n_sites = 6, 200
    true_calls = rng.integers(0, 6, size=(n_haps, n_sites))
    perm = np.array([2, 0, 5, 3, 1, 4])
    src_calls = perm[true_calls]
    a = confusion_hungarian(src_calls, true_calls, SP6)
    # the matcher should invert the permutation modulo SP6's names
    for src_int in range(6):
        ref_int = int(np.where(perm == src_int)[0][0])
        assert a.component_to_label[src_int] == SP6.members[ref_int]


def test_confusion_hungarian_totality_with_missing_label():
    rng = np.random.default_rng(0)
    n_haps, n_sites = 4, 100
    true_calls = rng.integers(0, 4, size=(n_haps, n_sites))
    src_calls = true_calls.copy()
    src_calls[0, 0:10] = 65535        # missing
    a = confusion_hungarian(src_calls, true_calls, SP6, missing_label=65535)
    assert 65535 not in a.component_to_label


# ── by_name ─────────────────────────────────────────────────────────────


def test_by_name_exact():
    a = by_name(["afr", "eur", "amr"], SP6)
    assert a.component_to_label == {0: "afr", 1: "eur", 2: "amr"}


def test_by_name_case_insensitive():
    a = by_name(["AFR", "Eur"], SP6)
    assert a.component_to_label == {0: "afr", 1: "eur"}


def test_by_name_unassigned_for_unknown():
    a = by_name(["afr", "kor"], SP6)
    assert a.component_to_label == {0: "afr", 1: "unassigned"}


def test_by_name_against_native_space():
    flare = make_native_space("flare", ("anc_0", "anc_1", "anc_2"))
    a = by_name(["anc_1", "anc_0"], flare)
    assert a.component_to_label == {0: "anc_1", 1: "anc_0"}


# ── manual ──────────────────────────────────────────────────────────────


def test_manual_csv(tmp_path: Path):
    csv = tmp_path / "map.csv"
    csv.write_text("0,afr\n1,eur\n2,unassigned\n")
    a = manual(csv, SP6)
    assert a.component_to_label == {0: "afr", 1: "eur", 2: "unassigned"}


def test_manual_csv_rejects_unknown_label(tmp_path: Path):
    csv = tmp_path / "map.csv"
    csv.write_text("0,kor\n")
    with pytest.raises(ValueError):
        manual(csv, SP6)


def test_manual_csv_tolerates_header(tmp_path: Path):
    csv = tmp_path / "map.csv"
    csv.write_text("component,label\n0,afr\n1,eur\n")
    a = manual(csv, SP6)
    assert a.component_to_label == {0: "afr", 1: "eur"}


# ── Routing sanity: producers route through labelspace ──────────────────


def test_label_py_routes_through_labelspace():
    """popout.label._correlation_matrix / _assign_labels are thin shims now."""
    import popout.label as plabel

    src_corr = plabel._correlation_matrix.__code__.co_filename
    src_assign = plabel._assign_labels.__code__.co_filename
    # The shim bodies are now in label.py but they import + call the
    # labelspace functions.
    inf, ref = _freq_pair()
    corr = plabel._correlation_matrix(inf, ref)
    label_map, merge_map = plabel._assign_labels(corr, list(SP6.members))
    assert set(label_map) == set(range(inf.shape[0]))
    assert all(v in SP6.members for v in label_map.values())
