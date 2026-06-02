"""Deterministic synthetic inputs for label-space characterization tests.

All inputs are seeded so the goldens captured in Phase 0 are reproducible
without any external data. The fixture functions are imported by
``test_goldens.py`` and the per-phase regression tests in subsequent
phases of the label-space retrofit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SEED = 20260602
SP6: tuple[str, ...] = ("afr", "amr", "eas", "eur", "mid", "sas")
RYE_NATIVE: tuple[str, ...] = ("eur", "eas", "amr", "afr", "sas")

GOLDENS_DIR = Path(__file__).resolve().parent / "goldens"
GOLDENS_DIR.mkdir(exist_ok=True)


def _rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


# ── corr_hungarian inputs (popout/label.py) ──────────────────────────────


def synthetic_freq_inputs(k_inf: int = 8, k_ref: int = 6, n_sites: int = 200):
    """Inferred and reference allele-frequency matrices.

    Constructed so the first ``k_ref`` inferred rows are *close to* the
    matching ref rows and the extra ``k_inf - k_ref`` rows are noisy
    mixtures — exercises both the Hungarian and the merge branches of
    ``_assign_labels``.
    """
    rng = _rng()
    ref = rng.uniform(0.0, 1.0, size=(k_ref, n_sites)).astype(np.float64)
    inf = np.empty((k_inf, n_sites), dtype=np.float64)
    for i in range(k_ref):
        inf[i] = ref[i] + 0.05 * rng.standard_normal(n_sites)
    for i in range(k_ref, k_inf):
        a, b = rng.integers(0, k_ref, size=2)
        inf[i] = 0.5 * (ref[a] + ref[b]) + 0.10 * rng.standard_normal(n_sites)
    np.clip(inf, 0.0, 1.0, out=inf)
    return inf, ref, list(SP6[:k_ref])


# ── posterior_slope inputs (compare_to_rf.py) ────────────────────────────


def synthetic_popout_rf_pair(n_samples: int = 400, k_popout: int = 7):
    """A pair of (popout posteriors, rf prob matrix, rf hard calls).

    Each sample is drawn from one of the 6 RF labels (uniform); the
    popout proportions are a noisy 1-of-K_popout encoding that mostly
    aligns with the RF label but for indices ``5, 6`` are deliberate
    mixtures that exercise the slope-override branch.
    """
    rng = _rng()
    rf_labels = SP6
    sample_rf_idx = rng.integers(0, len(rf_labels), size=n_samples)

    rf_prob = rng.dirichlet([0.5] * len(rf_labels), size=n_samples).astype(np.float64)
    for i, ri in enumerate(sample_rf_idx):
        rf_prob[i, ri] += 1.5
        rf_prob[i] /= rf_prob[i].sum()

    popout = np.zeros((n_samples, k_popout), dtype=np.float64)
    for i, ri in enumerate(sample_rf_idx):
        pidx = ri if ri < k_popout - 1 else (k_popout - 1)
        popout[i, pidx] = 0.6 + 0.3 * rng.uniform()
        rest = 1.0 - popout[i, pidx]
        spread = rng.dirichlet([1.0] * (k_popout - 1))
        others = [j for j in range(k_popout) if j != pidx]
        for j, w in zip(others, spread):
            popout[i, j] = rest * w
    rf_hard = np.array([rf_labels[ri] for ri in sample_rf_idx], dtype=object)
    return popout, rf_prob, rf_hard, list(rf_labels)


def write_compare_to_rf_fixture(workdir: Path):
    """Lay out the on-disk inputs that ``compare_to_rf.py`` consumes.

    Returns ``(global_tsv, rf_tsv, out_dir)``. The global.tsv encodes the
    popout posteriors; the rf table encodes hard calls + soft probs.
    """
    popout, rf_prob, rf_hard, rf_labels = synthetic_popout_rf_pair()
    n, k = popout.shape

    workdir.mkdir(parents=True, exist_ok=True)
    sample_ids = [f"s{i:04d}" for i in range(n)]

    global_tsv = workdir / "fixture.global.tsv"
    cols = ["sample"] + [f"ancestry_{j}" for j in range(k)]
    with open(global_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for sid, row in zip(sample_ids, popout):
            f.write(sid + "\t" + "\t".join(f"{v:.6f}" for v in row) + "\n")

    rf_tsv = workdir / "fixture.rf.tsv"
    with open(rf_tsv, "w") as f:
        f.write("research_id\tancestry_pred\tprobabilities\n")
        for sid, hard, probs in zip(sample_ids, rf_hard, rf_prob):
            f.write(f"{sid}\t{hard}\t{json.dumps(list(probs))}\n")

    summary = workdir / "fixture.summary.json"
    summary.write_text(json.dumps({"config": {"method": "popout"}}))

    out_dir = workdir / "compare_to_rf_out"
    return global_tsv, rf_tsv, out_dir


# ── project_to_rf_basis inputs (dx_loaders.py) ───────────────────────────


def synthetic_projection_inputs():
    """A small popout-shaped q matrix plus a synthetic labels.json.

    The labels.json maps 7 popout components onto SP6 with two of them
    (indices 0 and 5) folded into ``afr`` — exercises the multi-source
    sum path of ``project_to_rf_basis``.
    """
    rng = _rng()
    n, k = 5, 7
    q = rng.dirichlet([1.0] * k, size=n).astype(np.float64)
    labels = {
        "rf_ref_labels": list(SP6),
        "popout_to_rf_label": {
            "0": "afr", "1": "amr", "2": "eas",
            "3": "eur", "4": "mid", "5": "afr", "6": "sas",
        },
        "rf_to_popout_components": {
            "afr": [0, 5], "amr": [1], "eas": [2],
            "eur": [3], "mid": [4], "sas": [6],
        },
    }
    return q, labels


def synthetic_rye_q():
    """A small rye-shaped (n, 5) q matrix in RYE_NATIVE order."""
    rng = _rng()
    return rng.dirichlet([1.0] * len(RYE_NATIVE), size=4).astype(np.float64)


def synthetic_rf_q():
    """A small RF-shaped (n, 6) prob matrix in SP6 order."""
    rng = _rng()
    return rng.dirichlet([1.0] * len(SP6), size=4).astype(np.float64)


# ── remap_to_rf_codes inputs (dx_local_align_metrics.py) ─────────────────


def synthetic_tractset_calls():
    """An ``(n_haps, n_sites)`` calls matrix using popout-native codes.

    Source codes 0..6 mirror the projection labels.json above; one site
    is set to MISSING_LABEL (the canonical value used by the loader) so
    the pass-through branch is exercised.
    """
    rng = _rng()
    n_haps, n_sites = 6, 20
    calls = rng.integers(0, 7, size=(n_haps, n_sites), dtype=np.int64)
    # carve a missing-label run
    calls[0, 0:3] = 65535     # MISSING_LABEL used by dx_local_align_metrics
    return calls.astype(np.uint16)
