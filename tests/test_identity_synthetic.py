"""Synthetic identity-discrimination tests (Step 9b/9c).

These tests validate that the framework's identity-matching machinery
actually identifies populations correctly. They are the core test of
the redesign — the property that priors_v1 failed and that this entire
framework exists to satisfy.

Setup
-----
Build a K=6 cohort whose component allele frequencies are sampled from
synthetic 1KG-superpop-style "truth" vectors plus per-component noise.
Build six priors, each with an :class:`FSTReferenceSignature` pointing
at one truth vector. Run :func:`assign_priors_to_components`.

Acceptance criteria (9b): each prior's *argmax* component matches the
truth-vector index.

Robustness (9c): when one component's freq is drawn 50/50 from two
superpops, the two corresponding priors both put non-trivial weight on
that component — graceful degradation under ambiguous identity.
"""

from __future__ import annotations

import numpy as np
import pytest

from popout.identity import ComponentState, FSTReferenceSignature
from popout.identity_assignment import assign_priors_to_components
from popout.prior_spec import (
    LinearAnnealingSchedule,
    Prior,
    Priors,
)


# --------------------------------------------------------------------------
# Synthetic 1KG-style data
# --------------------------------------------------------------------------


_SUPERPOPS = ("AFR", "EUR", "EAS", "AMR", "SAS", "MID")


def _make_synthetic_truths(n_sites: int, seed: int = 1):
    """One reference allele frequency vector per superpop.

    Each superpop has its own random freq draw with a small per-site
    correlation between populations (so the populations aren't
    completely independent — which would make the test trivial).
    """
    rng = np.random.default_rng(seed)
    # Shared baseline freqs per site, then per-superpop offsets.
    baseline = rng.uniform(0.05, 0.95, n_sites)
    offsets = rng.normal(0.0, 0.25, (len(_SUPERPOPS), n_sites))
    truths = np.clip(baseline[None, :] + offsets, 0.01, 0.99)
    return truths  # (P, n_sites)


def _make_priors_from_truths(
    truths: np.ndarray, pos_bp: np.ndarray,
) -> Priors:
    """Six priors, one FSTReferenceSignature each, against one row of
    `truths`.

    The :class:`FSTReferenceSignature` is multi-chrom-aware so we set
    `ref_chrom` to a single-element array repeated to the full length.
    """
    n_sites = pos_bp.shape[0]
    chrom_arr = np.array(["1"] * n_sites, dtype=object)
    plist = []
    for i, name in enumerate(_SUPERPOPS):
        sig = FSTReferenceSignature(
            ref_freq=truths[i].astype(np.float64),
            ref_pos_bp=pos_bp.astype(np.int64),
            ref_chrom=chrom_arr,
            ref_name=f"1KG_{name}",
        )
        plist.append(
            Prior(
                name=name,
                identity_signatures=(sig,),
                gen_mean=5.0, gen_lo=2.0, gen_hi=12.0,
                alpha=2.0, beta=10.0,
            )
        )
    return Priors(
        priors=tuple(plist),
        morgans_per_step=1.2e-4,
        annealing=LinearAnnealingSchedule(
            tau_start=1.0, tau_end=0.1, ramp_iters=10,
        ),
        fingerprint="x" * 64,
        source_path="<synthetic>",
    )


def _make_states(
    truths: np.ndarray,
    pos_bp: np.ndarray,
    *,
    seed: int = 2,
    noise_sd: float = 0.03,
) -> list[ComponentState]:
    """One ComponentState per truth row, equal to truths[k] + noise."""
    rng = np.random.default_rng(seed)
    n_sites = pos_bp.shape[0]
    K = truths.shape[0]
    states = []
    for k in range(K):
        freq = np.clip(truths[k] + rng.normal(0.0, noise_sd, n_sites),
                       0.001, 0.999)
        states.append(ComponentState(
            freq=freq, mu=1.0 / K,
            pos_bp=pos_bp, chrom="1",
        ))
    return states


# --------------------------------------------------------------------------
# 9b — identity discrimination
# --------------------------------------------------------------------------


def test_identity_discrimination_six_superpops_argmax_matches_truth():
    """Each prior's argmax component is the matching truth index."""
    n_sites = 1500
    pos_bp = np.arange(10_000, 10_000 + n_sites, dtype=np.int64)
    truths = _make_synthetic_truths(n_sites, seed=11)
    priors = _make_priors_from_truths(truths, pos_bp)
    states = _make_states(truths, pos_bp, seed=12, noise_sd=0.03)

    # Final iteration so tau is at tau_end (cool, sharp argmax).
    weights = assign_priors_to_components(priors, states, iteration=10)
    assert weights.shape == (len(_SUPERPOPS), len(_SUPERPOPS))

    for p, name in enumerate(_SUPERPOPS):
        k_dom = int(np.argmax(weights[p]))
        assert k_dom == p, (
            f"prior {name} argmax landed on comp {k_dom} (truth was {p}); "
            f"weights row = {weights[p]}"
        )


def test_identity_discrimination_dominant_weight_above_uniform():
    """Each dominant weight should clear the uniform-1/K baseline by a
    healthy margin — otherwise the framework has not actually
    identified anything."""
    n_sites = 1500
    pos_bp = np.arange(10_000, 10_000 + n_sites, dtype=np.int64)
    truths = _make_synthetic_truths(n_sites, seed=13)
    priors = _make_priors_from_truths(truths, pos_bp)
    states = _make_states(truths, pos_bp, seed=14, noise_sd=0.03)

    weights = assign_priors_to_components(priors, states, iteration=10)
    K = weights.shape[1]
    uniform = 1.0 / K
    for p, name in enumerate(_SUPERPOPS):
        w_dom = float(weights[p, p])
        assert w_dom > 3 * uniform, (
            f"prior {name}: dominant weight {w_dom:.3f} not clearly "
            f"above uniform {uniform:.3f}"
        )


# --------------------------------------------------------------------------
# 9c — robustness on an admixed component
# --------------------------------------------------------------------------


def test_admixed_component_attracts_both_priors():
    """A genuinely admixed component (50/50 AFR-EUR) and *no* pure AFR
    or EUR component: both AFR and EUR priors should latch onto the
    admixed one (their only viable match). This is the documented
    "racing priors" diagnostic event — visible in the assignment dump,
    not a constraint violation.
    """
    n_sites = 1500
    pos_bp = np.arange(10_000, 10_000 + n_sites, dtype=np.int64)
    truths = _make_synthetic_truths(n_sites, seed=21)
    priors = _make_priors_from_truths(truths, pos_bp)

    # K=6 components, but no pure AFR or EUR — only an AFR/EUR admixed
    # component, the other 5 are pure of EAS/AMR/SAS/MID + one random.
    rng = np.random.default_rng(23)
    states = []
    chrom_arr = "1"
    K = 6

    # comp 0: 50/50 AFR (idx 0) + EUR (idx 1) admixed
    admixed_freq = np.clip(
        0.5 * truths[0] + 0.5 * truths[1] + rng.normal(0.0, 0.03, n_sites),
        0.001, 0.999,
    )
    states.append(ComponentState(
        freq=admixed_freq, mu=1.0 / K, pos_bp=pos_bp, chrom=chrom_arr,
    ))
    # comps 1..4: pure EAS, AMR, SAS, MID (truths[2..5])
    for k in (2, 3, 4, 5):
        freq = np.clip(truths[k] + rng.normal(0.0, 0.03, n_sites),
                       0.001, 0.999)
        states.append(ComponentState(
            freq=freq, mu=1.0 / K, pos_bp=pos_bp, chrom=chrom_arr,
        ))
    # comp 5: random freqs unrelated to any superpop
    states.append(ComponentState(
        freq=rng.uniform(0.05, 0.95, n_sites),
        mu=1.0 / K, pos_bp=pos_bp, chrom=chrom_arr,
    ))

    weights = assign_priors_to_components(priors, states, iteration=10)
    # AFR (idx 0) and EUR (idx 1) priors both target comp 0 — the only
    # AFR-or-EUR-ish component in the cohort.
    afr_w_on_admix = float(weights[0, 0])
    eur_w_on_admix = float(weights[1, 0])
    assert afr_w_on_admix > 0.5, (
        f"AFR prior failed to attach to admixed comp 0: {weights[0]}"
    )
    assert eur_w_on_admix > 0.5, (
        f"EUR prior failed to attach to admixed comp 0: {weights[1]}"
    )

    # Other priors should still find their pure matches. EAS prior →
    # comp 1, AMR → comp 2, SAS → comp 3, MID → comp 4.
    expected_argmax = {
        "EAS": 1, "AMR": 2, "SAS": 3, "MID": 4,
    }
    for p, name in enumerate(_SUPERPOPS):
        if name in expected_argmax:
            k_dom = int(np.argmax(weights[p]))
            assert k_dom == expected_argmax[name], (
                f"prior {name} wandered to comp {k_dom}, expected "
                f"{expected_argmax[name]}; weights={weights[p]}"
            )
