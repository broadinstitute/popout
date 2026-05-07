"""End-to-end plumbing of priors through ``run_em``.

These integration tests verify that:
  1. priors=None reproduces the pre-priors run_em behavior bit-for-bit.
  2. priors!=None changes the fitted model and produces gen_per_comp.
  3. The mutex with --per-hap-T is enforced at the AncestryModel level.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from popout.em import run_em
from popout.simulate import simulate_admixed
from tests.conftest import make_priors_uniform


@pytest.fixture
def sim_chrom():
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=8.0, chrom_length_cm=40.0, rng_seed=11,
    )
    return chrom_data


def _model_signature(model):
    return (
        np.array(model.mu),
        np.array(model.allele_freq),
        float(model.gen_since_admix),
    )


def test_priors_none_matches_baseline(sim_chrom):
    """priors=None must produce the identical model to pre-priors behavior."""
    res_baseline = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=2, gen_since_admix=8.0, rng_seed=0,
    )
    res_none = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=2, gen_since_admix=8.0, rng_seed=0,
        priors=None,
    )
    mu_b, af_b, T_b = _model_signature(res_baseline.model)
    mu_n, af_n, T_n = _model_signature(res_none.model)

    np.testing.assert_array_equal(mu_b, mu_n)
    np.testing.assert_array_equal(af_b, af_n)
    assert T_b == T_n
    # Baseline must NOT carry per-component T.
    assert res_baseline.model.gen_per_comp is None
    assert res_none.model.gen_per_comp is None


def test_priors_set_changes_fitted_model(sim_chrom):
    """Supplying priors yields a different model with gen_per_comp set."""
    priors = make_priors_uniform([(2, 1, 4), (50, 30, 80)])

    res_priors = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=3, gen_since_admix=8.0, rng_seed=0,
        priors=priors,
    )

    assert res_priors.model.gen_per_comp is not None
    assert res_priors.model.gen_per_comp.shape == (3,)
    gpc = np.array(res_priors.model.gen_per_comp)
    assert np.isfinite(gpc).all()
    assert (gpc >= 1.0).all() and (gpc <= 1000.0).all()


def test_priors_per_hap_T_mutex_at_model_level():
    """Bundling per-hap-T with priors raises at AncestryModel level."""
    from popout.datatypes import AncestryModel
    with pytest.raises(ValueError, match="gen_per_comp"):
        AncestryModel(
            n_ancestries=2,
            mu=jnp.array([0.5, 0.5]),
            gen_since_admix=10.0,
            allele_freq=jnp.zeros((2, 5)),
            gen_per_comp=jnp.array([5.0, 10.0]),
            gen_per_hap=jnp.full((10,), 10.0),
        )


# --------------------------------------------------------------------------
# Phase 2 — compute_panel_freqs_per_comp + sidecar plumbing
# --------------------------------------------------------------------------


def test_compute_panel_freqs_per_comp_matches_expected_weighted_average():
    """The mu-weighted formula is a weighted average per component:
    panel_freq[k, l] = sum_h γ̄[h,k] * geno[h, l] / sum_h γ̄[h,k].
    Construct a tiny case with known γ and verify the output."""
    from popout.em import compute_panel_freqs_per_comp
    from popout.pgen_io import PanelGeno

    # 4 haps, 3 panel positions, K=2.
    # hap 0,1 are pure component 0 (γ̄=[1,0])
    # hap 2,3 are pure component 1 (γ̄=[0,1])
    mu_per_hap_sum = np.array([
        [10.0, 0.0],
        [10.0, 0.0],
        [0.0, 10.0],
        [0.0, 10.0],
    ])
    geno = np.array([
        [1, 0, 1],   # hap 0
        [1, 1, 1],   # hap 1
        [0, 1, 0],   # hap 2
        [0, 0, 0],   # hap 3
    ], dtype=np.uint8)
    panel = PanelGeno(
        geno=geno,
        chrom=np.array(["1", "2", "3"], dtype=object),
        pos_bp=np.array([100, 200, 300], dtype=np.int64),
    )

    out = compute_panel_freqs_per_comp(mu_per_hap_sum, n_sites_chrom=10, panel_geno=panel)

    # Component 0: weighted-average geno over haps 0,1 → (1, 0.5, 1).
    assert out[0][("1", 100)] == pytest.approx(1.0)
    assert out[0][("2", 200)] == pytest.approx(0.5)
    assert out[0][("3", 300)] == pytest.approx(1.0)
    # Component 1: weighted-average geno over haps 2,3 → (0, 0.5, 0).
    assert out[1][("1", 100)] == pytest.approx(0.0)
    assert out[1][("2", 200)] == pytest.approx(0.5)
    assert out[1][("3", 300)] == pytest.approx(0.0)


def test_compute_panel_freqs_per_comp_rejects_hap_count_mismatch():
    """The H dimensions of mu_per_hap_sum and panel_geno.geno must match;
    if the upstream extraction emitted a different cohort the read_panel_geno
    psam check would already have caught it, but defend in depth."""
    from popout.em import compute_panel_freqs_per_comp
    from popout.pgen_io import PanelGeno

    mu_per_hap_sum = np.zeros((4, 2))
    panel = PanelGeno(
        geno=np.zeros((6, 3), dtype=np.uint8),  # H=6 ≠ 4
        chrom=np.array(["1", "1", "1"], dtype=object),
        pos_bp=np.array([1, 2, 3], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="hap count"):
        compute_panel_freqs_per_comp(mu_per_hap_sum, n_sites_chrom=10, panel_geno=panel)


def test_run_em_panel_geno_changes_priors_assignment(sim_chrom):
    """End-to-end: run_em with priors+panel_geno produces a different
    fitted model than priors alone, because the panel signal reaches
    the M-step's prior-assignment via the μ-weighted off-chrom freqs.

    Builds priors with real AIMSignatures pointing at off-chrom
    panel positions — without --panel-geno those rows score 0 (no
    on-chrom overlap); with --panel-geno they get μ-weighted freqs
    and the score becomes a non-trivial discriminator.
    """
    from popout.identity import AIMPanel, AIMSignature
    from popout.pgen_io import PanelGeno
    from popout.prior_spec import (
        LinearAnnealingSchedule, Prior, Priors, prior_to_beta,
    )

    # Two off-chrom AIM panels — one per prior. Off-chrom positions
    # ensure the on-chrom path scores 0 and panel_geno is the only way
    # to influence the M-step.
    panel_a = AIMPanel(
        chrom=np.array(["99", "99", "99", "99"], dtype=object),
        pos_bp=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        expected_freq=np.array([0.9, 0.9, 0.1, 0.1]),
        marker_weight=np.array([1.0, 1.0, 1.0, 1.0]),
        source="test_a",
    )
    panel_b = AIMPanel(
        chrom=np.array(["99", "99", "99", "99"], dtype=object),
        pos_bp=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
        expected_freq=np.array([0.1, 0.1, 0.9, 0.9]),
        marker_weight=np.array([1.0, 1.0, 1.0, 1.0]),
        source="test_b",
    )

    def _make_prior(name, panel, mean, lo, hi):
        a, b = prior_to_beta(mean, lo, hi, 1.2e-4)
        return Prior(
            name=name,
            identity_signatures=(AIMSignature(panel=panel),),
            gen_mean=mean, gen_lo=lo, gen_hi=hi,
            alpha=a, beta=b,
        )

    priors = Priors(
        priors=(
            _make_prior("PA", panel_a, 2, 1, 4),
            _make_prior("PB", panel_b, 50, 30, 80),
        ),
        morgans_per_step=1.2e-4,
        annealing=LinearAnnealingSchedule(1.0, 0.1, 10),
        fingerprint="x" * 64,
        source_path="<test>",
    )

    # Run baseline: priors only, no sidecar. AIM panels are all on
    # chrom "99", which doesn't match sim_chrom.chrom, so AIMSignature
    # returns 0 for every component → soft assignment is uniform.
    res_no_panel = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=3, gen_since_admix=8.0, rng_seed=0,
        priors=priors,
    )

    # Sidecar with the same off-chrom positions; genotypes built so
    # half the cohort matches panel_a's expected (high at 1000/2000)
    # and the other half matches panel_b's (high at 3000/4000).
    H = sim_chrom.geno.shape[0]
    half = H // 2
    sidecar_geno = np.zeros((H, 4), dtype=np.uint8)
    sidecar_geno[:half, 0] = 1   # high at 1000 in first half
    sidecar_geno[:half, 1] = 1   # high at 2000 in first half
    sidecar_geno[half:, 2] = 1   # high at 3000 in second half
    sidecar_geno[half:, 3] = 1   # high at 4000 in second half
    panel = PanelGeno(
        geno=sidecar_geno,
        chrom=np.array(["99"] * 4, dtype=object),
        pos_bp=np.array([1000, 2000, 3000, 4000], dtype=np.int64),
    )

    res_with_panel = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=3, gen_since_admix=8.0, rng_seed=0,
        priors=priors, panel_geno=panel,
    )

    # gen_per_comp should differ between the two runs, proving the panel
    # signal influenced the prior assignment.
    gpc_no_panel = np.array(res_no_panel.model.gen_per_comp)
    gpc_with_panel = np.array(res_with_panel.model.gen_per_comp)
    assert not np.allclose(gpc_no_panel, gpc_with_panel), (
        f"panel_geno had no effect on gen_per_comp:\n"
        f"  no_panel:   {gpc_no_panel}\n"
        f"  with_panel: {gpc_with_panel}"
    )


def test_read_panel_geno_psam_set_mismatch_raises(tmp_path):
    """Same-count psams with different IIDs raise a clear error
    (lists the missing samples)."""
    from popout.pgen_io import read_panel_geno
    # We can construct just the psam files for this assertion since
    # read_panel_geno checks the IID set BEFORE opening the PGEN.
    # But the function also expects pgen/pvar to exist; build empty
    # placeholders that reach the assertion point.
    pgen = tmp_path / "p.pgen"
    pvar = tmp_path / "p.pvar"
    psam = tmp_path / "p.psam"
    pgen.write_bytes(b"")
    pvar.write_text("#CHROM\tPOS\tID\tREF\tALT\n1\t100\t.\tA\tG\n")
    psam.write_text("#IID\nS1\nS2\nS3\n")

    with pytest.raises(ValueError, match="sample SET does not match"):
        read_panel_geno(str(tmp_path / "p"), expected_sample_iids=["S1", "S2", "S99"])


def test_run_em_panel_geno_requires_priors(sim_chrom):
    """panel_geno without priors is meaningless — the helper that uses
    the sidecar (M-step priors-assignment block) wouldn't run. This is
    enforced at the CLI level; assert the run_em path doesn't blow up
    in some weird way (it just ignores panel_geno when priors is None)."""
    from popout.pgen_io import PanelGeno

    panel = PanelGeno(
        geno=sim_chrom.geno[:, :2].astype(np.uint8),
        chrom=np.array(["99", "99"], dtype=object),
        pos_bp=np.array([1, 2], dtype=np.int64),
    )

    # No priors: panel_geno is silently ignored. Run completes normally.
    res = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=2, gen_since_admix=8.0, rng_seed=0,
        priors=None, panel_geno=panel,
    )
    assert res.model.gen_per_comp is None
