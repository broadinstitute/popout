"""End-to-end plumbing of priors through run_em.

These are integration tests on small synthetic data — they verify that:
  1. priors=None reproduces the pre-priors run_em behavior bit-for-bit.
  2. priors!=None changes the fitted model.
"""

from __future__ import annotations

import textwrap

import jax.numpy as jnp
import numpy as np
import pytest

from popout.em import run_em
from popout.priors import load_priors
from popout.simulate import simulate_admixed


@pytest.fixture
def sim_chrom():
    chrom_data, _, _ = simulate_admixed(
        n_samples=80, n_sites=200, n_ancestries=3,
        gen_since_admix=8.0, chrom_length_cm=40.0, rng_seed=11,
    )
    return chrom_data


def _write_priors(tmp_path, body):
    p = tmp_path / "priors.yaml"
    p.write_text(textwrap.dedent(body).lstrip())
    return load_priors(p)


def _model_signature(model):
    """Reduce a fitted AncestryModel to a tuple of arrays for diffing."""
    return (
        np.array(model.mu),
        np.array(model.allele_freq),
        float(model.gen_since_admix),
    )


def test_priors_none_matches_baseline(sim_chrom):
    """priors=None must produce the identical model to pre-change behavior."""
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
    # baseline should have gen_per_comp=None on the fitted model
    assert res_baseline.model.gen_per_comp is None
    assert res_none.model.gen_per_comp is None


def test_priors_set_changes_fitted_model(sim_chrom, tmp_path):
    """Supplying priors yields a different model (non-trivial pull)."""
    priors = _write_priors(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 2, gen_lo: 1, gen_hi: 4}
          - {component_idx: 1, gen_mean: 50, gen_lo: 30, gen_hi: 80}
    """)

    res_none = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=3, gen_since_admix=8.0, rng_seed=0,
    )
    res_priors = run_em(
        sim_chrom, n_ancestries=3, n_em_iter=3, gen_since_admix=8.0, rng_seed=0,
        priors=priors,
    )

    # Priors run produces a per-comp T vector on the fitted model.
    assert res_priors.model.gen_per_comp is not None
    assert res_priors.model.gen_per_comp.shape == (3,)

    # The priors run should have visibly different per-comp T values
    # (component 0 is pulled toward 2, component 1 toward 50).
    gpc = np.array(res_priors.model.gen_per_comp)
    assert gpc[0] < gpc[1], f"per-comp T not separated: {gpc}"


def test_priors_per_hap_T_mutex_at_model_level(sim_chrom, tmp_path):
    """Trying to bundle per-hap-T with priors raises at AncestryModel level."""
    # Setting gen_per_comp explicitly alongside gen_per_hap is rejected by
    # AncestryModel.__post_init__.
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
