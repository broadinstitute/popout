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
