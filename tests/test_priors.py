"""Tests for popout.priors — YAML loading, Beta solver, fingerprinting."""

from __future__ import annotations

import math
import textwrap

import pytest
from scipy.stats import beta as scipy_beta

from popout.priors import (
    ComponentTPrior,
    Priors,
    load_priors,
    prior_to_beta,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r_of_T(T: float, mps: float) -> float:
    return 1.0 - math.exp(-T * mps)


def _write_yaml(tmp_path, body: str):
    p = tmp_path / "priors.yaml"
    p.write_text(textwrap.dedent(body).lstrip())
    return p


# ---------------------------------------------------------------------------
# Beta-solver tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "gen_mean, gen_lo, gen_hi, mps",
    [
        (7.0, 4.0, 12.0, 1.2e-4),
        (2.0, 1.0, 4.0, 1.2e-4),
        (20.0, 10.0, 40.0, 1.2e-4),
        (5.0, 3.0, 8.0, 5e-4),  # different cohort grid density
    ],
)
def test_prior_to_beta_percentiles_match(gen_mean, gen_lo, gen_hi, mps):
    """Beta(α, β) has its 5th/95th percentiles at the requested r values."""
    alpha, beta = prior_to_beta(gen_mean, gen_lo, gen_hi, mps)

    p05 = scipy_beta.ppf(0.05, alpha, beta)
    p95 = scipy_beta.ppf(0.95, alpha, beta)

    r_lo = _r_of_T(gen_lo, mps)
    r_hi = _r_of_T(gen_hi, mps)

    # 1% relative tolerance per the spec.
    assert abs(p05 - r_lo) / r_lo < 0.01, f"p05={p05} r_lo={r_lo}"
    assert abs(p95 - r_hi) / r_hi < 0.01, f"p95={p95} r_hi={r_hi}"


def test_prior_to_beta_rejects_bad_morgans_per_step():
    with pytest.raises(ValueError, match="morgans_per_step"):
        prior_to_beta(5.0, 2.0, 10.0, morgans_per_step=0.0)
    with pytest.raises(ValueError, match="morgans_per_step"):
        prior_to_beta(5.0, 2.0, 10.0, morgans_per_step=-1.0)


# ---------------------------------------------------------------------------
# ComponentTPrior validation
# ---------------------------------------------------------------------------

def test_component_validation_rejects_lo_geq_mean():
    with pytest.raises(ValueError, match="gen_lo < gen_mean"):
        ComponentTPrior(
            component_idx=0, gen_mean=5.0, gen_lo=5.0, gen_hi=10.0,
            alpha=2.0, beta=2.0,
        )


def test_component_validation_rejects_mean_geq_hi():
    with pytest.raises(ValueError, match="gen_mean < gen_hi"):
        ComponentTPrior(
            component_idx=0, gen_mean=10.0, gen_lo=2.0, gen_hi=10.0,
            alpha=2.0, beta=2.0,
        )


def test_component_validation_rejects_negative():
    with pytest.raises(ValueError, match="gen_lo"):
        ComponentTPrior(
            component_idx=0, gen_mean=5.0, gen_lo=-1.0, gen_hi=10.0,
            alpha=2.0, beta=2.0,
        )


def test_component_validation_rejects_negative_idx():
    with pytest.raises(ValueError, match="component_idx"):
        ComponentTPrior(
            component_idx=-1, gen_mean=5.0, gen_lo=1.0, gen_hi=10.0,
            alpha=2.0, beta=2.0,
        )


def test_component_validation_rejects_nonpositive_alpha_beta():
    with pytest.raises(ValueError, match="positive"):
        ComponentTPrior(
            component_idx=0, gen_mean=5.0, gen_lo=1.0, gen_hi=10.0,
            alpha=0.0, beta=2.0,
        )


# ---------------------------------------------------------------------------
# YAML round-trip and Priors container
# ---------------------------------------------------------------------------

VALID_YAML = """
morgans_per_step: 1.2e-4
components:
  - component_idx: 7
    gen_mean: 7
    gen_lo: 4
    gen_hi: 12
    source: "AFR primary phase"
  - component_idx: 2
    gen_mean: 2
    gen_lo: 1
    gen_hi: 4
    source: "Recent EUR immigration"
"""


def test_load_priors_basic(tmp_path):
    p = _write_yaml(tmp_path, VALID_YAML)
    priors = load_priors(p)

    assert priors.morgans_per_step == pytest.approx(1.2e-4)
    assert len(priors.components) == 2
    assert priors.has(7)
    assert priors.has(2)
    assert not priors.has(99)

    afr = priors.get(7)
    assert afr is not None
    assert afr.gen_mean == 7
    assert afr.gen_lo == 4
    assert afr.gen_hi == 12
    assert afr.source == "AFR primary phase"
    # Beta materialized
    assert afr.alpha > 0
    assert afr.beta > 0


def test_load_priors_missing_morgans_per_step(tmp_path):
    p = _write_yaml(tmp_path, """
        components:
          - component_idx: 0
            gen_mean: 5
            gen_lo: 1
            gen_hi: 10
    """)
    with pytest.raises(ValueError, match="morgans_per_step"):
        load_priors(p)


def test_load_priors_missing_components(tmp_path):
    p = _write_yaml(tmp_path, "morgans_per_step: 1.2e-4\n")
    with pytest.raises(ValueError, match="components"):
        load_priors(p)


def test_load_priors_duplicate_idx(tmp_path):
    p = _write_yaml(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 5, gen_lo: 1, gen_hi: 10}
          - {component_idx: 0, gen_mean: 8, gen_lo: 4, gen_hi: 16}
    """)
    with pytest.raises(ValueError, match="duplicate"):
        load_priors(p)


def test_load_priors_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_priors(tmp_path / "does_not_exist.yaml")


def test_load_priors_validates_bad_range(tmp_path):
    p = _write_yaml(tmp_path, """
        morgans_per_step: 1.2e-4
        components:
          - {component_idx: 0, gen_mean: 5, gen_lo: 8, gen_hi: 10}
    """)
    with pytest.raises(ValueError, match="gen_lo"):
        load_priors(p)


# ---------------------------------------------------------------------------
# Fingerprint
# ---------------------------------------------------------------------------

def test_fingerprint_stable_across_reads(tmp_path):
    p = _write_yaml(tmp_path, VALID_YAML)
    fp1 = load_priors(p).fingerprint
    fp2 = load_priors(p).fingerprint
    assert fp1 == fp2
    assert len(fp1) == 64  # sha256 hex


def test_fingerprint_changes_when_file_changes(tmp_path):
    p = _write_yaml(tmp_path, VALID_YAML)
    fp1 = load_priors(p).fingerprint

    # Modify only morgans_per_step
    p.write_text(VALID_YAML.replace("1.2e-4", "5e-4"))
    fp2 = load_priors(p).fingerprint
    assert fp1 != fp2


def test_fingerprint_changes_with_component_change(tmp_path):
    p = _write_yaml(tmp_path, VALID_YAML)
    fp1 = load_priors(p).fingerprint

    # Tweak gen_mean of one component
    p.write_text(VALID_YAML.replace("gen_mean: 7", "gen_mean: 8"))
    fp2 = load_priors(p).fingerprint
    assert fp1 != fp2
