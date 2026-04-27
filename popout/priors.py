"""Per-component T priors for popout EM.

Each component k can carry an anthropologically-motivated prior on
generations since admixture: a documented central estimate ``gen_mean``
and a 90% interval ``[gen_lo, gen_hi]`` from demographic uncertainty.
The MAP estimator in :mod:`popout.em` shifts the per-component T-update
by a Beta(α_k, β_k) prior on the per-step transition probability
``r = 1 - exp(-T * morgans_per_step)``.

The Beta parameters are solved at load time so the per-step distance
``morgans_per_step`` is decoupled from the cohort grid density —
the same priors file applies regardless of how densely sites are
sampled.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import yaml
from scipy.optimize import least_squares
from scipy.stats import beta as scipy_beta


@dataclass(frozen=True)
class ComponentTPrior:
    """Anthropogenic prior on generations since admixture for one component.

    Attributes
    ----------
    component_idx : int
        Index of the popout component this prior applies to.
    gen_mean : float
        Documented central estimate of generations since admixture.
    gen_lo, gen_hi : float
        5th and 95th percentiles of demographic uncertainty.
    alpha, beta : float
        Beta(α, β) parameters on the per-step transition probability,
        solved so that the 5th/95th percentiles of the Beta map to
        ``r(gen_hi)`` / ``r(gen_lo)``. Materialized by ``load_priors``.
    source : str
        Free-text citation/justification.
    """

    component_idx: int
    gen_mean: float
    gen_lo: float
    gen_hi: float
    alpha: float
    beta: float
    source: str = ""

    def __post_init__(self) -> None:
        if self.component_idx < 0:
            raise ValueError(
                f"component_idx must be >= 0, got {self.component_idx}"
            )
        if not (0 < self.gen_lo < self.gen_mean < self.gen_hi):
            raise ValueError(
                f"need 0 < gen_lo < gen_mean < gen_hi, got "
                f"({self.gen_lo}, {self.gen_mean}, {self.gen_hi})"
            )
        if self.alpha <= 0 or self.beta <= 0:
            raise ValueError(
                f"Beta parameters must be positive, got "
                f"alpha={self.alpha}, beta={self.beta}"
            )


@dataclass(frozen=True)
class Priors:
    """Container for per-component priors plus file fingerprint."""

    morgans_per_step: float
    components: tuple[ComponentTPrior, ...]
    source_path: str
    fingerprint: str  # sha256 hex of the file contents

    _by_idx: dict[int, ComponentTPrior] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        # Frozen dataclass — bypass __setattr__ to populate the index.
        seen: dict[int, ComponentTPrior] = {}
        for c in self.components:
            if c.component_idx in seen:
                raise ValueError(
                    f"duplicate component_idx {c.component_idx} in priors file"
                )
            seen[c.component_idx] = c
        object.__setattr__(self, "_by_idx", seen)

    def has(self, component_idx: int) -> bool:
        return component_idx in self._by_idx

    def get(self, component_idx: int) -> ComponentTPrior | None:
        return self._by_idx.get(component_idx)


def _r_of_T(T: float, morgans_per_step: float) -> float:
    """Per-step transition probability for generations T."""
    return 1.0 - math.exp(-T * morgans_per_step)


def prior_to_beta(
    gen_mean: float,
    gen_lo: float,
    gen_hi: float,
    morgans_per_step: float,
) -> tuple[float, float]:
    """Solve for Beta(α, β) matching the [gen_lo, gen_hi] percentile band.

    The Beta is on the per-step transition probability
    ``r = 1 - exp(-T * morgans_per_step)``. Higher T → higher r, so
    ``gen_hi`` maps to the *upper* tail of r.

    We require:

    * ``Beta.percentile(0.05) ≈ r(gen_lo)``
    * ``Beta.percentile(0.95) ≈ r(gen_hi)``

    Two equations, two unknowns (α, β). Solved with non-linear
    least-squares in log-space (positivity by construction).
    """
    if morgans_per_step <= 0:
        raise ValueError(
            f"morgans_per_step must be positive, got {morgans_per_step}"
        )

    r_lo = _r_of_T(gen_lo, morgans_per_step)
    r_hi = _r_of_T(gen_hi, morgans_per_step)
    r_mean = _r_of_T(gen_mean, morgans_per_step)

    # Initial guess: method-of-moments-ish. Use mean ≈ α/(α+β) and a
    # concentration that puts ~90% of mass between r_lo and r_hi.
    width = max(r_hi - r_lo, 1e-9)
    # Larger width → less concentrated → smaller α+β.
    # 1/width is a rough scale; clamp to avoid pathological inits.
    concentration = max(min(1.0 / width, 1e6), 4.0)
    alpha0 = max(r_mean * concentration, 1.01)
    beta0 = max((1.0 - r_mean) * concentration, 1.01)

    def residual(log_ab: list[float]) -> list[float]:
        a, b = math.exp(log_ab[0]), math.exp(log_ab[1])
        # Beta percentiles via scipy.
        p05 = scipy_beta.ppf(0.05, a, b)
        p95 = scipy_beta.ppf(0.95, a, b)
        return [p05 - r_lo, p95 - r_hi]

    sol = least_squares(
        residual,
        x0=[math.log(alpha0), math.log(beta0)],
        method="lm",
        xtol=1e-10,
        ftol=1e-10,
    )
    alpha, beta = math.exp(sol.x[0]), math.exp(sol.x[1])
    return alpha, beta


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_priors(path: str | Path) -> Priors:
    """Load a YAML priors file and materialize Beta parameters per component.

    Format:

    .. code-block:: yaml

        morgans_per_step: 1.2e-4
        components:
          - component_idx: 7
            gen_mean: 7
            gen_lo: 4
            gen_hi: 12
            source: "..."
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"priors file not found: {p}")

    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"priors file {p} must be a YAML mapping at top level")

    morgans_per_step = raw.get("morgans_per_step")
    if morgans_per_step is None:
        raise ValueError(
            f"priors file {p} missing required top-level 'morgans_per_step'"
        )
    morgans_per_step = float(morgans_per_step)
    if morgans_per_step <= 0:
        raise ValueError(
            f"morgans_per_step must be positive, got {morgans_per_step}"
        )

    comps_raw = raw.get("components", [])
    if not isinstance(comps_raw, list) or not comps_raw:
        raise ValueError(f"priors file {p} must have a non-empty 'components' list")

    comps: list[ComponentTPrior] = []
    for entry in comps_raw:
        if not isinstance(entry, dict):
            raise ValueError(f"priors entry must be a mapping, got {type(entry)}")
        gen_mean = float(entry["gen_mean"])
        gen_lo = float(entry["gen_lo"])
        gen_hi = float(entry["gen_hi"])
        alpha, beta = prior_to_beta(gen_mean, gen_lo, gen_hi, morgans_per_step)
        comps.append(
            ComponentTPrior(
                component_idx=int(entry["component_idx"]),
                gen_mean=gen_mean,
                gen_lo=gen_lo,
                gen_hi=gen_hi,
                alpha=alpha,
                beta=beta,
                source=str(entry.get("source", "")),
            )
        )

    return Priors(
        morgans_per_step=morgans_per_step,
        components=tuple(comps),
        source_path=str(p.resolve()),
        fingerprint=_hash_file(p),
    )
