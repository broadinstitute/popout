"""Shared test fixtures and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from popout.identity import ComponentState
from popout.prior_spec import (
    LinearAnnealingSchedule,
    Prior,
    Priors,
    prior_to_beta,
)


@dataclass(frozen=True)
class UniformScoreSig:
    """Test-only identity signature.

    Returns the same score for every component, which makes
    :func:`popout.identity.compose_scores` skip it (zero std → degenerate).
    The downstream softmax in
    :func:`popout.identity_assignment.assign_priors_to_components`
    therefore produces a uniform 1/K weight per row. This isolates
    M-step plumbing tests from identity-scoring details, which are
    covered by ``test_identity.py`` and ``test_identity_synthetic.py``.
    """

    weight: float = 1.0

    def score(self, cs: ComponentState) -> float:
        return 0.0


def make_priors_uniform(
    gen_specs: Iterable[tuple[float, float, float]],
    *,
    morgans_per_step: float = 1.2e-4,
    names: list[str] | None = None,
) -> Priors:
    """Build a :class:`Priors` directly (no YAML round-trip) for tests.

    Each prior is given a :class:`UniformScoreSig` so the soft assignment
    is uniform across components — every component gets every prior's
    pseudocount, weighted equally by 1/K.
    """
    specs = list(gen_specs)
    if names is None:
        names = [f"P{i}" for i in range(len(specs))]
    plist = []
    for nm, (mean, lo, hi) in zip(names, specs):
        a, b = prior_to_beta(mean, lo, hi, morgans_per_step)
        plist.append(
            Prior(
                name=nm,
                identity_signatures=(UniformScoreSig(),),
                gen_mean=mean, gen_lo=lo, gen_hi=hi,
                alpha=a, beta=b,
            )
        )
    return Priors(
        priors=tuple(plist),
        morgans_per_step=morgans_per_step,
        annealing=LinearAnnealingSchedule(1.0, 0.1, 10),
        fingerprint="x" * 64,
        source_path="<test>",
    )
