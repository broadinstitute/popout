"""Soft assignment of priors to ancestry components.

At each EM iteration, each prior's identity signatures are scored
against every component, then softmaxed across components with an
annealed temperature τ. Returns a ``(P, K)`` weight matrix; each row
sums to 1.

τ comes from the prior bundle's annealing schedule:
hot early (priors gently bias many components), cool late (priors
target their best match). The downstream MAP estimator
(:func:`popout.em.update_generations_with_priors`) folds these weights
into per-component Beta(α, β) pseudocounts.

Cross-prior normalization is intentionally NOT done. Each prior
softmaxes independently across components — two priors picking the
same component is a *diagnostic event* (visible in the assignment
dump), not a constraint violation. Coupling unrelated priors via a
joint normalization would be unjustified and harmful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .identity import ComponentState, compose_scores

if TYPE_CHECKING:
    from .prior_spec import Priors


def assign_priors_to_components(
    priors: "Priors",
    component_states: list[ComponentState],
    iteration: int,
) -> np.ndarray:
    """Compute the soft prior→component weight matrix.

    Parameters
    ----------
    priors
        Container of priors with annealing schedule.
    component_states
        K component snapshots for the current EM iteration.
    iteration
        EM iteration index (0-based). The annealing schedule maps this
        to the temperature τ.

    Returns
    -------
    (P, K) ndarray of soft weights; each row sums to 1.
    """
    K = len(component_states)
    P = len(priors.priors)
    if K == 0 or P == 0:
        return np.zeros((P, K), dtype=np.float64)

    tau = float(priors.annealing.tau_at(iteration))
    scores = np.zeros((P, K), dtype=np.float64)
    for p, prior in enumerate(priors.priors):
        scores[p] = compose_scores(prior.identity_signatures, component_states)

    return _softmax(scores / tau, axis=1)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Stable softmax along ``axis`` — subtracts max before exp."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)
