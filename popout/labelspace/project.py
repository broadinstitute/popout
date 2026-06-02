"""One projector for proportions AND tract codes.

Replaces ``validation/popout_dx/scripts/dx_loaders.project_to_rf_basis``
and ``validation/popout_dx/scripts/dx_local_align_metrics.remap_to_rf_codes``
with one implementation that consumes an ``Assignment``. Adds an
explicit ``collapse`` for SP6 → SP5 (and similar coarsenings).
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

from .assignment import Assignment
from .registry import LabelSpace, get


# ── Proportions ──────────────────────────────────────────────────────────


def project_proportions(q: np.ndarray, assignment: Assignment) -> np.ndarray:
    """Sum ``q`` (n × K_source) onto the assignment's target space.

    The result has shape ``(n, |target|)`` in target-member order.
    Components without an assignment (``unassigned`` sentinel) are
    dropped; their mass disappears from the projection but the total
    over surviving labels equals 1.0 − unassigned_mass per row.
    """
    if q.ndim != 2:
        raise ValueError(f"project_proportions: q must be 2-D, got shape {q.shape}")
    target = assignment.target_space
    out = np.zeros((q.shape[0], len(target.members)), dtype=np.float64)
    for label, idxs in assignment.label_to_components.items():
        if label == "unassigned" or not idxs:
            continue
        if label not in target.members:
            raise ValueError(
                f"project_proportions: label {label!r} not in target {target.tag}"
            )
        j = target.index(label)
        out[:, j] = q[:, list(idxs)].sum(axis=1)
    return out


# ── Tract codes ──────────────────────────────────────────────────────────


def project_tracts(
    tractset,                       # popout.benchmark.common.TractSet
    assignment: Assignment,
    *,
    missing_label: int = 65535,
):
    """Remap a ``TractSet``'s integer codes into the assignment's target space.

    Returns a new TractSet whose ``calls`` array is in ``target``-ordinal
    coordinates (e.g. 0=afr, 1=amr, … 5=sas for ``SP6``). The
    ``label_map`` is rebuilt from the target members.

    Multiple source codes may map to the same target label — the LUT
    handles that vectorised. Source codes not present in the assignment
    are an error (silent drops would mask schema drift). ``missing_label``
    is preserved through the remap.
    """
    from popout.benchmark.common import TractSet  # lazy: tests for proportions
                                                  # path don't need this import.

    target = assignment.target_space
    target_to_int = {name: i for i, name in enumerate(target.members)}
    src_to_dst: dict[int, int] = {}
    for src, lab in assignment.component_to_label.items():
        if lab == "unassigned":
            src_to_dst[int(src)] = missing_label
            continue
        if lab not in target_to_int:
            raise ValueError(
                f"project_tracts: label {lab!r} not in target {target.tag}"
            )
        src_to_dst[int(src)] = target_to_int[lab]

    src_codes_in_calls = set(np.unique(tractset.calls).tolist()) - {missing_label}
    unknown = src_codes_in_calls - set(src_to_dst)
    if unknown:
        raise ValueError(
            f"project_tracts: TractSet contains label code(s) {sorted(unknown)} "
            f"not present in the assignment's component_to_label"
        )

    max_src = max(src_to_dst) if src_to_dst else 0
    lut_size = max(max_src + 1, missing_label + 1)
    lut = np.full(lut_size, missing_label, dtype=np.uint16)
    for src, dst in src_to_dst.items():
        lut[src] = dst
    lut[missing_label] = missing_label

    new_calls = lut[tractset.calls]
    new_label_map = {i: name for name, i in target_to_int.items()}

    return TractSet(
        tool_name=tractset.tool_name,
        chrom=tractset.chrom,
        hap_ids=tractset.hap_ids.copy(),
        site_positions=tractset.site_positions.copy(),
        calls=new_calls,
        label_map=new_label_map,
        metadata=dict(tractset.metadata),
    )


# ── SP6 ↔ SP5 collapse ──────────────────────────────────────────────────


def collapse(
    q: np.ndarray,
    src: LabelSpace,
    dst: LabelSpace,
    *,
    rule: str = "drop",
) -> np.ndarray:
    """Coarsen ``q`` from ``src`` to ``dst`` (e.g. SP6 → SP5).

    Today ``dst`` must be ``src.without_mid()`` and the only supported
    rule is ``"drop"`` (MID mass disappears). ``"redistribute"`` is
    reserved for future use.
    """
    if q.shape[1] != len(src.members):
        raise ValueError(
            f"collapse: q has {q.shape[1]} columns but {src.tag} has {len(src.members)} members"
        )
    if rule != "drop":
        raise ValueError(f"collapse: unsupported rule {rule!r}; only 'drop' is implemented")
    if dst is not src.without_mid():
        raise ValueError(
            f"collapse: dst {dst.tag} is not the without_mid sibling of {src.tag}"
        )
    keep = [i for i, m in enumerate(src.members) if m in dst.members]
    return q[:, keep].copy()
