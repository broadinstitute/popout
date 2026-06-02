"""Site alignment, haplotype alignment, and label matching."""

from __future__ import annotations

import warnings
from copy import deepcopy

import numpy as np

from popout.benchmark.common import MISSING_LABEL, TractSet


def align_sites(
    tracts_a: TractSet,
    tracts_b: TractSet,
    strategy: str = "intersect",
) -> tuple[TractSet, TractSet]:
    """Align two TractSets to a common set of site positions.

    Parameters
    ----------
    strategy : "intersect" (common positions) or "project_a_onto_b"
        (evaluate A's tracts at B's positions).
    """
    if strategy == "intersect":
        common = np.intersect1d(tracts_a.site_positions, tracts_b.site_positions)
        if len(common) == 0:
            raise ValueError("No common site positions between the two TractSets")
        a_out = _subset_sites(tracts_a, common)
        b_out = _subset_sites(tracts_b, common)
        return a_out, b_out
    elif strategy == "project_a_onto_b":
        # Evaluate A's tracts at B's site positions
        a_out = _project_onto_sites(tracts_a, tracts_b.site_positions)
        b_out = deepcopy(tracts_b)
        return a_out, b_out
    else:
        raise ValueError(f"Unknown site strategy: {strategy!r}")


def _subset_sites(ts: TractSet, positions: np.ndarray) -> TractSet:
    """Subset a TractSet to the given positions."""
    idx = np.searchsorted(ts.site_positions, positions)
    # Verify positions actually match
    valid = idx < len(ts.site_positions)
    valid[valid] &= ts.site_positions[idx[valid]] == positions[valid]
    if not valid.all():
        raise ValueError("Some requested positions not found in TractSet")
    return TractSet(
        tool_name=ts.tool_name,
        chrom=ts.chrom,
        hap_ids=ts.hap_ids.copy(),
        site_positions=positions.copy(),
        calls=ts.calls[:, idx].copy(),
        label_map=dict(ts.label_map),
        metadata=dict(ts.metadata),
    )


def _project_onto_sites(ts: TractSet, target_positions: np.ndarray) -> TractSet:
    """Project a TractSet's tracts onto target site positions.

    For each target position, find which tract covers it and assign that label.
    """
    tracts = ts.to_tracts()
    n_haps = ts.n_haps
    n_target = len(target_positions)
    calls = np.full((n_haps, n_target), MISSING_LABEL, dtype=np.uint16)

    for hap_idx, start_idx, end_idx, label in tracts:
        start_bp = ts.site_positions[start_idx]
        end_bp = ts.site_positions[end_idx - 1]
        mask = (target_positions >= start_bp) & (target_positions <= end_bp)
        calls[hap_idx, mask] = label

    return TractSet(
        tool_name=ts.tool_name,
        chrom=ts.chrom,
        hap_ids=ts.hap_ids.copy(),
        site_positions=target_positions.copy(),
        calls=calls,
        label_map=dict(ts.label_map),
        metadata=dict(ts.metadata),
    )


def align_haps(
    tracts_a: TractSet,
    tracts_b: TractSet,
) -> tuple[TractSet, TractSet]:
    """Subset both TractSets to their common haplotype IDs, in matched order."""
    set_a = set(tracts_a.hap_ids.tolist())
    set_b = set(tracts_b.hap_ids.tolist())
    common = sorted(set_a & set_b)

    if not common:
        raise ValueError("No common haplotype IDs between the two TractSets")

    dropped_a = set_a - set(common)
    dropped_b = set_b - set(common)
    if dropped_a:
        warnings.warn(f"Dropped {len(dropped_a)} haplotypes from {tracts_a.tool_name}")
    if dropped_b:
        warnings.warn(f"Dropped {len(dropped_b)} haplotypes from {tracts_b.tool_name}")

    a_out = _subset_haps(tracts_a, common)
    b_out = _subset_haps(tracts_b, common)
    return a_out, b_out


def _subset_haps(ts: TractSet, hap_ids: list[str]) -> TractSet:
    """Subset a TractSet to the given haplotype IDs in order."""
    id_to_idx = {h: i for i, h in enumerate(ts.hap_ids.tolist())}
    indices = [id_to_idx[h] for h in hap_ids]
    return TractSet(
        tool_name=ts.tool_name,
        chrom=ts.chrom,
        hap_ids=np.array(hap_ids, dtype=object),
        site_positions=ts.site_positions.copy(),
        calls=ts.calls[indices].copy(),
        label_map=dict(ts.label_map),
        metadata=dict(ts.metadata),
    )


def match_labels(
    tracts_src: TractSet,
    tracts_ref: TractSet,
    method: str = "hungarian",
) -> dict[int, int]:
    """Find best label mapping from src to ref.

    Phase 6 of the label-space retrofit: routed through
    :func:`popout.labelspace.confusion_hungarian`. The returned dict
    shape (``{src_label: ref_label}`` with integer ref-label codes) is
    preserved for back-compat.
    """
    if method != "hungarian":
        raise ValueError(f"Unknown method: {method!r}")

    src_labels = sorted(k for k in tracts_src.label_map if k != MISSING_LABEL)
    ref_labels = sorted(k for k in tracts_ref.label_map if k != MISSING_LABEL)

    # Build a transient SP6-ish target whose members are the *names* of
    # the ref label_map values (in ref-label order). confusion_hungarian
    # returns string-valued component_to_label; we invert via the
    # ref_label_map to get back integer ref codes.
    from popout.labelspace.matching import confusion_hungarian
    from popout.labelspace.registry import make_native_space

    ref_name_for_int = {int(k): str(v) for k, v in tracts_ref.label_map.items()
                        if k != MISSING_LABEL}
    ref_names_ordered = [ref_name_for_int[r] for r in ref_labels]
    target = make_native_space("ref", tuple(ref_names_ordered))
    a = confusion_hungarian(
        tracts_src.calls, tracts_ref.calls, target,
        src_labels=src_labels, ref_labels=ref_labels,
        missing_label=MISSING_LABEL,
    )

    name_to_ref_int = {name: int(r) for r, name in
                       zip(ref_labels, ref_names_ordered)}
    mapping: dict[int, int] = {}
    for src_int, label in a.component_to_label.items():
        if label == "unassigned" or label not in name_to_ref_int:
            continue
        mapping[int(src_int)] = name_to_ref_int[label]
    return mapping


def apply_label_map(tracts: TractSet, mapping: dict[int, int]) -> TractSet:
    """Return a new TractSet with calls remapped via mapping."""
    # Build lookup table
    max_label = max(max(mapping.keys()), max(mapping.values(), default=0))
    lut = np.arange(max_label + 2, dtype=np.uint16)  # identity by default
    for src, dst in mapping.items():
        lut[src] = dst

    # Apply: for labels within lut range, remap; others pass through
    new_calls = tracts.calls.copy()
    mask = new_calls <= max_label
    new_calls[mask] = lut[new_calls[mask]]

    # Build new label_map with ref names
    new_label_map = {}
    for src, dst in mapping.items():
        # Use the ref's name for the destination label
        new_label_map[dst] = tracts.label_map.get(src, str(dst))

    return TractSet(
        tool_name=tracts.tool_name,
        chrom=tracts.chrom,
        hap_ids=tracts.hap_ids.copy(),
        site_positions=tracts.site_positions.copy(),
        calls=new_calls,
        label_map=new_label_map,
        metadata=dict(tracts.metadata),
    )
