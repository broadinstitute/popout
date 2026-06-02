"""Stable subcomponent naming via correlation rank.

Replaces three legacy producers
(``validation/scripts/compare_to_rf.py:291``, ``popout/viz/_style.py:52``,
``popout/label.py`` ``merge_map``) with one rule:

    Within a continental label that received more than one component, sort
    those components by *descending correlation with the reference label*
    and suffix ``.1, .2, …`` (1-based, dense). Singletons stay unsuffixed.

The 1-based dense rule makes ``afr.1`` mean "the component most strongly
correlated with AFR," reproducibly across seeds — closing audit C6.
"""

from __future__ import annotations

from typing import Mapping, Sequence


def name_components(
    label_to_components: Mapping[str, Sequence[int]],
    *,
    correlations: Sequence[Sequence[float]] | None = None,
    target_members: Sequence[str] | None = None,
) -> dict[int, str]:
    """Return ``{component_index: subcomponent_name}``.

    ``label_to_components`` may be pre-sorted by descending correlation
    (as the matching strategies in this module produce). If
    ``correlations`` and ``target_members`` are supplied this function
    re-sorts to ensure rank-order; otherwise the input ordering is
    trusted.
    """
    out: dict[int, str] = {}
    for label, indices in label_to_components.items():
        idxs = list(indices)
        if correlations is not None and target_members is not None and label in target_members:
            ref_col = list(target_members).index(label)
            idxs.sort(key=lambda i: -float(correlations[i][ref_col]))
        if len(idxs) == 1:
            out[idxs[0]] = label
        else:
            for rank, idx in enumerate(idxs, start=1):
                out[idx] = f"{label}.{rank}"
    return out


def ordered_subcomponent_names(
    component_to_label: Mapping[int, str],
    *,
    correlations: Sequence[Sequence[float]] | None = None,
    target_members: Sequence[str] | None = None,
) -> list[str]:
    """Convenience: return names in ascending component-index order.

    Replaces ``popout/viz/_style.py::ancestry_names`` and
    ``compare_to_rf.py::popout_names``.
    """
    # Build label_to_components in canonical (descending-corr) order if
    # we can, then derive the per-index name via name_components.
    label_to_components: dict[str, list[int]] = {}
    for idx, label in sorted(component_to_label.items()):
        label_to_components.setdefault(label, []).append(idx)
    if correlations is not None and target_members is not None:
        target_list = list(target_members)
        for label, idxs in label_to_components.items():
            if label in target_list:
                ref_col = target_list.index(label)
                idxs.sort(key=lambda i: -float(correlations[i][ref_col]))
    names = name_components(label_to_components)
    return [names[i] for i in sorted(component_to_label)]
