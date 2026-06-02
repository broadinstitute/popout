"""Phase 3: stable subcomponent naming (1-based dense rank)."""

from __future__ import annotations

import numpy as np

from popout.labelspace.naming import (
    name_components,
    ordered_subcomponent_names,
)


def test_singleton_keeps_bare_label():
    out = name_components({"afr": [0]})
    assert out == {0: "afr"}


def test_split_gets_dense_suffixes():
    out = name_components({"afr": [3, 1]})
    # input order is honored when correlations aren't supplied
    assert out == {3: "afr.1", 1: "afr.2"}


def test_rank_by_descending_correlation():
    # corr[component][rf_index]: component 5 has corr 0.9 with afr, component 0 has corr 0.5
    corr = [
        [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],   # idx 0: afr=0.5
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.9, 0.0, 0.0, 0.0, 0.0, 0.0],   # idx 5: afr=0.9 (rank 1)
    ]
    out = name_components(
        {"afr": [0, 5]},
        correlations=corr,
        target_members=("afr", "amr", "eas", "eur", "mid", "sas"),
    )
    assert out == {5: "afr.1", 0: "afr.2"}


def test_ordered_subcomponent_names_returns_by_index():
    p2rf = {0: "afr", 1: "amr", 2: "eas", 3: "eur",
            4: "mid", 5: "afr", 6: "sas"}
    names = ordered_subcomponent_names(p2rf)
    assert len(names) == 7
    assert names[1] == "amr"
    assert names[4] == "mid"
    # afr split: positions 0 and 5 in the returned list, in order of input
    assert names[0] == "afr.1"
    assert names[5] == "afr.2"


def test_singletons_pass_through_unchanged():
    p2rf = {0: "afr", 1: "amr", 2: "eas"}
    assert ordered_subcomponent_names(p2rf) == ["afr", "amr", "eas"]


def test_reseed_stability_under_correlation_ranking():
    """afr.1 should always be the highest-corr afr component, regardless of
    which raw index that is."""
    # Two label_to_components inputs that disagree on order but agree on corr
    p2rf = {0: "afr", 5: "afr"}
    corr_a = [
        [0.95, 0, 0, 0, 0, 0],     # idx 0: afr=0.95
        None, None, None, None,
        [0.20, 0, 0, 0, 0, 0],     # idx 5: afr=0.20
    ]
    corr_a = [c if c is not None else [0.0] * 6 for c in corr_a]
    out_a = name_components(
        {"afr": [5, 0]},     # input order [5, 0]
        correlations=corr_a,
        target_members=("afr", "amr", "eas", "eur", "mid", "sas"),
    )
    out_b = name_components(
        {"afr": [0, 5]},     # input order [0, 5]
        correlations=corr_a,
        target_members=("afr", "amr", "eas", "eur", "mid", "sas"),
    )
    # Both should assign afr.1 to idx 0 (the higher-corr component).
    assert out_a == out_b
    assert out_a == {0: "afr.1", 5: "afr.2"}
