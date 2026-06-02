"""Phase 1: registry — SP6/SP5/SP6.sub + parent maps + MID convenience."""

from __future__ import annotations

import pytest

from popout.labelspace import get
from popout.labelspace.registry import (
    SP5,
    SP6,
    LabelSpace,
    make_native_space,
    make_sub_space,
    make_truth_space,
)


def test_sp6_canonical():
    sp6 = get("SP6")
    assert sp6 is SP6
    assert sp6.members == ("afr", "amr", "eas", "eur", "mid", "sas")
    assert sp6.has_mid
    assert sp6.index("eur") == 3
    assert "amr" in sp6
    assert "kor" not in sp6


def test_sp5_canonical_and_mid_siblings():
    sp5 = get("SP5")
    assert sp5.members == ("afr", "amr", "eas", "eur", "sas")
    assert not sp5.has_mid
    assert sp5.with_mid() is SP6
    assert SP6.without_mid() is SP5
    assert SP6.with_mid() is SP6
    assert SP5.without_mid() is SP5


def test_unknown_tag_raises():
    with pytest.raises(KeyError):
        get("not-a-space")


def test_index_raises_on_unknown_label():
    with pytest.raises(KeyError):
        SP6.index("not-a-label")


def test_sub_space_basic():
    sub = make_sub_space(SP6, {"afr": 2, "eur": 3, "amr": 1, "eas": 0,
                                "mid": 0, "sas": 1})
    assert sub.parent is SP6
    assert sub.members == ("afr.1", "afr.2", "amr",
                            "eur.1", "eur.2", "eur.3", "sas")
    assert sub.parent_of["afr.1"] == "afr"
    assert sub.parent_of["amr"] == "amr"
    assert "eas" not in sub


def test_sub_space_singleton_keeps_parent_name():
    sub = make_sub_space(SP6, {"afr": 1, "eur": 1})
    assert sub.members == ("afr", "eur")
    assert sub.parent_of["afr"] == "afr"


def test_truth_space():
    t = make_truth_space(4)
    assert t.tag == "TRUTH"
    assert t.members == ("anc_0", "anc_1", "anc_2", "anc_3")


def test_native_space():
    flare = make_native_space("flare", ("anc_0", "anc_1", "anc_2"))
    assert flare.tag == "flare.native"
    assert flare.members == ("anc_0", "anc_1", "anc_2")


def test_duplicate_members_rejected():
    with pytest.raises(ValueError):
        LabelSpace(tag="bad", members=("a", "a"))


def test_negative_member_count_rejected():
    with pytest.raises(ValueError):
        make_sub_space(SP6, {"afr": -1})


def test_unknown_parent_rejected():
    with pytest.raises(ValueError):
        make_sub_space(SP6, {"kor": 1})
