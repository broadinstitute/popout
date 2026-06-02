"""Phase 1: shorthand tag grammar."""

from __future__ import annotations

import pytest

from popout.labelspace import Assignment, get
from popout.labelspace.shorthand import format, parse, version_hash


SP6 = get("SP6")
SP5 = get("SP5")


def _a(tool: str, method: str = "corrH", component_to_label=None) -> Assignment:
    return Assignment(
        target_space=SP6, source={"tool": tool},
        method=method, input_space="allele_freq",
        component_to_label=component_to_label or {0: "afr", 1: "eur"},
        label_to_components={"afr": [0], "eur": [1]},
        subcomponent_names={}, diagnostics={}, provenance={},
    )


def test_format_basic():
    tag = format(SP6, [_a("popout"), _a("flare", "name")])
    assert tag.startswith("L=SP6/MID+ | ")
    assert "flare=>name" in tag
    assert "popout=>corrH" in tag
    assert " | v=" in tag


def test_format_sp5_marks_mid_minus():
    a = Assignment(
        target_space=SP5, source={"tool": "rye"},
        method="name", input_space="posterior",
        component_to_label={0: "afr", 1: "eur"},
        label_to_components={"afr": [0], "eur": [1]},
        subcomponent_names={}, diagnostics={}, provenance={},
    )
    tag = format(SP5, [a])
    assert tag.startswith("L=SP5/MID- | ")


def test_format_stable_under_input_ordering():
    a1 = _a("popout")
    a2 = _a("flare", "name")
    assert format(SP6, [a1, a2]) == format(SP6, [a2, a1])


def test_parse_roundtrip():
    tag = format(SP6, [_a("popout"), _a("flare", "name")])
    parsed = parse(tag)
    assert parsed["target"] == "SP6"
    assert parsed["mid"] == "MID+"
    tools = {c["tool"]: c["method"] for c in parsed["clauses"]}
    assert tools == {"popout": "corrH", "flare": "name"}
    assert "version" in parsed


def test_hash_changes_when_map_changes():
    base = _a("popout")
    alt = _a("popout", component_to_label={0: "amr", 1: "eur"})
    assert version_hash(base) != version_hash(alt)


def test_hash_stable_across_calls():
    a = _a("popout")
    assert version_hash(a) == version_hash(a)


def test_hash_independent_of_diagnostics():
    base = _a("popout")
    flip = Assignment(**{**base.__dict__,
                         "diagnostics": {"correlations": [[1.0]]}})
    assert version_hash(base) == version_hash(flip)


def test_parse_rejects_garbage():
    with pytest.raises(ValueError):
        parse("not a tag")
    with pytest.raises(ValueError):
        parse("L=SP6/MID+ | malformed clause | v=abc")
