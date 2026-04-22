"""Tests for ancestry name parsing."""

import tempfile
from pathlib import Path

from popout.names import parse_ancestry_names


def test_default_names():
    names = parse_ancestry_names(None, 4)
    assert names == ["anc_0", "anc_1", "anc_2", "anc_3"]


def test_comma_separated():
    names = parse_ancestry_names("afr,eas,eur", 3)
    assert names == ["afr", "eas", "eur"]


def test_comma_separated_with_spaces():
    names = parse_ancestry_names("afr, eas, eur", 3)
    assert names == ["afr", "eas", "eur"]


def test_file_input():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        f.write("afr\neas\neur\n")
        f.flush()
        names = parse_ancestry_names(f.name, 3)
    assert names == ["afr", "eas", "eur"]


def test_length_mismatch():
    try:
        parse_ancestry_names("afr,eas", 3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "2 entries" in str(e)
        assert "need 3" in str(e)
