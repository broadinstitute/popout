"""Tests for the --priors-dump-assignments output (Step 6).

The dump is the load-bearing audit artifact: a (P, K) TSV where rows
are prior names, columns are component indices, plus a comment line
annotating each component with its nearest 1KG superpop. This is the
file you read first to spot prior-to-component mismatches like the
priors_v1 failure that motivated the redesign.
"""

from __future__ import annotations

import numpy as np
import pytest

from popout.em import (
    run_em,
    write_priors_assignment_dump,
)
from popout.simulate import simulate_admixed
from tests.conftest import make_priors_uniform


# --------------------------------------------------------------------------
# Direct dump-writer tests
# --------------------------------------------------------------------------


def _fake_model(K: int):
    """Tiny stub matching what write_priors_assignment_dump needs."""
    class _M:
        n_ancestries = K
        allele_freq = np.tile(np.linspace(0.1, 0.9, 4)[None, :], (K, 1))
    return _M()


def _fake_chrom():
    class _C:
        chrom = "1"
        pos_bp = np.array([100, 200, 300, 400], dtype=np.int64)
    return _C()


def test_dump_writes_expected_layout(tmp_path):
    """Header row, prior rows, correct dimensions, parseable as TSV."""
    P, K = 3, 4
    rng = np.random.default_rng(0)
    raw = rng.uniform(size=(P, K))
    assignment = raw / raw.sum(axis=1, keepdims=True)

    priors = make_priors_uniform(
        [(2, 1, 4), (10, 5, 20), (50, 30, 80)],
        names=["A", "B", "C"],
    )
    out = tmp_path / "dump.tsv"
    write_priors_assignment_dump(
        out, assignment, priors, _fake_model(K), _fake_chrom(),
    )

    text = out.read_text()
    lines = text.rstrip("\n").split("\n")
    # Comment + column header + P rows.
    assert len(lines) == 2 + P
    assert lines[0].startswith("# nearest_1KG\t")
    assert lines[1].startswith("prior\t")
    cols = lines[1].split("\t")
    assert cols == ["prior"] + [f"comp_{k}" for k in range(K)]
    # Round-trip the weights.
    for p, prior_name in enumerate(["A", "B", "C"]):
        cells = lines[2 + p].split("\t")
        assert cells[0] == prior_name
        weights = np.array([float(x) for x in cells[1:]])
        # Dump format writes 6 decimal places; tolerate that precision.
        np.testing.assert_allclose(weights, assignment[p], atol=1e-6)


def test_dump_handles_missing_1kg_cache(tmp_path, monkeypatch):
    """When the 1KG cache isn't populated, the annotation row falls
    back to '-' and the dump still writes."""
    # Force the 1KG resolver to fail.
    from popout import fetch_superpop_freqs

    def _missing(*args, **kwargs):
        raise FileNotFoundError("test stub: cache missing")

    monkeypatch.setattr(
        fetch_superpop_freqs, "resolve_superpop_freqs_path", _missing,
    )

    P, K = 2, 3
    assignment = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    priors = make_priors_uniform([(2, 1, 4), (10, 5, 20)], names=["X", "Y"])
    out = tmp_path / "dump.tsv"
    write_priors_assignment_dump(
        out, assignment, priors, _fake_model(K), _fake_chrom(),
    )
    lines = out.read_text().rstrip("\n").split("\n")
    annot = lines[0].split("\t")[1:]
    assert all(a == "-" for a in annot)


# --------------------------------------------------------------------------
# End-to-end through run_em
# --------------------------------------------------------------------------


def test_run_em_writes_dump_when_path_supplied(tmp_path):
    chrom_data, _, _ = simulate_admixed(
        n_samples=60, n_sites=200, n_ancestries=3,
        gen_since_admix=8, chrom_length_cm=40.0, rng_seed=21,
    )
    priors = make_priors_uniform([(2, 1, 4), (10, 5, 20)])
    out = tmp_path / "assignments.tsv"

    run_em(
        chrom_data, n_ancestries=3, n_em_iter=2, gen_since_admix=8.0,
        rng_seed=0, priors=priors, priors_dump_path=str(out),
    )

    assert out.exists()
    lines = out.read_text().rstrip("\n").split("\n")
    # 1 annotation + 1 header + 2 priors = 4 lines.
    assert len(lines) == 4
    # Each prior row has 3 weights.
    for line in lines[2:]:
        cells = line.split("\t")
        assert len(cells) == 4  # name + 3 weights
        weights = np.array([float(x) for x in cells[1:]])
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-5)


def test_run_em_no_dump_when_path_none(tmp_path):
    """priors_dump_path=None must not write a file."""
    chrom_data, _, _ = simulate_admixed(
        n_samples=60, n_sites=200, n_ancestries=3,
        gen_since_admix=8, chrom_length_cm=40.0, rng_seed=21,
    )
    priors = make_priors_uniform([(2, 1, 4)])
    out = tmp_path / "should_not_exist.tsv"

    run_em(
        chrom_data, n_ancestries=3, n_em_iter=2, gen_since_admix=8.0,
        rng_seed=0, priors=priors, priors_dump_path=None,
    )
    assert not out.exists()
