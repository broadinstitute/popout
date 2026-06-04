"""Phase 1: per-tool Estimate loaders."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from popout.estimates import (
    read_flare_aggregated,
    read_flare_panel_names,
    read_popout_global,
    read_rf_table,
    read_rye_q,
)
from popout.labelspace import by_name, get
from popout.labelspace.registry import SP5, SP6


# ── FLARE panel header parser ────────────────────────────────────────────


def _write_flare_vcf_header(path: Path, ancestry_line: str) -> None:
    body = (
        "##fileformat=VCFv4.2\n"
        f"{ancestry_line}\n"
        "##INFO=<ID=AC,Number=A,Type=Integer,Description=\"AC\">\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )
    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(body)
    else:
        path.write_text(body)


@pytest.mark.parametrize("ancestry_line, expected", [
    ("##ANCESTRY=<afr,eas,amr,eur,sas>",  ["afr", "eas", "amr", "eur", "sas"]),
    ("##ANCESTRY=afr eas amr eur sas",    ["afr", "eas", "amr", "eur", "sas"]),
    ("##ANCESTRY=AFR EAS AMR EUR SAS",    ["afr", "eas", "amr", "eur", "sas"]),
    ("##ANCESTRY=<AFR, EAS, AMR, EUR, SAS>", ["afr", "eas", "amr", "eur", "sas"]),
])
def test_read_flare_panel_names_variants(tmp_path: Path, ancestry_line, expected):
    vcf = tmp_path / "x.vcf"
    _write_flare_vcf_header(vcf, ancestry_line)
    assert read_flare_panel_names(vcf) == expected


def test_read_flare_panel_names_gzipped(tmp_path: Path):
    vcf = tmp_path / "x.vcf.gz"
    _write_flare_vcf_header(vcf, "##ANCESTRY=<afr,amr,eas,eur,sas>")
    assert read_flare_panel_names(vcf) == ["afr", "amr", "eas", "eur", "sas"]


def test_read_flare_panel_names_missing(tmp_path: Path):
    vcf = tmp_path / "x.vcf"
    vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\n")
    with pytest.raises(ValueError, match="##ANCESTRY="):
        read_flare_panel_names(vcf)


# ── FLARE aggregated TSV (both shapes) ───────────────────────────────────


def _write_global_tsv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(c) for c in r) + "\n")


def test_read_flare_aggregated_named(tmp_path: Path):
    tsv = tmp_path / "named.global.tsv"
    _write_global_tsv(
        tsv,
        ["sample_id", "afr", "amr", "eas", "eur", "sas"],
        [["s0", 0.1, 0.2, 0.0, 0.6, 0.1],
         ["s1", 0.0, 0.0, 0.5, 0.4, 0.1]],
    )
    e = read_flare_aggregated(tsv, scope=("cluster_000", "chr1"))
    assert e.tool == "flare"
    assert e.members == ("afr", "amr", "eas", "eur", "sas")
    np.testing.assert_allclose(e.column("eas"), [0.0, 0.5])


def test_read_flare_aggregated_anonymous_rename(tmp_path: Path):
    tsv = tmp_path / "anon.global.tsv"
    _write_global_tsv(
        tsv,
        ["sample_id", "ancestry_0", "ancestry_1", "ancestry_2",
         "ancestry_3", "ancestry_4"],
        [["s0", 0.1, 0.2, 0.0, 0.6, 0.1]],
    )
    e = read_flare_aggregated(
        tsv, scope=("cluster_000", "chr1"),
        panel_names=["afr", "amr", "eas", "eur", "sas"],
    )
    assert e.members == ("afr", "amr", "eas", "eur", "sas")
    np.testing.assert_allclose(e.column("eur"), [0.6])


def test_read_flare_aggregated_anonymous_requires_panel_names(tmp_path: Path):
    tsv = tmp_path / "anon.global.tsv"
    _write_global_tsv(
        tsv,
        ["sample_id", "ancestry_0", "ancestry_1"],
        [["s0", 0.5, 0.5]],
    )
    with pytest.raises(ValueError, match="require panel_names"):
        read_flare_aggregated(tsv, scope=("x",))


# ── Rye loader ──────────────────────────────────────────────────────────


def test_read_rye_q_trusts_header(tmp_path: Path):
    rye = tmp_path / "rye.Q"
    _write_global_tsv(
        rye,
        ["eur", "eas", "amr", "afr", "sas", "research_id"],
        [[0.6, 0.0, 0.1, 0.2, 0.1, "s0"],
         [0.1, 0.0, 0.0, 0.9, 0.0, "s1"]],
    )
    e = read_rye_q(rye, scope=("cohort",))
    assert e.tool == "rye"
    assert e.members == ("eur", "eas", "amr", "afr", "sas")
    np.testing.assert_allclose(e.column("afr"), [0.2, 0.9])
    assert e.sample_ids == ("s0", "s1")


def test_read_rye_q_roster_aligns(tmp_path: Path):
    rye = tmp_path / "rye.Q"
    _write_global_tsv(
        rye,
        ["eur", "afr", "research_id"],
        [[0.5, 0.5, "s0"], [0.9, 0.1, "s1"], [0.0, 1.0, "s2"]],
    )
    e = read_rye_q(rye, scope=("cohort",), sample_ids=["s2", "s0"])
    assert e.sample_ids == ("s2", "s0")
    np.testing.assert_allclose(e.column("eur"), [0.0, 0.5])


# ── RF loader ───────────────────────────────────────────────────────────


def test_read_rf_table_emits_sp6_with_hard_calls(tmp_path: Path):
    rf = tmp_path / "rf.tsv"
    rf.write_text(
        "research_id\tancestry_pred\tprobabilities\n"
        "s0\tafr\t[0.9, 0.05, 0.0, 0.05, 0.0, 0.0]\n"
        "s1\teur\t[0.0, 0.0, 0.0, 0.95, 0.05, 0.0]\n"
    )
    e = read_rf_table(rf, scope=("cohort",))
    assert e.tool == "rf"
    assert e.label_space is SP6
    assert e.hard_calls is not None
    assert tuple(e.hard_calls) == ("afr", "eur")
    np.testing.assert_allclose(e.column("eur"), [0.05, 0.95])


def test_read_rf_table_rejects_wrong_length(tmp_path: Path):
    rf = tmp_path / "rf.tsv"
    rf.write_text(
        "research_id\tancestry_pred\tprobabilities\n"
        "s0\tafr\t[0.5, 0.5]\n"
    )
    with pytest.raises(ValueError, match=r"length 2"):
        read_rf_table(rf, scope=("cohort",))


# ── popout loader (requires Assignment) ─────────────────────────────────


def test_read_popout_global_requires_assignment(tmp_path: Path):
    g = tmp_path / "popout.global.tsv"
    _write_global_tsv(
        g,
        ["sample_id", "ancestry_0", "ancestry_1", "ancestry_2"],
        [["s0", 0.5, 0.3, 0.2]],
    )
    with pytest.raises(ValueError, match="Assignment is required"):
        read_popout_global(g, scope=("cohort",))


def test_read_popout_global_applies_assignment(tmp_path: Path):
    g = tmp_path / "popout.global.tsv"
    # Three popout components map into SP5: 0→afr, 1→eur, 2→eur (a fold).
    _write_global_tsv(
        g,
        ["sample_id", "ancestry_0", "ancestry_1", "ancestry_2"],
        [["s0", 0.6, 0.2, 0.2],
         ["s1", 0.0, 0.5, 0.5]],
    )
    # Build a manual Assignment that lands in SP5.
    from popout.labelspace import Assignment
    a = Assignment(
        target_space=SP5,
        source={"tool": "popout"},
        method="manual", input_space="posterior",
        component_to_label={0: "afr", 1: "eur", 2: "eur"},
        label_to_components={"afr": [0], "eur": [1, 2]},
        subcomponent_names={1: "eur.1", 2: "eur.2"},
        diagnostics={}, provenance={},
    )
    e = read_popout_global(g, scope=("cohort",), assignment=a)
    assert e.tool == "popout"
    assert e.members == SP5.members
    np.testing.assert_allclose(e.column("afr"), [0.6, 0.0])
    np.testing.assert_allclose(e.column("eur"), [0.4, 1.0])
