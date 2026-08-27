"""End-to-end tests for validation/scripts/extract_tract_events.py.

Every test builds a tiny synthetic VCF, invokes the CLI, and reads back the
emitted parquet + JSON. We do not import the extractor module; the CLI is
the contract.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest


REPO = Path(__file__).resolve().parents[2]
EXTRACTOR = REPO / "validation" / "scripts" / "extract_tract_events.py"


def _run(vcf: Path, out_dir: Path, cluster_id: str = "test_cluster",
         expect_ok: bool = True, extra: list[str] | None = None) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, str(EXTRACTOR), str(vcf),
        "--out-dir", str(out_dir),
        "--cluster-id", cluster_id,
        "--skip-input-sha256",
    ]
    if extra:
        cmd.extend(extra)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"extractor failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def _read_parquet(path: Path):
    return pq.read_table(path).to_pylist()


# ── T1: panel verbatim ────────────────────────────────────────────────────

def test_panel_verbatim_uses_bundle_supplied_names(tmp_path, write_vcf):
    """Panel names must be echoed byte-for-byte from ##ANCESTRY, not
    reconstructed from a hard-coded list. Uses non-standard names to prove
    it."""
    vcf = write_vcf(
        tmp_path / "panel.vcf",
        panel_line="##ANCESTRY=<blue=0,green=1,red=2>",
        contigs=[("chr22", 50_000_000)],
        samples=["s1", "s2"],
        records=[
            ("chr22", 1000, [("0", "1"), ("2", "0")]),
            ("chr22", 2000, [("0", "1"), ("2", "0")]),
        ],
    )
    out = tmp_path / "out"
    _run(vcf, out)
    panel = json.loads((out / "panel.json").read_text())
    assert panel["panel_names"] == ["blue", "green", "red"]
    assert panel["K"] == 3
    assert panel["panel_source_raw"] == "##ANCESTRY=<blue=0,green=1,red=2>"
    assert panel["chrom_lengths"] == {"chr22": 50_000_000}
    assert panel["reference_build"] == "GRCh38"


def test_panel_indices_out_of_order(tmp_path, write_vcf):
    """Panel indices assigned out of textual order should still map to the
    correct integer slots."""
    vcf = write_vcf(
        tmp_path / "panel.vcf",
        panel_line="##ANCESTRY=<amr=2,eas=0,eur=1>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1"],
        records=[("chr1", 100, [("0", "1")])],
    )
    out = tmp_path / "out"
    _run(vcf, out)
    panel = json.loads((out / "panel.json").read_text())
    assert panel["panel_names"] == ["eas", "eur", "amr"]


# ── T2: state machine golden ──────────────────────────────────────────────

def test_state_machine_golden(tmp_path, write_vcf):
    """Hand-computed 3-sample × 5-site fixture with known tracts.

    Sample layout (columns are sites at pos 100, 200, 300, 400, 500):

      s1 h0: 0 0 1 1 1   → tract A(anc=0, 100..200, 2 sites), B(anc=1, 300..500, 3 sites)
      s1 h1: 2 2 2 2 2   → single tract (anc=2, 100..500, 5 sites)
      s2 h0: 0 1 0 1 0   → four transitions, five tracts of 1 site each
      s2 h1: 1 1 1 1 1   → single tract (anc=1, 100..500, 5 sites)
      s3 h0: 0 0 0 0 0   → single tract (anc=0, 100..500, 5 sites)
      s3 h1: 0 0 0 0 0   → single tract (anc=0, 100..500, 5 sites)
    """
    layout = [
        # (an1, an2) per sample at each site
        [("0", "2"), ("0", "1"), ("0", "0")],  # pos 100
        [("0", "2"), ("1", "1"), ("0", "0")],  # pos 200
        [("1", "2"), ("0", "1"), ("0", "0")],  # pos 300
        [("1", "2"), ("1", "1"), ("0", "0")],  # pos 400
        [("1", "2"), ("0", "1"), ("0", "0")],  # pos 500
    ]
    positions = [100, 200, 300, 400, 500]
    vcf = write_vcf(
        tmp_path / "sm.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1,eur=2>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1", "s2", "s3"],
        records=[("chr1", p, row) for p, row in zip(positions, layout)],
    )
    out = tmp_path / "out"
    _run(vcf, out)

    tracts = _read_parquet(out / "tracts.parquet")
    # 2 (s1h0) + 1 (s1h1) + 5 (s2h0) + 1 (s2h1) + 1 (s3h0) + 1 (s3h1) = 11
    assert len(tracts) == 11

    # sample_idx 0 = s1
    s1_h0 = sorted([t for t in tracts if t["sample_idx"] == 0 and t["hap"] == 0],
                   key=lambda t: t["start_bp"])
    assert len(s1_h0) == 2
    assert s1_h0[0] == {
        "sample_idx": 0, "hap": 0, "chrom": "chr1",
        "start_bp": 100, "end_bp": 200, "n_sites": 2,
        "ancestry_idx": 0, "close_reason": "an_change",
    }
    assert s1_h0[1] == {
        "sample_idx": 0, "hap": 0, "chrom": "chr1",
        "start_bp": 300, "end_bp": 500, "n_sites": 3,
        "ancestry_idx": 1, "close_reason": "shard_end",
    }

    # s1 h1 = single tract, closed at shard_end
    s1_h1 = [t for t in tracts if t["sample_idx"] == 0 and t["hap"] == 1]
    assert s1_h1 == [{
        "sample_idx": 0, "hap": 1, "chrom": "chr1",
        "start_bp": 100, "end_bp": 500, "n_sites": 5,
        "ancestry_idx": 2, "close_reason": "shard_end",
    }]

    # s2 h0 = five singleton tracts alternating 0,1,0,1,0
    s2_h0 = sorted([t for t in tracts if t["sample_idx"] == 1 and t["hap"] == 0],
                   key=lambda t: t["start_bp"])
    assert len(s2_h0) == 5
    ancs = [t["ancestry_idx"] for t in s2_h0]
    assert ancs == [0, 1, 0, 1, 0]
    for t in s2_h0[:-1]:
        assert t["close_reason"] == "an_change"
        assert t["n_sites"] == 1
        assert t["start_bp"] == t["end_bp"]
    assert s2_h0[-1]["close_reason"] == "shard_end"

    # Transitions.
    trans = _read_parquet(out / "transitions.parquet")
    assert len(trans) == 5  # s1h0 (1) + s2h0 (4)
    s2_trans = sorted(
        [t for t in trans if t["sample_idx"] == 1 and t["hap"] == 0],
        key=lambda t: t["position_bp"],
    )
    assert [t["position_bp"] for t in s2_trans] == [200, 300, 400, 500]
    assert [(t["from_ancestry_idx"], t["to_ancestry_idx"]) for t in s2_trans] == \
        [(0, 1), (1, 0), (0, 1), (1, 0)]


# ── T3: no-missing-AN abort ───────────────────────────────────────────────

def test_missing_an_aborts(tmp_path, write_vcf):
    """Missing AN calls violate the FLARE hard-call invariant; the
    extractor must abort with the no_missing_an reason (never silently drop)."""
    vcf = write_vcf(
        tmp_path / "bad.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1"],
        records=[
            ("chr1", 100, [("0", "1")]),
            ("chr1", 200, [(".", "1")]),
        ],
    )
    out = tmp_path / "out"
    proc = _run(vcf, out, expect_ok=False)
    assert proc.returncode != 0
    assert "no_missing_an" in proc.stderr or "missing AN" in proc.stderr


# ── T4: site_positions.parquet monotonic ──────────────────────────────────

def test_site_positions_monotonic(tmp_path, write_vcf):
    """Emitted site_positions.parquet is strictly ascending per chrom, and
    the on-disk row count matches the number of sites in the VCF."""
    vcf = write_vcf(
        tmp_path / "mono.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1"],
        records=[
            ("chr1", 100, [("0", "1")]),
            ("chr1", 200, [("0", "1")]),
            ("chr1", 300, [("0", "1")]),
        ],
    )
    out = tmp_path / "out"
    _run(vcf, out)
    sp = _read_parquet(out / "site_positions.parquet")
    assert [r["position_bp"] for r in sp if r["chrom"] == "chr1"] == [100, 200, 300]

    # Non-monotonic input must be caught even if the state machine would
    # otherwise silently accept it.
    bad = write_vcf(
        tmp_path / "bad.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1"],
        records=[
            ("chr1", 200, [("0", "1")]),
            ("chr1", 100, [("0", "1")]),
        ],
    )
    proc = _run(bad, tmp_path / "out_bad", expect_ok=False)
    assert proc.returncode != 0
    assert "ascending" in proc.stderr or "monotonic" in proc.stderr.lower()


# ── T5: sanity checks all pass on golden fixture ──────────────────────────

def test_all_checks_pass_on_golden(tmp_path, write_vcf):
    """All eight sanity checks must be True in provenance.json for a
    well-formed input."""
    vcf = write_vcf(
        tmp_path / "ok.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1,eur=2>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1", "s2"],
        records=[
            ("chr1", 100, [("0", "1"), ("2", "0")]),
            ("chr1", 200, [("0", "1"), ("2", "0")]),
            ("chr1", 300, [("1", "1"), ("2", "0")]),
        ],
    )
    out = tmp_path / "out"
    _run(vcf, out)
    prov = json.loads((out / "provenance.json").read_text())
    checks = prov["checks_passed"]
    expected = {
        "panel_K", "no_missing_an", "ancestry_idx_in_range",
        "site_positions_monotonic", "tract_bounds", "n_sites_match",
        "n_transitions_match", "all_samples_covered",
    }
    assert set(checks.keys()) == expected
    assert all(checks.values()), f"checks failed: {checks} reasons={prov['check_reasons']}"


# ── T6: per_sample_totals round-trip ──────────────────────────────────────

def test_per_sample_totals_totals(tmp_path, write_vcf):
    """per_sample_totals must sum to n_tracts_emitted and each row's
    total_bp is (end - start + 1) summed across tracts of that ancestry."""
    vcf = write_vcf(
        tmp_path / "t.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1"],
        records=[
            ("chr1", 100, [("0", "0")]),
            ("chr1", 200, [("0", "0")]),
            ("chr1", 300, [("1", "0")]),
            ("chr1", 400, [("1", "0")]),
        ],
    )
    out = tmp_path / "out"
    _run(vcf, out)
    prov = json.loads((out / "provenance.json").read_text())
    pst = _read_parquet(out / "per_sample_totals.parquet")

    assert sum(r["n_tracts"] for r in pst) == prov["n_tracts_emitted"]

    # s1 h0: two tracts. anc=0 covers 100..200 (101 bp). anc=1 covers 300..400 (101 bp).
    s1_h0 = {r["ancestry_idx"]: r for r in pst
             if r["sample_idx"] == 0 and r["hap"] == 0}
    assert s1_h0[0]["n_tracts"] == 1 and s1_h0[0]["total_bp"] == 101
    assert s1_h0[1]["n_tracts"] == 1 and s1_h0[1]["total_bp"] == 101

    # s1 h1: one tract, anc=0 covers 100..400 = 301 bp.
    s1_h1 = {r["ancestry_idx"]: r for r in pst
             if r["sample_idx"] == 0 and r["hap"] == 1}
    assert s1_h1[0]["n_tracts"] == 1 and s1_h1[0]["total_bp"] == 301


# ── T7: no ANCESTRY header aborts ─────────────────────────────────────────

def test_no_ancestry_header_aborts(tmp_path, write_vcf):
    vcf_path = tmp_path / "no_anc.vcf"
    vcf_path.write_text(
        "##fileformat=VCFv4.2\n"
        "##contig=<ID=chr1,length=1000000,assembly=GRCh38>\n"
        '##FORMAT=<ID=AN1,Number=1,Type=Integer,Description="a">\n'
        '##FORMAT=<ID=AN2,Number=1,Type=Integer,Description="b">\n'
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\n"
        "chr1\t100\t.\tA\tT\t.\tPASS\t.\tAN1:AN2\t0:1\n"
    )
    proc = _run(vcf_path, tmp_path / "out", expect_ok=False)
    assert proc.returncode != 0
    assert "##ANCESTRY" in proc.stderr


# ── T8: n_transitions == n_tracts - 1 per (sample, hap, chrom) ────────────

def test_multi_chrom_state_machine(tmp_path, write_vcf):
    """Two chroms in one shard. The state machine must close all open
    tracts with close_reason='chrom_end' when crossing a chrom boundary,
    then re-initialise per-sample state on the next chrom. No state
    leakage between chroms."""
    vcf = write_vcf(
        tmp_path / "multi.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1,eur=2>",
        contigs=[("chr1", 1_000_000), ("chr2", 1_000_000)],
        samples=["s1", "s2"],
        records=[
            # chr1: s1 stays anc=0 (h0), s2 flips 2 -> 1 at pos 200 (h0)
            ("chr1", 100, [("0", "1"), ("2", "0")]),
            ("chr1", 200, [("0", "1"), ("1", "0")]),
            # chr2: s1 flips 0 -> 2 at pos 100->200 (h0),
            #       s2 stays anc=1 (h0)
            ("chr2", 100, [("0", "1"), ("1", "0")]),
            ("chr2", 200, [("2", "1"), ("1", "0")]),
        ],
    )
    out = tmp_path / "out"
    _run(vcf, out)

    tracts = _read_parquet(out / "tracts.parquet")

    # Every (sample, hap, chrom1) tract that closed at the chrom boundary
    # must use close_reason='chrom_end'; every (sample, hap, chrom2) tract
    # that closed at the end of the shard uses 'shard_end'.
    chrom1_last_by_key: dict[tuple[int, int], dict] = {}
    chrom2_last_by_key: dict[tuple[int, int], dict] = {}
    for t in tracts:
        key = (t["sample_idx"], t["hap"])
        if t["chrom"] == "chr1":
            existing = chrom1_last_by_key.get(key)
            if existing is None or t["end_bp"] > existing["end_bp"]:
                chrom1_last_by_key[key] = t
        else:
            existing = chrom2_last_by_key.get(key)
            if existing is None or t["end_bp"] > existing["end_bp"]:
                chrom2_last_by_key[key] = t

    for k, t in chrom1_last_by_key.items():
        assert t["close_reason"] == "chrom_end", (
            f"{k} chr1 last tract closed as {t['close_reason']}, expected chrom_end"
        )
    for k, t in chrom2_last_by_key.items():
        assert t["close_reason"] == "shard_end", (
            f"{k} chr2 last tract closed as {t['close_reason']}, expected shard_end"
        )

    # No tract carries chr1's ancestry into chr2's coordinate range:
    # every tract's chrom must match its position range, per site_positions.
    sp = _read_parquet(out / "site_positions.parquet")
    chr1_sites = {r["position_bp"] for r in sp if r["chrom"] == "chr1"}
    chr2_sites = {r["position_bp"] for r in sp if r["chrom"] == "chr2"}
    assert chr1_sites == {100, 200}
    assert chr2_sites == {100, 200}

    # Per-chrom, per-sample tract span equals the chrom's site span.
    for c, positions in (("chr1", chr1_sites), ("chr2", chr2_sites)):
        min_pos, max_pos = min(positions), max(positions)
        for sidx in (0, 1):
            for hap in (0, 1):
                sub = [t for t in tracts
                       if t["chrom"] == c
                       and t["sample_idx"] == sidx and t["hap"] == hap]
                assert sub, f"no tracts for {(sidx, hap, c)}"
                got_min = min(t["start_bp"] for t in sub)
                got_max = max(t["end_bp"] for t in sub)
                assert got_min == min_pos, f"{(sidx, hap, c)} start {got_min} != {min_pos}"
                assert got_max == max_pos, f"{(sidx, hap, c)} end {got_max} != {max_pos}"

    # Provenance carries per-chrom site spans.
    prov = json.loads((out / "provenance.json").read_text())
    assert set(prov["chrom_site_spans"].keys()) == {"chr1", "chr2"}
    assert prov["chrom_site_spans"]["chr1"] == {
        "first_pos": 100, "last_pos": 200, "n_sites": 2,
    }
    assert prov["chrom_site_spans"]["chr2"] == {
        "first_pos": 100, "last_pos": 200, "n_sites": 2,
    }
    assert all(prov["checks_passed"].values()), prov["check_reasons"]


def test_n_transitions_equals_n_tracts_minus_one(tmp_path, write_vcf):
    vcf = write_vcf(
        tmp_path / "n.vcf",
        panel_line="##ANCESTRY=<eas=0,amr=1,eur=2>",
        contigs=[("chr1", 1_000_000)],
        samples=["s1", "s2"],
        records=[
            ("chr1", 100, [("0", "0"), ("2", "2")]),
            ("chr1", 200, [("1", "0"), ("2", "1")]),
            ("chr1", 300, [("1", "0"), ("0", "1")]),
            ("chr1", 400, [("2", "0"), ("0", "1")]),
        ],
    )
    out = tmp_path / "out"
    _run(vcf, out)

    tracts = _read_parquet(out / "tracts.parquet")
    trans = _read_parquet(out / "transitions.parquet")

    from collections import Counter
    tract_counts: Counter = Counter()
    trans_counts: Counter = Counter()
    for t in tracts:
        tract_counts[(t["sample_idx"], t["hap"], t["chrom"])] += 1
    for t in trans:
        trans_counts[(t["sample_idx"], t["hap"], t["chrom"])] += 1
    for k, nt in tract_counts.items():
        assert trans_counts.get(k, 0) == nt - 1, (
            f"key {k}: n_tracts={nt} but n_transitions={trans_counts.get(k, 0)}"
        )
