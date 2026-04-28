"""Tests for popout.prior_spec — schema-v2 priors YAML loader."""

from __future__ import annotations

import gzip
import textwrap
from pathlib import Path

import numpy as np
import pytest

from popout.identity import AIMSignature, FSTReferenceSignature
from popout.prior_spec import (
    LinearAnnealingSchedule,
    Prior,
    Priors,
    load_priors,
    prior_to_beta,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _write_aim_tsv(path: Path, n: int = 5, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    lines = ["chrom\tpos_bp\tref\talt\texpected_freq\tweight\tsource"]
    for i in range(n):
        chrom = "1" if i < n // 2 else "2"
        pos = 10_000 + i * 100
        freq = float(rng.uniform(0.1, 0.9))
        lines.append(f"{chrom}\t{pos}\tA\tG\t{freq:.4f}\t1.0\trsTEST{i}")
    path.write_text("\n".join(lines) + "\n")


def _write_ref_tsv(path: Path, n_per_chrom: int = 10, seed: int = 0) -> None:
    """Write a tiny fake 1KG-style reference TSV."""
    rng = np.random.default_rng(seed)
    rows = ["#chrom\tpos\tref\talt\tEUR\tEAS\tAMR\tAFR\tSAS"]
    for chrom in ("1", "2"):
        for i in range(n_per_chrom):
            pos = 10_000 + i * 100
            freqs = rng.uniform(0.05, 0.95, 5)
            rows.append(
                f"{chrom}\t{pos}\tA\tG\t"
                + "\t".join(f"{x:.4f}" for x in freqs),
            )
    body = ("\n".join(rows) + "\n").encode()
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as f:
            f.write(body)
    else:
        path.write_bytes(body)


@pytest.fixture
def yaml_dir(tmp_path):
    aim = tmp_path / "afr.tsv"
    _write_aim_tsv(aim, n=6, seed=1)
    ref = tmp_path / "ref.tsv.gz"
    _write_ref_tsv(ref, n_per_chrom=20, seed=2)
    return tmp_path, aim, ref


def _yaml_v2(aim_rel: str, ref_rel: str) -> str:
    return textwrap.dedent(
        f"""\
        schema_version: 2
        morgans_per_step: 1.0e-4
        priors:
          - name: AFR
            identity:
              aims:
                panel: {aim_rel}
                weight: 1.5
              fst_reference:
                superpop: AFR
                superpop_freqs_path: {ref_rel}
            parameters:
              gen:
                mean: 7
                range: [4, 12]
            source: "test AFR"
          - name: EUR
            identity:
              fst_reference:
                superpop: EUR
                superpop_freqs_path: {ref_rel}
            parameters:
              gen:
                mean: 2
                range: [1, 4]
        annealing:
          schedule: linear
          tau_start: 0.8
          tau_end: 0.2
          ramp_iters: 5
        """
    )


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


def test_load_priors_round_trip(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "priors.yaml"
    yml.write_text(_yaml_v2(aim.name, ref.name))

    priors = load_priors(yml)
    assert isinstance(priors, Priors)
    assert priors.morgans_per_step == 1.0e-4
    assert len(priors.priors) == 2

    afr = priors.get("AFR")
    assert afr is not None
    assert afr.gen_mean == 7
    assert afr.gen_lo == 4
    assert afr.gen_hi == 12
    assert afr.alpha > 0 and afr.beta > 0
    assert afr.source == "test AFR"
    # AFR has both AIM + F_ST
    assert len(afr.identity_signatures) == 2
    assert isinstance(afr.identity_signatures[0], AIMSignature)
    assert afr.identity_signatures[0].weight == 1.5
    assert isinstance(afr.identity_signatures[1], FSTReferenceSignature)

    eur = priors.get("EUR")
    assert eur is not None
    # EUR has F_ST only
    assert len(eur.identity_signatures) == 1
    assert isinstance(eur.identity_signatures[0], FSTReferenceSignature)

    # Annealing
    assert priors.annealing.tau_start == 0.8
    assert priors.annealing.tau_end == 0.2
    assert priors.annealing.ramp_iters == 5


def test_load_priors_rejects_v1_schema(tmp_path):
    yml = tmp_path / "v1.yaml"
    yml.write_text(textwrap.dedent("""\
        morgans_per_step: 1.0e-4
        components:
          - component_idx: 0
            gen_mean: 7
            gen_lo: 4
            gen_hi: 12
    """))
    with pytest.raises(ValueError, match="Schema v1"):
        load_priors(yml)


def test_load_priors_rejects_explicit_v1(tmp_path):
    yml = tmp_path / "explicit_v1.yaml"
    yml.write_text(textwrap.dedent("""\
        schema_version: 1
        morgans_per_step: 1.0e-4
    """))
    with pytest.raises(ValueError, match="Schema v1"):
        load_priors(yml)


def test_load_priors_rejects_unknown_schema(tmp_path):
    yml = tmp_path / "future.yaml"
    yml.write_text(textwrap.dedent("""\
        schema_version: 99
        morgans_per_step: 1.0e-4
        priors: []
    """))
    with pytest.raises(ValueError, match="schema_version"):
        load_priors(yml)


def test_load_priors_rejects_missing_morgans_per_step(tmp_path):
    yml = tmp_path / "no_mps.yaml"
    yml.write_text(textwrap.dedent("""\
        schema_version: 2
        priors:
          - name: X
            identity: {fst_reference: {superpop: AFR}}
            parameters: {gen: {mean: 7, range: [4, 12]}}
    """))
    with pytest.raises(ValueError, match="morgans_per_step"):
        load_priors(yml)


def test_load_priors_rejects_duplicate_names(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "dup.yaml"
    yml.write_text(textwrap.dedent(f"""\
        schema_version: 2
        morgans_per_step: 1.0e-4
        priors:
          - name: AFR
            identity:
              fst_reference: {{superpop: AFR, superpop_freqs_path: {ref.name}}}
            parameters: {{gen: {{mean: 7, range: [4, 12]}}}}
          - name: AFR
            identity:
              fst_reference: {{superpop: AFR, superpop_freqs_path: {ref.name}}}
            parameters: {{gen: {{mean: 7, range: [4, 12]}}}}
    """))
    with pytest.raises(ValueError, match="duplicate prior name"):
        load_priors(yml)


def test_load_priors_rejects_empty_identity(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "no_id.yaml"
    yml.write_text(textwrap.dedent("""\
        schema_version: 2
        morgans_per_step: 1.0e-4
        priors:
          - name: BARE
            identity: {}
            parameters: {gen: {mean: 7, range: [4, 12]}}
    """))
    with pytest.raises(ValueError, match="identity"):
        load_priors(yml)


def test_load_priors_rejects_gen_out_of_range(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "bad_gen.yaml"
    yml.write_text(textwrap.dedent(f"""\
        schema_version: 2
        morgans_per_step: 1.0e-4
        priors:
          - name: BAD
            identity:
              fst_reference: {{superpop: AFR, superpop_freqs_path: {ref.name}}}
            parameters: {{gen: {{mean: 100, range: [4, 12]}}}}
    """))
    with pytest.raises(ValueError, match="gen_lo < gen_mean < gen_hi"):
        load_priors(yml)


# --------------------------------------------------------------------------
# Fingerprint
# --------------------------------------------------------------------------


def test_fingerprint_stable_across_loads(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "p.yaml"
    yml.write_text(_yaml_v2(aim.name, ref.name))
    a = load_priors(yml).fingerprint
    b = load_priors(yml).fingerprint
    assert a == b


def test_fingerprint_changes_when_yaml_changes(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "p.yaml"
    yml.write_text(_yaml_v2(aim.name, ref.name))
    a = load_priors(yml).fingerprint
    # Tweak the YAML (different gen_mean) — fingerprint must change.
    yml.write_text(_yaml_v2(aim.name, ref.name).replace("mean: 7", "mean: 8"))
    b = load_priors(yml).fingerprint
    assert a != b


def test_fingerprint_changes_when_aim_panel_content_changes(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "p.yaml"
    yml.write_text(_yaml_v2(aim.name, ref.name))
    a = load_priors(yml).fingerprint
    # Rewrite the panel with different content.
    _write_aim_tsv(aim, n=6, seed=42)
    b = load_priors(yml).fingerprint
    assert a != b, "panel content change must invalidate fingerprint"


def test_fingerprint_changes_when_ref_tsv_content_changes(yaml_dir):
    tmp_path, aim, ref = yaml_dir
    yml = tmp_path / "p.yaml"
    yml.write_text(_yaml_v2(aim.name, ref.name))
    a = load_priors(yml).fingerprint
    _write_ref_tsv(ref, n_per_chrom=20, seed=99)
    b = load_priors(yml).fingerprint
    assert a != b, "ref TSV content change must invalidate fingerprint"


# --------------------------------------------------------------------------
# Annealing
# --------------------------------------------------------------------------


def test_linear_annealing_schedule_endpoints():
    s = LinearAnnealingSchedule(tau_start=1.0, tau_end=0.1, ramp_iters=10)
    assert s.tau_at(0) == 1.0
    assert s.tau_at(10) == 0.1
    assert s.tau_at(15) == 0.1                # constant after
    assert pytest.approx(s.tau_at(5)) == 0.55  # linear midpoint


def test_linear_annealing_rejects_bad_params():
    with pytest.raises(ValueError):
        LinearAnnealingSchedule(tau_start=0.0, tau_end=0.1, ramp_iters=10)
    with pytest.raises(ValueError):
        LinearAnnealingSchedule(tau_start=1.0, tau_end=0.1, ramp_iters=0)


# --------------------------------------------------------------------------
# Beta solver (smoke — full coverage in the original v1 tests was deleted)
# --------------------------------------------------------------------------


def test_prior_to_beta_basic():
    a, b = prior_to_beta(gen_mean=7, gen_lo=4, gen_hi=12, morgans_per_step=1e-4)
    assert a > 0 and b > 0
    # Beta should be skewed toward small r (small T) — meaning beta > alpha.
    assert b > a
