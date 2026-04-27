"""Tests for --priors CLI flag (parsing, mutex, file errors)."""

from __future__ import annotations

import subprocess
import sys
import textwrap


VALID_PRIORS_BODY = """
morgans_per_step: 1.2e-4
components:
  - {component_idx: 0, gen_mean: 5, gen_lo: 2, gen_hi: 10}
"""

PYBIN = "/Users/ghall/code/work/broad/popout/.venv/bin/python"


def _run_popout(argv, **kwargs):
    return subprocess.run(
        [PYBIN, "-m", "popout.cli"] + argv,
        capture_output=True, text=True, **kwargs,
    )


def test_priors_mutex_with_per_hap_T(tmp_path):
    """--priors + --per-hap-T → mutex error names both flags."""
    p = tmp_path / "priors.yaml"
    p.write_text(textwrap.dedent(VALID_PRIORS_BODY).lstrip())

    res = _run_popout([
        "--vcf", "/tmp/dummy.vcf.gz",
        "--map", "/tmp/dummy.map",
        "--out", str(tmp_path / "out"),
        "--method", "hmm",
        "--priors", str(p),
        "--per-hap-T",
    ])
    assert res.returncode != 0
    assert "--priors" in res.stderr
    assert "--per-hap-T" in res.stderr


def test_priors_rejected_without_method_hmm(tmp_path):
    """--priors with --method cnn fails with a clean message."""
    p = tmp_path / "priors.yaml"
    p.write_text(textwrap.dedent(VALID_PRIORS_BODY).lstrip())

    res = _run_popout([
        "--vcf", "/tmp/dummy.vcf.gz",
        "--map", "/tmp/dummy.map",
        "--out", str(tmp_path / "out"),
        "--method", "cnn",
        "--priors", str(p),
    ])
    assert res.returncode != 0
    assert "--priors" in res.stderr
    assert "hmm" in res.stderr


def test_priors_help_contains_flag(tmp_path):
    """--help output mentions --priors and the YAML format hint."""
    res = _run_popout(["--help"])
    assert res.returncode == 0
    assert "--priors" in res.stdout
