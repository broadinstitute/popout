#!/usr/bin/env python3
"""Smoke test: build the popout DX PDF report from the e2e fixture bundle.

Reuses ``test_e2e_fixture.run()`` to materialise a tiny cohort bundle
(2 clusters × chr1 × 60 synthetic samples), then drives
``build_popout_dx_report.py`` against it.

By default emits a markdown report and verifies its structure (no
pandoc/xelatex required). Set ``POPOUT_DX_REPORT_TEST_PDF=1`` to also
emit + sanity-check the PDF.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


THIS = Path(__file__).resolve()
PKG_ROOT = THIS.parents[1]                      # validation/popout_dx
REPO_ROOT = PKG_ROOT.parents[1]                 # gpulai
SCRIPTS = PKG_ROOT / "scripts"
REPORT_SCRIPT = SCRIPTS / "build_popout_dx_report.py"

EXPECTED_SECTIONS = (
    "# popout DX cohort report",
    "# Reading guide",
    "# Headline pass-rate grid",
    "# Per-tool global concordance",
    "# Per-sample MAE distribution",
    "# Hard-call confusion",
    "# Per-(cluster, chrom) performance grid",
    "# Provenance",
)

# Phase 5 of the label-space retrofit: the cover carries the figure-tag
# shorthand sourced from the cohort's labels.json provenance.tag.
EXPECTED_TOKENS = (
    "**Label space:** `L=SP6/",
)


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, env={**os.environ})
    if r.returncode != 0:
        raise SystemExit(f"command failed (exit {r.returncode}): {cmd}")


def main() -> int:
    from validation.popout_dx.tests import test_e2e_fixture

    workdir = Path(tempfile.mkdtemp(prefix="popout_dx_report_smoke_"))
    try:
        test_e2e_fixture.run(workdir, mode="global")
        bundle = workdir / "cohort_dx.smoke_e2e_global.v1.0.0.tar.gz"
        if not bundle.is_file():
            raise SystemExit(f"fixture did not produce expected bundle: {bundle}")

        md_out = workdir / "report.md"
        _run([
            sys.executable, str(REPORT_SCRIPT),
            "--cohort-bundle", str(bundle),
            "--out", str(md_out),
        ])
        if not md_out.is_file():
            raise SystemExit(f"report builder did not write {md_out}")
        md = md_out.read_text()
        missing = [h for h in EXPECTED_SECTIONS if h not in md]
        if missing:
            raise SystemExit(
                f"markdown report missing expected headings: {missing}"
            )
        missing_tokens = [t for t in EXPECTED_TOKENS if t not in md]
        if missing_tokens:
            raise SystemExit(
                f"markdown report missing expected tokens: {missing_tokens}"
            )

        assets = workdir / "report_assets"
        if not (assets / "traffic_light.png").is_file():
            raise SystemExit(f"missing traffic-light asset under {assets}")

        if os.environ.get("POPOUT_DX_REPORT_TEST_PDF"):
            for tool in ("pandoc", "xelatex"):
                if shutil.which(tool) is None:
                    print(f"  skipping PDF: {tool} not on PATH")
                    break
            else:
                pdf_out = workdir / "report.pdf"
                _run([
                    sys.executable, str(REPORT_SCRIPT),
                    "--cohort-bundle", str(bundle),
                    "--out", str(pdf_out),
                ])
                if not pdf_out.is_file() or pdf_out.stat().st_size < 1024:
                    raise SystemExit(
                        f"PDF not produced or suspiciously small: {pdf_out}"
                    )

        print(f"\n✓ popout DX report smoke PASS — md: {md_out}")
        return 0
    finally:
        if not os.environ.get("POPOUT_DX_REPORT_TEST_KEEP"):
            shutil.rmtree(workdir, ignore_errors=True)
        else:
            print(f"  (kept workspace at {workdir})")


if __name__ == "__main__":
    sys.exit(main())
