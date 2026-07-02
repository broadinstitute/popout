#!/usr/bin/env python3
"""FLARE validation Stage 3 — build a PDF from a cohort bundle.

Thin driver. All section content lives in:

  - ``popout.reports.charts`` (one chart module per section)
  - ``popout.reports.templates.flare_validation`` (one Jinja2 template per section)
  - ``configs/flare_validation_report.yaml`` (the section manifest)

The legacy monolithic builder that lived here in pre-Jinja2 days has
been retired; its 2,483 lines have moved into the modular structure
above. See ``my_notes/validation/COLLECTOR_FIXES.md`` for what the
stats collector needs to do so reports stay faithful.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from popout.reports import (                                # noqa: E402
    ReportContext,
    load_report_config,
    render_report,
    run_pandoc,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "flare_validation_report.yaml"


def _log(msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] build_flare_validation_report: {msg}",
          file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the FLARE validation PDF from a cohort bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cohort-bundle", type=Path, required=True,
                   help="path to the unpacked cohort_bundle/ directory")
    p.add_argument("--out", type=Path, required=True,
                   help="destination PDF (or .md) path")
    p.add_argument("--report-config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML report manifest")
    p.add_argument("--keep-md", action="store_true",
                   help="keep the intermediate .md document next to the PDF")
    p.add_argument("--rye-q", type=Path, default=None,
                   help="external Rye Q TSV (per-sample SP5 proportions). "
                        "Required by the panel_coverage_attribution section; "
                        "other sections render without it.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    bundle_dir = args.cohort_bundle
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"{bundle_dir}: not a directory")
    if not (bundle_dir / "cohort_manifest.json").is_file():
        inner = bundle_dir / "cohort_bundle"
        if (inner / "cohort_manifest.json").is_file():
            bundle_dir = inner
        else:
            raise FileNotFoundError(
                f"{args.cohort_bundle}: no cohort_manifest.json "
                "(neither top-level nor under cohort_bundle/)"
            )

    manifest = json.loads((bundle_dir / "cohort_manifest.json").read_text())
    summary: dict = {}
    if (bundle_dir / "cohort_summary.json").exists():
        summary = json.loads((bundle_dir / "cohort_summary.json").read_text())
    bundle = {**manifest, **summary}

    cfg = load_report_config(args.report_config)
    _log(f"loaded {len(cfg.sections)} sections from {args.report_config}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md_only = args.out.suffix.lower() == ".md"
    md_path = args.out if md_only else args.out.parent / f"{args.out.stem}.md"
    assets_dir = args.out.parent / f"{args.out.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    ctx = ReportContext(
        bundle=bundle, bundle_dir=bundle_dir,
        config=cfg, estimates={}, assets_dir=assets_dir,
        rye_q=args.rye_q,
    )
    md = render_report(ctx)
    md_path.write_text(md)
    _log(f"wrote {md_path} ({md.count(chr(10))} lines)")

    if md_only:
        _log("--out is .md; skipping pandoc")
    else:
        run_pandoc(md_path, args.out, style=cfg.style, draft=cfg.draft)
        if not args.keep_md:
            md_path.unlink(missing_ok=True)
        _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
