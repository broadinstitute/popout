#!/usr/bin/env python3
"""popout DX report — build a PDF from a cohort bundle.

Thin driver. All section content lives in:

  - ``popout.reports.charts`` (one chart module per section)
  - ``popout.reports.templates.popout_dx`` (one Jinja2 template per section)
  - ``configs/popout_dx_report.yaml`` (the section manifest)

Phase-4 rewrite: the legacy 1,100-line monolithic builder that
lived here has been retired into the modular structure above.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from popout.reports import (                                # noqa: E402
    ReportContext,
    load_report_config,
    render_report,
    run_pandoc,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "popout_dx_report.yaml"


def _log(msg: str) -> None:
    ts = dt.datetime.now(dt.timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] build_popout_dx_report: {msg}", file=sys.stderr, flush=True)


def _untar_to(src: Path, dest: Path) -> Path:
    """Extract a popout DX cohort tarball; return the inner ``cohort_dx`` dir."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(src, "r:*") as tar:
        members = tar.getmembers()
        if not members:
            raise RuntimeError(f"{src} is empty")
        top = members[0].name.split("/", 1)[0]
        tar.extractall(dest, filter="data")
    inner = dest / top
    if not (inner / "cohort_manifest.json").is_file():
        raise RuntimeError(
            f"{src}: extracted {inner} does not contain cohort_manifest.json"
        )
    return inner


def resolve_bundle_dir(arg: Path, tmpdir: Path) -> Path:
    """Accept either an unpacked cohort_dx/ dir or a *.tar.gz tarball."""
    if arg.is_dir():
        if (arg / "cohort_manifest.json").is_file():
            return arg
        inner = arg / "cohort_dx"
        if (inner / "cohort_manifest.json").is_file():
            return inner
        raise FileNotFoundError(
            f"{arg}: no cohort_manifest.json at top level or inside cohort_dx/"
        )
    if arg.is_file() and (arg.name.endswith(".tar.gz") or arg.name.endswith(".tgz")):
        return _untar_to(arg, tmpdir)
    raise FileNotFoundError(f"{arg}: not a directory or tarball")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render the popout DX report PDF from a cohort bundle.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cohort-bundle", type=Path, required=True,
                   help="path to cohort_dx/<run>.tar.gz or an unpacked cohort_dx/ dir")
    p.add_argument("--out", type=Path, required=True,
                   help="destination PDF (or .md) path")
    p.add_argument("--report-config", type=Path, default=DEFAULT_CONFIG,
                   help="YAML report manifest")
    p.add_argument("--keep-md", action="store_true",
                   help="keep the intermediate .md document next to the PDF")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    md_only = args.out.suffix.lower() == ".md"
    md_path = args.out if md_only else args.out.parent / f"{args.out.stem}.md"
    assets_dir = args.out.parent / f"{args.out.stem}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="popout_dx_report_") as tmpdir:
        bundle_dir = resolve_bundle_dir(args.cohort_bundle, Path(tmpdir))
        manifest = json.loads((bundle_dir / "cohort_manifest.json").read_text())
        summary: dict = {}
        if (bundle_dir / "cohort_summary.json").exists():
            summary = json.loads((bundle_dir / "cohort_summary.json").read_text())
        bundle = {**manifest, **summary}

        cfg = load_report_config(args.report_config)
        _log(f"loaded {len(cfg.sections)} sections from {args.report_config}")

        ctx = ReportContext(
            bundle=bundle, bundle_dir=bundle_dir,
            config=cfg, estimates={}, assets_dir=assets_dir,
        )
        md = render_report(ctx)
        md_path.write_text(md)
        _log(f"wrote {md_path} ({md.count(chr(10))} lines)")

        if md_only:
            _log("--out is .md; skipping pandoc")
        else:
            run_pandoc(md_path, args.out, style=cfg.style)
            if not args.keep_md:
                md_path.unlink(missing_ok=True)
            _log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
