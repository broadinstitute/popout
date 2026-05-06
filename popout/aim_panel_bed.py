"""Emit a 0-based half-open BED of every AIM panel position from a priors YAML.

The BED is consumed by the upstream Terra ``filter_pgen.wdl`` task to
protect AIM panel sites from cohort-shape filtering (``--maf``, ``--hwe``,
``--exclude-palindromic-snps``, etc.). It can also seed the static
``popout/data/aim_panels/all_panels.bed`` artifact bundled with the
six default panels.

Usage::

    popout aim-panel-bed --priors configs/priors_v2.yaml --out aim.bed
    popout aim-panel-bed --priors configs/priors_v2.yaml         # → stdout

Output format: one row per (chrom, pos_bp) referenced by any AIM panel
in any prior. ``chrom`` retains a ``chr`` prefix when present in the
panel TSV; coordinates are 0-based half-open
(``chrom\\t{pos_bp - 1}\\t{pos_bp}``). Sorted by (chrom_natural, start).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .identity import _normalize_chrom


log = logging.getLogger(__name__)


def _chrom_sort_key(c: str) -> tuple[int, str]:
    """Sort 1..22, then X, Y, MT, then anything else lexically."""
    norm = _normalize_chrom(c)
    if norm.isdigit():
        return (int(norm), "")
    return (1000, norm)


def emit_bed(priors_path: str | Path, out: object) -> int:
    """Write the BED to ``out`` (a file-like object). Return row count."""
    from .prior_spec import load_priors
    from .identity import AIMSignature

    # skip_fst=True: this utility only needs AIM panel paths to emit
    # the BED; the FST signature would require loading a (possibly
    # absent) superpop_freqs TSV which adds nothing here.
    priors = load_priors(priors_path, skip_fst=True)

    # Aggregate (chrom, pos_bp) preserving panel-supplied chrom strings;
    # the BED feeds plink2 and chrom-name format must match the cohort
    # PGEN. _normalize_chrom is used only for sorting/dedup keys.
    by_chrom: dict[str, set[int]] = {}
    for prior in priors.priors:
        for sig in prior.identity_signatures:
            if not isinstance(sig, AIMSignature):
                continue
            panel = sig.panel
            for c, p in zip(panel.chrom, panel.pos_bp):
                by_chrom.setdefault(str(c), set()).add(int(p))

    rows = []
    for chrom, positions in by_chrom.items():
        for pos in positions:
            rows.append((chrom, pos))
    rows.sort(key=lambda r: (_chrom_sort_key(r[0]), r[1]))

    n = 0
    for chrom, pos in rows:
        out.write(f"{chrom}\t{pos - 1}\t{pos}\n")
        n += 1
    return n


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout aim-panel-bed``."""
    parser = argparse.ArgumentParser(
        description=(
            "Emit a 0-based half-open BED of AIM panel positions from a "
            "priors YAML, for use with plink2 --extract bed0."
        ),
    )
    parser.add_argument(
        "--priors", required=True,
        help="Path to a priors v2 YAML (e.g. configs/priors_v2.yaml).",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output BED path (default: stdout).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            n = emit_bed(args.priors, f)
        log.info("Wrote %d AIM panel positions to %s", n, out_path)
    else:
        n = emit_bed(args.priors, sys.stdout)
        log.info("Wrote %d AIM panel positions to stdout", n)
