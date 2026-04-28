"""LD-prune, rank, and finalize the production panel for one population.

Phases 4-6 of the spec compressed into one driver:

* Phase 4 — pairwise within-population LD pruning. Without 1KG
  genotypes we can't compute true r²; the proxy is the heuristic
  "candidates within ``min_separation_bp`` of each other on the same
  chromosome are likely correlated." When two candidates are close,
  drop the one with smaller separation. Conservative — most genuine
  AIMs are far apart, and the framework's variance-normalized scoring
  partially absorbs redundancy anyway.
* Phase 5 — rank surviving candidates by separation magnitude.
* Phase 6 — take the top N and write the production panel TSV at
  ``popout/data/aim_panels/{name}.tsv`` plus the build log at
  ``docs/aim_panels/build_logs/{pop}.tsv``.

Determinism: candidate ordering is by (separation desc, chrom asc,
pos asc); identical inputs always produce identical outputs.

Usage::

    python -m scripts.build_aim_panels.build_panel \\
        --pop AFR --target-size 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .lib import (
    BuildLogEntry,
    Candidate,
    read_candidates,
    write_build_log,
    write_panel,
)

log = logging.getLogger(__name__)


# Production-panel output filenames.
_FILENAMES: dict[str, str] = {
    "AFR": "african.tsv",
    "EUR": "european.tsv",
    "EAS": "east_asian.tsv",
    "AMR": "native_american.tsv",
    "SAS": "south_asian.tsv",
    "MID": "middle_east.tsv",
}


def _chrom_sort_key(c: str) -> tuple[int, str]:
    """Sort chromosomes 1..22 then X, Y, MT."""
    norm = c.replace("chr", "")
    try:
        return (int(norm), "")
    except ValueError:
        return (1000, norm)


def _prune_linkage(
    cands: list[Candidate], min_sep_bp: int,
) -> tuple[list[Candidate], list[BuildLogEntry]]:
    """Drop the weaker of any pair of candidates within
    ``min_sep_bp`` on the same chromosome.

    Operates on the input order (which the caller can choose to be
    by-separation desc — that way the "stronger" candidate wins ties
    deterministically).
    """
    kept: list[Candidate] = []
    pruned_log: list[BuildLogEntry] = []
    by_chrom: dict[str, list[int]] = {}
    for c in cands:
        positions = by_chrom.setdefault(c.chrom, [])
        too_close = any(abs(c.pos - p) <= min_sep_bp for p in positions)
        if too_close:
            pruned_log.append(BuildLogEntry(
                chrom=c.chrom, pos=c.pos, ref=c.ref, alt=c.alt,
                target_pop=c.target_pop, expected_freq=c.expected_freq,
                others_max_freq=c.others_max_freq,
                separation=c.separation, stage="linkage_pruned",
                reason=(
                    f"within {min_sep_bp} bp of stronger candidate "
                    f"on {c.chrom}"
                ),
            ))
            continue
        positions.append(c.pos)
        kept.append(c)
    return kept, pruned_log


def build(
    *,
    pop: str,
    candidates_path: Path,
    panel_dir: Path,
    build_log_dir: Path,
    target_size: int,
    min_sep_bp: int = 1_000_000,
) -> dict[str, object]:
    """Run phases 4-6 on one population's candidate file.

    Returns a small summary dict (sizes, output paths) for logging.
    """
    candidates = read_candidates(candidates_path)
    if not candidates:
        raise ValueError(
            f"No candidates in {candidates_path}; run scan_candidates first."
        )
    log.info(
        "loaded %d %s candidates from %s",
        len(candidates), pop, candidates_path,
    )

    # Sort by (separation desc, chrom asc, pos asc) for deterministic
    # downstream behavior.
    candidates.sort(
        key=lambda c: (-c.separation, _chrom_sort_key(c.chrom), c.pos),
    )

    pruned, prune_log = _prune_linkage(candidates, min_sep_bp=min_sep_bp)
    log.info(
        "%d candidates remain after %d-bp linkage pruning (dropped %d)",
        len(pruned), min_sep_bp, len(prune_log),
    )

    selected = pruned[:target_size]
    log.info("selected top %d for %s panel", len(selected), pop)

    panel_path = panel_dir / _FILENAMES[pop]
    write_panel(panel_path, selected, pop)
    log.info("wrote panel to %s", panel_path)

    log_entries: list[BuildLogEntry] = []
    for c in selected:
        log_entries.append(BuildLogEntry(
            chrom=c.chrom, pos=c.pos, ref=c.ref, alt=c.alt,
            target_pop=pop, expected_freq=c.expected_freq,
            others_max_freq=c.others_max_freq,
            separation=c.separation, stage="selected",
            reason="top-N by separation, post-pruning",
        ))
    log_entries.extend(prune_log)
    # Track candidates that survived pruning but weren't in top N.
    for c in pruned[target_size:]:
        log_entries.append(BuildLogEntry(
            chrom=c.chrom, pos=c.pos, ref=c.ref, alt=c.alt,
            target_pop=pop, expected_freq=c.expected_freq,
            others_max_freq=c.others_max_freq,
            separation=c.separation, stage="below_top_n",
            reason="passed pruning but ranked below target_size",
        ))

    log_path = build_log_dir / f"{pop.lower()}.tsv"
    write_build_log(log_path, log_entries)
    log.info("wrote build log to %s", log_path)

    return {
        "pop": pop,
        "n_candidates": len(candidates),
        "n_pruned": len(prune_log),
        "n_selected": len(selected),
        "panel_path": str(panel_path),
        "build_log_path": str(log_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pop", required=True,
        choices=tuple(_FILENAMES),
        help="AoU superpopulation",
    )
    parser.add_argument(
        "--candidates", type=Path, default=None,
        help="Candidate TSV (default: build/aim_panels/{pop}_candidates.tsv)",
    )
    parser.add_argument(
        "--panel-dir", type=Path,
        default=Path("popout/data/aim_panels"),
    )
    parser.add_argument(
        "--build-log-dir", type=Path,
        default=Path("docs/aim_panels/build_logs"),
    )
    parser.add_argument(
        "--target-size", type=int, default=10,
        help="Top N candidates to write to the production panel",
    )
    parser.add_argument(
        "--min-sep-bp", type=int, default=1_000_000,
        help="Minimum bp distance between panel markers on the same chrom",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cands_path = (
        args.candidates if args.candidates is not None
        else Path("build/aim_panels") / f"{args.pop.lower()}_candidates.tsv"
    )

    summary = build(
        pop=args.pop,
        candidates_path=cands_path,
        panel_dir=args.panel_dir,
        build_log_dir=args.build_log_dir,
        target_size=args.target_size,
        min_sep_bp=args.min_sep_bp,
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
