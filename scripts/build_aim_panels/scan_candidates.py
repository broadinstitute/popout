"""Scan the 1KG superpop frequency TSV for AIM candidates.

For each AoU superpopulation, emits one candidate TSV at
``build/aim_panels/{pop}_candidates.tsv`` listing every site in the
1KG TSV whose target-vs-max-others alt-allele frequency separation
clears a configurable threshold (default 0.40, per the user's spec).

MID is the special case: 1KG has no MID superpop. The scan still
produces a MID candidate file by selecting markers whose 1KG SAS
and EUR frequencies are mutually distinct — a coarse proxy for
"distinguishes Levantine from Indo-European-and-East-Asian." The
real MID frequencies come from the test reference fixture at
``tests/data/aim_panels/mid_reference.tsv``; this scan just provides
the candidate position list.

Output candidate TSVs are gitignored (live under ``build/``); the
provenance docs at ``docs/aim_panels/{pop}.md`` carry the
human-readable enumeration.

Usage::

    python -m scripts.build_aim_panels.scan_candidates \\
        --ref ~/.popout/ref/GRCh38/1kg_superpop_freq.tsv.gz \\
        --out-dir build/aim_panels \\
        [--separation-threshold 0.40]
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path

from popout.fetch_superpop_freqs import resolve_superpop_freqs_path

from .lib import (
    SUPERPOPS_AOU,
    Candidate,
    iter_1kg_rows,
    write_candidates,
)

log = logging.getLogger(__name__)


def _candidate_for_pop(row, pop: str, threshold: float) -> Candidate | None:
    """Score one site for one target population.

    Returns a Candidate if the site clears the separation threshold,
    None otherwise. Direction is recorded so downstream stages can
    distinguish "high in target" from "low in target" markers — both
    are diagnostic; the variance-normalized scoring is direction-
    agnostic.
    """
    pops_freqs = {
        "EUR": row.eur, "EAS": row.eas, "AMR": row.amr,
        "AFR": row.afr, "SAS": row.sas,
    }
    if pop == "MID":
        # MID has no 1KG superpop. The synthetic MID proxy is the
        # midpoint between EUR and SAS — a coarse Levantine model
        # (Levantine populations cluster between European and South
        # Asian ancestries in 1KG-derived PCA, with secondary AFR
        # contributions). The proxy is principled enough for the
        # framework to identify Levantine-like components in
        # production cohorts; biobank-grade MID priors would refresh
        # this against Behar/Haber data when available.
        #
        # A useful MID candidate satisfies:
        #   1. EUR and SAS differ by ≥ threshold (so the midpoint is
        #      meaningfully distinct from both pure populations).
        #   2. The midpoint is at least threshold/2 away from AFR,
        #      EAS, and AMR.
        # These constraints make the panel score the synthetic-MID
        # component higher than any pure 1KG superpop component.
        eur_sas_sep = abs(row.eur - row.sas)
        if eur_sas_sep < threshold:
            return None
        target_freq = (row.eur + row.sas) / 2.0
        margin_eur = abs(target_freq - row.eur)
        margin_sas = abs(target_freq - row.sas)
        margin_afr = abs(target_freq - row.afr)
        margin_eas = abs(target_freq - row.eas)
        margin_amr = abs(target_freq - row.amr)
        # Midpoint is by definition |EUR - SAS|/2 from EUR and SAS;
        # require AFR/EAS/AMR to also be at least threshold/2 away.
        min_other_margin = min(margin_afr, margin_eas, margin_amr)
        if min_other_margin < threshold / 2.0:
            return None
        max_other = max(row.afr, row.eas, row.amr)
        return Candidate(
            chrom=row.chrom, pos=row.pos, ref=row.ref, alt=row.alt,
            expected_freq=target_freq,
            target_pop=pop,
            others_max_freq=max_other,
            separation=min(margin_eur, margin_sas, margin_afr,
                           margin_eas, margin_amr),
            direction="+" if target_freq > max_other else "-",
            pops_freqs=pops_freqs,
            rationale=(
                "synthetic MID proxy: midpoint(EUR, SAS); "
                "Levantine proxy without 1KG MID coverage"
            ),
        )

    target_freq = pops_freqs[pop]
    others = {k: v for k, v in pops_freqs.items() if k != pop}
    max_other = max(others.values())
    min_other = min(others.values())

    # Direction-agnostic separation: prefer the larger of
    # (target - max_other) or (min_other - target).
    sep_high = target_freq - max_other
    sep_low = min_other - target_freq
    if sep_high >= sep_low:
        sep = sep_high
        direction = "+"
        # When target is *high* in target pop, separation is
        # vs max_other.
    else:
        sep = sep_low
        direction = "-"
    if sep < threshold:
        return None

    return Candidate(
        chrom=row.chrom, pos=row.pos, ref=row.ref, alt=row.alt,
        expected_freq=target_freq,
        target_pop=pop,
        others_max_freq=max_other if direction == "+" else min_other,
        separation=sep,
        direction=direction,
        pops_freqs=pops_freqs,
        rationale="data-driven from 1KG separation",
    )


def scan(
    superpop_freqs_path: Path,
    out_dir: Path,
    separation_threshold: float = 0.40,
    log_every: int = 1_000_000,
) -> dict[str, int]:
    """Scan the 1KG TSV; emit per-population candidate files.

    Returns a mapping from population to candidate count.
    """
    candidates: dict[str, list[Candidate]] = defaultdict(list)
    n_rows = 0
    for row in iter_1kg_rows(superpop_freqs_path):
        n_rows += 1
        if n_rows % log_every == 0:
            log.info(
                "scanned %d sites; candidates per pop=%s",
                n_rows,
                {p: len(candidates[p]) for p in SUPERPOPS_AOU},
            )
        for pop in SUPERPOPS_AOU:
            cand = _candidate_for_pop(row, pop, separation_threshold)
            if cand is not None:
                candidates[pop].append(cand)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for pop in SUPERPOPS_AOU:
        counts[pop] = len(candidates[pop])
        out_path = out_dir / f"{pop.lower()}_candidates.tsv"
        write_candidates(out_path, candidates[pop])
        log.info(
            "wrote %d %s candidates to %s",
            len(candidates[pop]), pop, out_path,
        )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--superpop-freqs", type=Path, default=None,
        help=(
            "Path to 1KG superpop freq TSV (default: resolve via "
            "popout.fetch_superpop_freqs.resolve_superpop_freqs_path)"
        ),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("build/aim_panels"),
        help="Output directory for candidate TSVs",
    )
    parser.add_argument(
        "--separation-threshold", type=float, default=0.40,
        help="Minimum target-vs-max-others freq separation",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    superpop_freqs_path = (
        args.superpop_freqs if args.superpop_freqs is not None
        else resolve_superpop_freqs_path()
    )
    counts = scan(
        superpop_freqs_path=superpop_freqs_path,
        out_dir=args.out_dir,
        separation_threshold=args.separation_threshold,
    )
    print("Per-population candidate counts:")
    for pop, n in counts.items():
        print(f"  {pop}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
