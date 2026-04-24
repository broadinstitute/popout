"""Reference panel extraction from popout posteriors.

Extracts high-confidence single-ancestry haplotypes and segments from
the posterior ancestry probabilities, producing reference panels suitable
for downstream LAI tools (FLARE, RFMix, LAMP-LD).

Activated by ``--export-panel`` on the CLI.
"""

from __future__ import annotations

import gzip
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .datatypes import AncestryResult, ChromData

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PanelConfig:
    """Configuration for panel extraction."""

    whole_hap_threshold: float = 0.95
    segment_threshold: float = 0.99
    min_segment_cm: float = 1.0
    min_segment_sites: int = 50
    max_per_ancestry: int | None = None


@dataclass
class WholeHapExtraction:
    """Result of whole-haplotype extraction."""

    hap_indices: np.ndarray       # (N,) int — indices of passing haplotypes
    ancestry_labels: np.ndarray   # (N,) int — assigned ancestry per haplotype
    min_posteriors: np.ndarray    # (N,) float — genome-wide min posterior
    mean_posteriors: np.ndarray   # (N,) float — genome-wide mean of max posterior


@dataclass
class Segment:
    """A single extracted ancestry segment."""

    hap_index: int
    chrom: str
    start_site: int
    end_site: int       # inclusive
    start_bp: int
    end_bp: int
    start_cm: float
    end_cm: float
    ancestry: int
    mean_posterior: float
    n_sites: int


# ---------------------------------------------------------------------------
# Extraction logic
# ---------------------------------------------------------------------------

def extract_whole_haplotypes(
    results: list[AncestryResult],
    threshold: float = 0.95,
) -> WholeHapExtraction:
    """Identify haplotypes that are single-ancestry genome-wide.

    For each haplotype *h*, computes ``min_t max_a gamma[h, t, a]`` across
    all chromosomes. Haplotypes whose genome-wide minimum exceeds *threshold*
    are classified as single-ancestry.

    Parameters
    ----------
    results : list of AncestryResult (one per chromosome)
    threshold : minimum posterior to classify as single-ancestry
    """
    if not results:
        return WholeHapExtraction(
            hap_indices=np.array([], dtype=np.intp),
            ancestry_labels=np.array([], dtype=np.int8),
            min_posteriors=np.array([], dtype=np.float32),
            mean_posteriors=np.array([], dtype=np.float32),
        )

    n_haps = results[0].calls.shape[0]

    # Accumulate per-hap genome-wide min-of-max and mean-of-max posteriors
    genome_min = np.ones(n_haps, dtype=np.float64)
    genome_mean_sum = np.zeros(n_haps, dtype=np.float64)
    total_sites = 0
    # Accumulate per-ancestry site counts for mode computation
    A = results[0].model.n_ancestries
    ancestry_site_counts = np.zeros((n_haps, A), dtype=np.int64)
    # Track whether any ancestry switch occurs
    prev_calls: Optional[np.ndarray] = None
    has_switch = np.zeros(n_haps, dtype=bool)

    _CHUNK = 50_000
    for result in results:
        hard_calls = np.asarray(result.calls)  # zero-copy

        # Compute per-hap min and sum of max-posterior
        if result.decode is not None and result.decode.max_post is not None:
            max_post = result.decode.max_post
            chrom_min = max_post.min(axis=1)
            genome_min = np.minimum(genome_min, chrom_min)
            genome_mean_sum += max_post.sum(axis=1, dtype=np.float64)
        elif (result.decode is not None
              and result.decode.parquet_path is not None):
            from .output import _iter_max_post_groups
            for rg_start, rg_end, mp_chunk in _iter_max_post_groups(
                result.decode.parquet_path,
            ):
                chunk_min = mp_chunk.min(axis=1)
                genome_min[rg_start:rg_end] = np.minimum(
                    genome_min[rg_start:rg_end], chunk_min,
                )
                genome_mean_sum[rg_start:rg_end] += mp_chunk.sum(
                    axis=1, dtype=np.float64,
                )
        elif result.posteriors is not None:
            gamma = np.asarray(result.posteriors)
            max_post = gamma.max(axis=2)
            hard_calls = gamma.argmax(axis=2)
            chrom_min = max_post.min(axis=1)
            genome_min = np.minimum(genome_min, chrom_min)
            genome_mean_sum += max_post.sum(axis=1, dtype=np.float64)
        # else: no posterior info — genome_min stays at 1.0

        total_sites += hard_calls.shape[1]

        # Accumulate ancestry counts for mode (chunked to avoid H×T int64)
        H_res = hard_calls.shape[0]
        for start in range(0, H_res, _CHUNK):
            end = min(start + _CHUNK, H_res)
            chunk = hard_calls[start:end]
            for a in range(A):
                ancestry_site_counts[start:end, a] += (chunk == a).sum(axis=1)

        # Check for ancestry switches within this chromosome
        if hard_calls.shape[1] > 1:
            has_switch |= np.any(hard_calls[:, 1:] != hard_calls[:, :-1], axis=1)

        # Check for ancestry switches between chromosomes
        if prev_calls is not None:
            has_switch |= (prev_calls != hard_calls[:, 0])
        prev_calls = hard_calls[:, -1]

    genome_mean = genome_mean_sum / max(total_sites, 1)

    # Filter: pass threshold AND no ancestry switches
    passing = (genome_min > threshold) & ~has_switch
    hap_indices = np.where(passing)[0]

    # Assign ancestry = mode across all sites (constant for passing haps)
    ancestry_labels = ancestry_site_counts[hap_indices].argmax(axis=1).astype(np.int8)
    min_posteriors = genome_min[hap_indices].astype(np.float32)
    mean_posteriors = genome_mean[hap_indices].astype(np.float32)

    log.info(
        "Whole-haplotype extraction: %d / %d haplotypes pass threshold %.3f",
        len(hap_indices), n_haps, threshold,
    )

    return WholeHapExtraction(
        hap_indices=hap_indices,
        ancestry_labels=ancestry_labels,
        min_posteriors=min_posteriors,
        mean_posteriors=mean_posteriors,
    )


def extract_segments(
    result: AncestryResult,
    cdata: ChromData,
    threshold: float = 0.99,
    min_cm: float = 1.0,
    min_sites: int = 50,
) -> list[Segment]:
    """Extract contiguous single-ancestry segments from one chromosome.

    For each haplotype, finds maximal contiguous runs where the dominant
    ancestry posterior exceeds *threshold* at every site, the dominant
    ancestry is constant, and the segment meets minimum length requirements.

    Parameters
    ----------
    result : AncestryResult for one chromosome
    cdata : ChromData for the same chromosome
    threshold : minimum per-site posterior for confidence
    min_cm : minimum segment genetic length in centiMorgans
    min_sites : minimum number of sites in a segment
    """
    hard_calls = np.asarray(result.calls)  # zero-copy
    gamma = np.asarray(result.posteriors) if result.posteriors is not None else None

    H, T = hard_calls.shape

    if T == 0:
        return []

    pos_bp = cdata.pos_bp
    pos_cm = cdata.pos_cm
    chrom = cdata.chrom

    segments: list[Segment] = []

    # Determine max_post source
    if result.decode is not None and result.decode.max_post is not None:
        max_post = result.decode.max_post
        _extract_segments_range(
            0, H, hard_calls, max_post, gamma, threshold,
            min_cm, min_sites, pos_bp, pos_cm, chrom, segments,
        )
    elif (result.decode is not None
          and result.decode.parquet_path is not None):
        from .output import _iter_max_post_groups
        for rg_start, rg_end, mp_chunk in _iter_max_post_groups(
            result.decode.parquet_path,
        ):
            _extract_segments_range(
                rg_start, rg_end, hard_calls, mp_chunk, None, threshold,
                min_cm, min_sites, pos_bp, pos_cm, chrom, segments,
            )
    elif gamma is not None:
        max_post = gamma.max(axis=2)
        hard_calls = gamma.argmax(axis=2)
        _extract_segments_range(
            0, H, hard_calls, max_post, gamma, threshold,
            min_cm, min_sites, pos_bp, pos_cm, chrom, segments,
        )
    else:
        _extract_segments_range(
            0, H, hard_calls, None, None, threshold,
            min_cm, min_sites, pos_bp, pos_cm, chrom, segments,
        )

    return segments


def _extract_segments_range(
    hi_start: int,
    hi_end: int,
    hard_calls: np.ndarray,
    max_post: np.ndarray | None,
    gamma: np.ndarray | None,
    threshold: float,
    min_cm: float,
    min_sites: int,
    pos_bp: np.ndarray,
    pos_cm: np.ndarray,
    chrom: str,
    segments: list[Segment],
) -> None:
    """Extract segments for haplotypes [hi_start, hi_end).

    *max_post*, when provided, covers rows ``[0, hi_end - hi_start)``
    (it may be a row-group chunk), so we index with ``hi - hi_start``.
    """
    T = hard_calls.shape[1]

    for hi in range(hi_start, hi_end):
        hap_calls = hard_calls[hi]

        if max_post is not None:
            hap_confident = max_post[hi - hi_start] > threshold
            labels = np.where(hap_confident, hap_calls, np.int8(-1))
        else:
            labels = hap_calls

        changes = np.where(labels[1:] != labels[:-1])[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [T]])

        for k in range(len(starts)):
            s, e = int(starts[k]), int(ends[k])
            anc = int(labels[s])
            if anc < 0:
                continue

            n_seg_sites = e - s
            if n_seg_sites < min_sites:
                continue

            seg_cm = float(pos_cm[e - 1] - pos_cm[s])
            if seg_cm < min_cm:
                continue

            if gamma is not None:
                mean_post = float(gamma[hi, s:e, anc].mean())
            elif max_post is not None:
                mean_post = float(max_post[hi - hi_start, s:e].mean())
            else:
                mean_post = 1.0
            segments.append(Segment(
                hap_index=hi,
                chrom=chrom,
                start_site=s,
                end_site=e - 1,
                start_bp=int(pos_bp[s]),
                end_bp=int(pos_bp[e - 1]),
                start_cm=float(pos_cm[s]),
                end_cm=float(pos_cm[e - 1]),
                ancestry=anc,
                mean_posterior=mean_post,
                n_sites=n_seg_sites,
            ))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_panel_haplotypes(
    sample_names: list[str],
    extraction: WholeHapExtraction,
    out_path: str,
    stats=None,
) -> None:
    """Write whole-haplotype ancestry assignments to TSV."""
    with open(out_path, "w") as f:
        f.write("sample_id\thaplotype\tancestry\tmin_posterior\tmean_posterior\n")
        for i in range(len(extraction.hap_indices)):
            hi = int(extraction.hap_indices[i])
            si = hi // 2
            hap = hi % 2
            f.write(
                f"{sample_names[si]}\t{hap}\t{int(extraction.ancestry_labels[i])}"
                f"\t{extraction.min_posteriors[i]:.4f}"
                f"\t{extraction.mean_posteriors[i]:.4f}\n"
            )
    log.info(
        "Wrote %d panel haplotypes to %s",
        len(extraction.hap_indices), out_path,
    )
    if stats is not None:
        stats.emit("panel/n_whole_haplotypes", len(extraction.hap_indices))


def write_panel_segments(
    segments: list[Segment],
    sample_names: list[str],
    out_path: str,
    stats=None,
) -> None:
    """Write segment-level ancestry extraction to gzipped BED-like TSV."""
    compress = out_path.endswith(".gz")
    opener = gzip.open if compress else open

    with opener(out_path, "wt") as f:
        f.write(
            "sample_id\thaplotype\tchrom\tstart_bp\tend_bp\t"
            "start_cm\tend_cm\tancestry\tmean_posterior\tn_sites\n"
        )
        for seg in segments:
            si = seg.hap_index // 2
            hap = seg.hap_index % 2
            f.write(
                f"{sample_names[si]}\t{hap}\t{seg.chrom}\t{seg.start_bp}\t{seg.end_bp}"
                f"\t{seg.start_cm:.4f}\t{seg.end_cm:.4f}"
                f"\t{seg.ancestry}\t{seg.mean_posterior:.4f}\t{seg.n_sites}\n"
            )
    log.info("Wrote %d panel segments to %s", len(segments), out_path)
    if stats is not None:
        stats.emit("panel/n_segments", len(segments))


def write_allele_frequencies(
    results: list[AncestryResult],
    chrom_data_list: list[ChromData],
    out_path: str,
    stats=None,
) -> None:
    """Write per-ancestry allele frequency table to gzipped TSV."""
    A = results[0].model.n_ancestries
    compress = out_path.endswith(".gz")
    opener = gzip.open if compress else open

    n_sites_total = 0
    with opener(out_path, "wt") as f:
        # Header
        anc_cols = "\t".join(f"ancestry_{a}_freq" for a in range(A))
        f.write(f"chrom\tpos\tsite_id\t{anc_cols}\n")

        for result, cdata in zip(results, chrom_data_list):
            freq = np.array(result.model.allele_freq)  # (A, T)
            T = freq.shape[1]
            n_sites_total += T
            has_ids = cdata.site_ids is not None

            for t in range(T):
                site_id = str(cdata.site_ids[t]) if has_ids else "."
                freq_vals = "\t".join(f"{freq[a, t]:.6f}" for a in range(A))
                f.write(f"{cdata.chrom}\t{cdata.pos_bp[t]}\t{site_id}\t{freq_vals}\n")

    log.info("Wrote allele frequencies (%d sites) to %s", n_sites_total, out_path)
    if stats is not None:
        stats.emit("panel/n_frequency_sites", n_sites_total)


def write_haplotype_proportions(
    results: list[AncestryResult],
    n_samples: int,
    sample_names: list[str],
    out_path: str,
    stats=None,
) -> None:
    """Write per-haplotype global ancestry proportions to TSV.

    Unlike ``write_global_ancestry`` (which averages over diploid pairs),
    this reports each haplotype individually — useful as soft labels for
    downstream reference panel construction.
    """
    A = results[0].model.n_ancestries
    n_haps = 2 * n_samples

    # Accumulate posteriors across chromosomes
    hap_sums = np.zeros((n_haps, A), dtype=np.float64)
    total_sites = 0

    for result in results:
        if result.decode is not None and result.decode.global_sums is not None:
            hap_sums += result.decode.global_sums
            total_sites += result.calls.shape[1]
        elif result.posteriors is not None:
            gamma = np.array(result.posteriors)  # (H, T, A)
            hap_sums += gamma.sum(axis=1)
            total_sites += gamma.shape[1]
        else:
            log.warning("No posteriors or decode for chrom %s, skipping", result.chrom)
            continue

    hap_props = hap_sums / max(total_sites, 1)

    with open(out_path, "w") as f:
        anc_cols = "\t".join(f"ancestry_{a}" for a in range(A))
        f.write(f"sample_id\thaplotype\t{anc_cols}\n")
        for hi in range(n_haps):
            si = hi // 2
            hap = hi % 2
            vals = "\t".join(f"{v:.4f}" for v in hap_props[hi])
            f.write(f"{sample_names[si]}\t{hap}\t{vals}\n")

    log.info("Wrote per-haplotype proportions to %s", out_path)
    if stats is not None:
        mean_props = hap_props.mean(axis=0).tolist()
        stats.emit("panel/haplotype_ancestry_proportions", mean_props)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def export_panel(
    results: list[AncestryResult],
    chrom_data_list: list[ChromData],
    n_samples: int,
    sample_names: list[str],
    out_prefix: str,
    config: PanelConfig,
    stats=None,
) -> None:
    """Run all panel extraction steps and write output files.

    Called from cli.py when ``--export-panel`` is set.
    """
    log.info("Extracting reference panel (threshold=%.2f, segment_threshold=%.2f)",
             config.whole_hap_threshold, config.segment_threshold)

    # Output 1a: Whole-haplotype extraction
    whole = extract_whole_haplotypes(results, threshold=config.whole_hap_threshold)
    write_panel_haplotypes(
        sample_names, whole,
        f"{out_prefix}.panel.haplotypes.tsv",
        stats=stats,
    )

    # Output 1b: Segment extraction
    all_segments: list[Segment] = []
    for result, cdata in zip(results, chrom_data_list):
        segs = extract_segments(
            result, cdata,
            threshold=config.segment_threshold,
            min_cm=config.min_segment_cm,
            min_sites=config.min_segment_sites,
        )
        all_segments.extend(segs)
    write_panel_segments(
        all_segments, sample_names,
        f"{out_prefix}.panel.segments.tsv.gz",
        stats=stats,
    )

    # Output 2: Allele frequencies
    write_allele_frequencies(
        results, chrom_data_list,
        f"{out_prefix}.panel.frequencies.tsv.gz",
        stats=stats,
    )

    # Output 3: Per-haplotype proportions
    write_haplotype_proportions(
        results, n_samples, sample_names,
        f"{out_prefix}.panel.proportions.tsv",
        stats=stats,
    )

    log.info("Panel export complete.")
