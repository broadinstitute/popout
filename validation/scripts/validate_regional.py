#!/usr/bin/env python3
"""§8.3 Regional QC: per-window ancestry deviation scan.

Streams a tracts file, computes per-(sliding-window, ancestry) mean
ancestry proportion across all haplotypes, z-scores against the chrom
mean, applies Benjamini-Hochberg FDR across (window, ancestry) tests,
and emits a manhattan-style figure annotated with optional region-mask
overlays (centromere, segdup, high-LD, HLA).

Usage:
    python validate_regional.py \\
        --tracts PATH/<prefix>.tracts.tsv.gz \\
        --chrom-sizes diagnostics/data/grch38.chrom.sizes \\
        --window-bp 1000000 --step-bp 250000 \\
        --region-mask-bed PATH/centromere.bed \\
        --region-mask-bed PATH/segdup.bed \\
        --out-dir PATH/diagnostics
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".." / "popout"))
from popout.viz._loaders import read_tracts


def load_chrom_sizes(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chrom, size = line.split("\t")
            out[chrom] = int(size)
    return out


def load_region_bed(path: Path) -> list[tuple[str, int, int, str]]:
    out: list[tuple[str, int, int, str]] = []
    name = path.stem
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("track"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                raise ValueError(f"BED line too short: {line!r}")
            chrom, start, end = parts[0], int(parts[1]), int(parts[2])
            rname = parts[3] if len(parts) >= 4 else name
            out.append((chrom, start, end, rname))
    return out


def make_windows(chrom_len: int, window_bp: int, step_bp: int) -> list[tuple[int, int]]:
    windows = []
    s = 0
    while s < chrom_len:
        e = min(s + window_bp, chrom_len)
        windows.append((s, e))
        if e == chrom_len:
            break
        s += step_bp
    return windows


def accumulate_window_bp(
    tracts_path: Path,
    *,
    chrom_sizes: dict[str, int],
    window_bp: int,
    step_bp: int,
) -> tuple[dict[str, list[tuple[int, int]]],
           dict[str, np.ndarray],
           dict[str, int]]:
    """For each (chrom, window, ancestry) accumulate intersected bp.

    Returns:
      windows_by_chrom: chrom -> [(start, end), ...]
      bp_by_chrom: chrom -> array[n_windows, n_ancestries]
      n_hap_by_chrom: chrom -> number of distinct (sample, hap) seen
    """
    # First pass: discover ancestries actually present so we don't hard-code K.
    ancestries: set[int] = set()
    n_hap: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for t in read_tracts(tracts_path):
        ancestries.add(t.ancestry)
        n_hap[t.chrom].add((t.sample, t.haplotype))
    K = max(ancestries) + 1
    if K == 0:
        raise RuntimeError("No ancestries found in tracts")
    print(f"  ancestries: {sorted(ancestries)} (K_max+1 = {K})")

    windows_by_chrom: dict[str, list[tuple[int, int]]] = {}
    bp_by_chrom: dict[str, np.ndarray] = {}
    n_hap_by_chrom: dict[str, int] = {}
    for chrom in n_hap:
        if chrom not in chrom_sizes:
            raise RuntimeError(f"chrom {chrom} not in chrom-sizes file")
        wins = make_windows(chrom_sizes[chrom], window_bp, step_bp)
        windows_by_chrom[chrom] = wins
        bp_by_chrom[chrom] = np.zeros((len(wins), K), dtype=np.int64)
        n_hap_by_chrom[chrom] = len(n_hap[chrom])
    print("  haps per chrom:")
    for c, n in n_hap_by_chrom.items():
        print(f"    {c}: {n}")

    # Second pass: stream and intersect each tract with overlapping windows.
    n_records = 0
    for t in read_tracts(tracts_path):
        wins = windows_by_chrom[t.chrom]
        bp_mat = bp_by_chrom[t.chrom]
        # Window step is regular; can binary-search the first overlapping window.
        # Windows are [start, end). Tract is [start_bp, end_bp] (inclusive).
        ts, te = t.start_bp, t.end_bp + 1  # half-open for math
        # Find first window with end > ts.
        # Linear scan is fine; with 1 Mb / 250 kb step on chr1 there are ~1000 windows.
        for wi, (ws, we) in enumerate(wins):
            if we <= ts:
                continue
            if ws >= te:
                break
            ov = min(te, we) - max(ts, ws)
            if ov > 0:
                bp_mat[wi, t.ancestry] += ov
        n_records += 1
    print(f"  intersected {n_records} tracts with windows")
    return windows_by_chrom, bp_by_chrom, n_hap_by_chrom


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracts", type=Path, required=True)
    p.add_argument("--chrom-sizes", type=Path, required=True)
    p.add_argument("--window-bp", type=int, default=1_000_000)
    p.add_argument("--step-bp", type=int, default=250_000)
    p.add_argument("--region-mask-bed", type=Path, action="append", default=[],
                   help="BED of named regions to overlay (repeatable)")
    p.add_argument("--fdr-q", type=float, default=0.05)
    p.add_argument("--labels-json", type=Path, default=None,
                   help="labels.json from compare_to_rf.py. When provided, "
                        "ancestry indices in plots and tables are replaced "
                        "with the RF reference label names (afr/amr/eas/...). "
                        "Without it, plots fall back to 'ancestry 0..K-1'.")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve ancestry index → display name. labels.json's
    # popout_to_rf_label maps each ancestry column (str index) to its RF
    # label. If multiple ancestries share a label (popout sub-continental),
    # disambiguate with the index suffix (e.g. afr.0, afr.1).
    anc_name: dict[int, str] = {}
    if args.labels_json is not None:
        import json as _json
        lj = _json.loads(args.labels_json.read_text())
        ptr = lj.get("popout_to_rf_label", {})
        if not ptr:
            raise RuntimeError(f"labels.json {args.labels_json} missing popout_to_rf_label key")
        raw = {int(k): v for k, v in ptr.items()}
        counts: dict[str, int] = {}
        for v in raw.values():
            counts[v] = counts.get(v, 0) + 1
        for i, lbl in raw.items():
            anc_name[i] = f"{lbl}.{i}" if counts[lbl] > 1 else lbl

    if args.step_bp > args.window_bp:
        raise ValueError("--step-bp must be <= --window-bp")
    if not args.tracts.exists():
        raise FileNotFoundError(args.tracts)
    if not args.chrom_sizes.exists():
        raise FileNotFoundError(args.chrom_sizes)

    print("Loading chrom sizes...")
    chrom_sizes = load_chrom_sizes(args.chrom_sizes)

    print("Loading region masks...")
    masks: list[tuple[str, int, int, str]] = []
    for bed in args.region_mask_bed:
        masks.extend(load_region_bed(bed))
    print(f"  {len(masks)} mask intervals across {len(args.region_mask_bed)} BEDs")

    print("Accumulating per-window bp counts...")
    windows_by_chrom, bp_by_chrom, n_hap_by_chrom = accumulate_window_bp(
        args.tracts, chrom_sizes=chrom_sizes,
        window_bp=args.window_bp, step_bp=args.step_bp,
    )

    # Per-(window, ancestry) mean proportion = bp / (window_size * n_haps).
    rows: list[dict] = []
    for chrom, wins in windows_by_chrom.items():
        bp_mat = bp_by_chrom[chrom]
        n_haps = n_hap_by_chrom[chrom]
        # window_lens used to normalize
        win_lens = np.array([(e - s) for s, e in wins], dtype=np.float64)
        denom = win_lens[:, None] * n_haps  # broadcast over K
        mean_anc = bp_mat / denom  # (n_windows, K)
        K = mean_anc.shape[1]

        # Per-ancestry chrom mean and SD across windows.
        chrom_mean = mean_anc.mean(axis=0)  # (K,)
        chrom_sd = mean_anc.std(axis=0, ddof=1)  # (K,)

        # Z and two-sided p for each (window, ancestry).
        for wi, (ws, we) in enumerate(wins):
            for a in range(K):
                if chrom_sd[a] == 0 or np.isnan(chrom_sd[a]):
                    z = 0.0
                    p_raw = 1.0
                else:
                    z = float((mean_anc[wi, a] - chrom_mean[a]) / chrom_sd[a])
                    p_raw = float(2.0 * norm.sf(abs(z)))
                # Region overlap.
                mask_names = []
                for m_chrom, m_s, m_e, m_name in masks:
                    if m_chrom != chrom:
                        continue
                    if min(m_e, we) > max(m_s, ws):
                        mask_names.append(m_name)
                rows.append({
                    "chrom": chrom,
                    "start": ws,
                    "end": we,
                    "ancestry": a,
                    "mean_anc": float(mean_anc[wi, a]),
                    "z": z,
                    "p": p_raw,
                    "mask_region": ",".join(sorted(set(mask_names))) if mask_names else "",
                })

    # BH-FDR over all rows.
    p_arr = np.array([r["p"] for r in rows])
    _, q_arr, _, _ = multipletests(p_arr, alpha=args.fdr_q, method="fdr_bh")
    for r, q in zip(rows, q_arr):
        r["q"] = float(q)

    # Attach display name to every row (uses RF reference label when
    # --labels-json was supplied; falls back to "ancestry <i>").
    def _name(idx: int) -> str:
        return anc_name.get(idx, f"ancestry {idx}")
    for r in rows:
        r["ancestry_name"] = _name(r["ancestry"])

    # Write per-window TSV (gzipped). Numeric formatting is fixed-width
    # so the table is readable in a PDF: mean_anc 4 sf, z ±.2f, p/q 1e
    # scientific.
    out_tsv = args.out_dir / "regional_windows.tsv.gz"
    cols = ["chrom", "start", "end", "ancestry_name", "mean_anc", "z", "p", "q", "mask_region"]
    with gzip.open(out_tsv, "wt") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write(
                f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['ancestry_name']}\t"
                f"{r['mean_anc']:.4f}\t{r['z']:+.2f}\t{r['p']:.2e}\t{r['q']:.2e}\t"
                f"{r['mask_region']}\n"
            )
    print(f"  wrote {out_tsv}  ({len(rows)} rows)")

    sig = [r for r in rows if r["q"] < args.fdr_q]
    out_bed = args.out_dir / "regional_significant.bed"
    with open(out_bed, "w") as f:
        f.write("#chrom\tstart\tend\tname\n")
        for r in sig:
            name = f"{r['ancestry_name']}|z{r['z']:+.2f}|q{r['q']:.1e}"
            if r["mask_region"]:
                name += f"|{r['mask_region']}"
            f.write(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{name}\n")
    print(f"  wrote {out_bed}  ({len(sig)} significant windows at q<{args.fdr_q})")

    # ── summary.json (schema §1.11) ──
    # Mask-overlap counts use case-insensitive substring matching on the
    # mask-region label so "HLA", "hla_class_I", "centromere", etc. all
    # land in the right bucket. A window can overlap multiple masks; we
    # count it once per bucket it matches.
    sig_ancestries = sorted({r["ancestry"] for r in rows})
    per_ancestry_summary = []
    for a in sig_ancestries:
        a_sig = [r for r in sig if r["ancestry"] == a]
        peak = None
        if a_sig:
            peak_row = max(a_sig, key=lambda r: abs(r["z"]))
            peak = {
                "chrom": peak_row["chrom"],
                "start": int(peak_row["start"]),
                "end": int(peak_row["end"]),
                "z": float(peak_row["z"]),
                "q": float(peak_row["q"]),
                "mask_region": peak_row["mask_region"],
            }
        per_ancestry_summary.append({
            "ancestry": int(a),
            "name": _name(a),
            "n_significant": len(a_sig),
            "peak_window": peak,
        })

    def _mask_count(token: str) -> int:
        token = token.lower()
        return sum(1 for r in sig if token in r["mask_region"].lower())

    outside_mask_n = sum(1 for r in sig if not r["mask_region"])

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "n_windows_total": len(rows),
        "n_windows_significant": len(sig),
        "fdr_q_threshold": float(args.fdr_q),
        "per_ancestry": per_ancestry_summary,
        "hla_overlap_n": _mask_count("hla"),
        "centromere_overlap_n": _mask_count("centromere"),
        "segdup_overlap_n": _mask_count("segdup"),
        "high_ld_overlap_n": _mask_count("high"),
        "outside_mask_n": outside_mask_n,
    }, indent=2))
    print(f"  wrote {summary_path}")

    # Manhattan-style plot per chrom.
    for chrom, wins in windows_by_chrom.items():
        chrom_rows = [r for r in rows if r["chrom"] == chrom]
        K = max(r["ancestry"] for r in chrom_rows) + 1
        fig, axs = plt.subplots(K, 1, figsize=(11, 1.5 * K + 0.5), sharex=True, squeeze=False)
        for a in range(K):
            ax = axs[a, 0]
            ar = [r for r in chrom_rows if r["ancestry"] == a]
            xs = np.array([(r["start"] + r["end"]) / 2 / 1e6 for r in ar])
            ys = np.array([-np.log10(max(r["p"], 1e-300)) for r in ar])
            sig_mask = np.array([r["q"] < args.fdr_q for r in ar])
            ax.scatter(xs[~sig_mask], ys[~sig_mask], s=8, color="C0", alpha=0.6,
                       label="n.s.")
            ax.scatter(xs[sig_mask], ys[sig_mask], s=12, color="red",
                       label=f"q<{args.fdr_q}")
            # Shade mask regions.
            for m_chrom, m_s, m_e, m_name in masks:
                if m_chrom != chrom:
                    continue
                ax.axvspan(m_s / 1e6, m_e / 1e6, color="gray", alpha=0.15)
            ax.set_ylabel(f"{_name(a)}\n-log10 p")
            if a == 0:
                ax.legend(fontsize=7, loc="upper right")
            if a == K - 1:
                ax.set_xlabel(f"{chrom} position (Mb)")
        fig.suptitle(
            f"§8.3 Regional QC — {chrom}  "
            f"(window={args.window_bp/1e6:g} Mb step={args.step_bp/1e6:g} Mb)\n"
            f"Each panel: per-window mean ancestry proportion vs the chromosome mean. "
            f"Y = -log10(p) of the z-test; red = FDR-significant (q<{args.fdr_q}).",
            y=0.995, fontsize=10,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_png = args.out_dir / f"regional_qc_{chrom}.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_png}")


if __name__ == "__main__":
    main()
