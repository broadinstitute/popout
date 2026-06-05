#!/usr/bin/env python3
"""Convert a FLARE chr output directory to popout-style diagnostics inputs.

FLARE writes:
  <prefix>.global.anc.gz   gzipped TSV: SAMPLE eas amr eur afr sas
  <prefix>.anc.vcf.gz       per-site ancestry VCF
  <prefix>.model            text model: T, mu, p[i][j], theta[i][j]
  <prefix>.log              human-readable run log
  <prefix>.qc.tsv           record counts in/out

Popout's downstream diagnostics expect a per-run "prefix" directory laid out
roughly as ``<data>/<prefix>.global.tsv``, ``<prefix>.model``, and (optionally)
``<prefix>.summary.json``.  This script rewrites the FLARE outputs into that
shape inside ``<out-dir>`` so ``compare_to_rf.py``, ``plot_concordance.py``
and ``build_report_pdf.py`` can consume them with no further changes.

Usage:
    python flare_to_popout_format.py \\
        --flare-prefix /path/to/null_cluster_sample_list.chr1 \\
        --out-dir /path/to/flare/v1_nc \\
        --run-prefix null_cluster_sample_list.chr1
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path


def parse_flare_model(model_path: Path) -> dict:
    """Read a FLARE .model file.

    Lines look like:
        # comment
        eas\\tamr\\teur\\tafr\\tsas        (ancestries)
        eas\\tamr\\teur\\tafr\\tsas        (reference panels)
        5.16                          (T, gen since admixture)
        0.0015\\t0.014\\t0.639\\t0.323\\t0.020   (mu)
        <p matrix, K x P>
        <theta matrix, K x P>
    """
    lines = [ln.strip() for ln in model_path.read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    ancestries = lines[0].split("\t")
    ref_panels = lines[1].split("\t")
    T = float(lines[2])
    mu = [float(x) for x in lines[3].split("\t")]
    K = len(ancestries)
    p_rows = [list(map(float, lines[4 + i].split("\t"))) for i in range(K)]
    theta_rows = [list(map(float, lines[4 + K + i].split("\t"))) for i in range(K)]
    return {
        "ancestries": ancestries,
        "ref_panels": ref_panels,
        "T": T,
        "mu": mu,
        "p": p_rows,
        "theta": theta_rows,
    }


def parse_flare_log(log_path: Path) -> dict:
    """Extract config + summary stats from a FLARE log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    cfg: dict = {"raw_params": {}}

    # Parameters block ("  key : value")
    in_params = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "Parameters":
            in_params = True
            continue
        if in_params:
            m = re.match(r"^([A-Za-z0-9_\-]+)\s*:\s*(.+)$", stripped)
            if m:
                cfg["raw_params"][m.group(1)] = m.group(2).strip()
            elif stripped == "" or stripped.startswith("Statistics"):
                in_params = False
                break

    m = re.search(r"reference samples\s*:\s*(\d+)", text)
    if m:
        cfg["reference_samples"] = int(m.group(1))
    m = re.search(r"target samples\s*:\s*(\d+)", text)
    if m:
        cfg["target_samples"] = int(m.group(1))
    m = re.search(r"markers\s*:\s*(\d+)", text)
    if m:
        cfg["markers"] = int(m.group(1))

    m = re.search(r"Wallclock Time\s*:\s*(.+)$", text, re.MULTILINE)
    if m:
        cfg["wallclock_str"] = m.group(1).strip()

    m = re.search(r"Program\s*:\s*(.+)$", text, re.MULTILINE)
    if m:
        cfg["program"] = m.group(1).strip()

    return cfg


def parse_wallclock(s: str) -> float:
    """Parse a string like '1 hour 21 minutes 11 seconds' to total seconds."""
    total = 0
    m = re.search(r"(\d+)\s*hour", s)
    if m: total += int(m.group(1)) * 3600
    m = re.search(r"(\d+)\s*minute", s)
    if m: total += int(m.group(1)) * 60
    m = re.search(r"(\d+)\s*second", s)
    if m: total += int(m.group(1))
    return float(total)


def write_popout_global(flare_global: Path, out_path: Path) -> tuple[list[str], int]:
    """Rewrite FLARE's global.anc.gz as a plain popout-style global.tsv.

    **Schema v3.0.0:** the FLARE panel-population names declared in the
    ``##ANCESTRY=`` VCF header (already echoed into the ``global.anc.gz``
    header as ``SAMPLE<TAB>eas<TAB>amr<TAB>...``) are preserved verbatim.
    Pre-v3 bundles used anonymous ``ancestry_0..K-1`` columns and had to
    re-derive the names downstream via posterior correlation —
    ``popout.labelspace.matching.posterior_slope`` (``postS``) — which
    invented fake subancestries like ``afr.1, afr.2`` when two FLARE
    components correlated strongest with the same RF label.

    ``popout.estimates.loaders.read_flare_aggregated`` accepts both the
    anonymous (legacy) and named (v3) shapes during cutover.
    """
    with gzip.open(flare_global, "rt") as f, open(out_path, "w") as g:
        header = f.readline().strip().split("\t")
        ancestries = header[1:]
        g.write("sample_id\t" + "\t".join(ancestries) + "\n")
        n = 0
        for line in f:
            g.write(line)
            n += 1
    return ancestries, n


def write_popout_model(model: dict, out_path: Path) -> None:
    """Emit a popout-style .model text file (n_ancestries, gen_since_admix, mu)."""
    K = len(model["ancestries"])
    mu_str = ",".join(f"{x:.4f}" for x in model["mu"])
    out_path.write_text(
        f"n_ancestries\t{K}\n"
        f"gen_since_admix\t{model['T']:.2f}\n"
        f"mu\t{mu_str}\n"
    )


def write_popout_summary(model: dict, log: dict, out_path: Path) -> None:
    """Emit a minimal popout-style summary.json that build_report_pdf
    can consume for the Run Configuration panel."""
    cfg = log.get("raw_params", {})
    wall = parse_wallclock(log.get("wallclock_str", "")) if log.get("wallclock_str") else 0.0
    K = len(model["ancestries"])
    summary = {
        "popout_version": f"FLARE  {log.get('program', '?')}",
        "config": {
            "method": "flare",
            "n_ancestries": K,
            "n_em_iter": "EM" if cfg.get("em", "false") == "true" else "false",
            "gen_since_admix": cfg.get("gen", "?"),
            "min_maf": cfg.get("min-maf", "?"),
            "min_mac": cfg.get("min-mac", "?"),
            "seed": cfg.get("seed", "?"),
        },
        "total_wall_clock_sec": wall,
        "site_filter_funnel": {
            "1": {
                "sites_biallelic": "?",
                "sites_after_thinning": log.get("markers", "?"),
                "sites_final": log.get("markers", "?"),
                "genetic_length_cm": "?",
            }
        },
        "runtime": {
            "device_info": {
                "platform": "cpu",
                "device_count": 1,
                "devices": [{"kind": cfg.get("nthreads", "?") + " threads", "id": 0}],
                "python_platform": "FLARE (java)",
            }
        },
        "final_model": {
            "mu": model["mu"],
            "T": model["T"],
            "ancestry_proportions": {str(i): m for i, m in enumerate(model["mu"])},
        },
        "flare_model": {
            "ancestries": model["ancestries"],
            "ref_panels": model["ref_panels"],
            "p": model["p"],
            "theta": model["theta"],
        },
        "flare_log": {
            "reference_samples": log.get("reference_samples"),
            "target_samples": log.get("target_samples"),
            "markers": log.get("markers"),
            "raw_params": cfg,
        },
    }
    out_path.write_text(json.dumps(summary, indent=2))


def write_stdout_shim(log: dict, out_path: Path) -> None:
    """Synthesize a minimal stdout for the PDF Run-Configuration panel.

    The PDF builder scans stdout for ``=== Running: ... ===``.  Reconstruct
    a flare command line from the parsed --params block.
    """
    raw = log.get("raw_params", {})
    if not raw:
        return
    parts = ["flare"]
    for k, v in raw.items():
        parts.append(f"--{k}={v}")
    cmd = " ".join(parts)
    out_path.write_text(f"=== Running: {cmd} ===\n"
                        f"Program: {log.get('program', '?')}\n"
                        f"Reference samples: {log.get('reference_samples', '?')}\n"
                        f"Target samples: {log.get('target_samples', '?')}\n"
                        f"Markers: {log.get('markers', '?')}\n"
                        f"Wallclock: {log.get('wallclock_str', '?')}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Adapt FLARE outputs to popout-style diagnostic inputs.")
    parser.add_argument("--flare-prefix", type=Path, required=True,
                        help="Path stem of FLARE outputs (e.g. /path/to/null_cluster_sample_list.chr1)")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Where to place the popout-style files. Created if missing.")
    parser.add_argument("--run-prefix", type=str, default=None,
                        help="Output prefix for the popout-style files. Defaults to the FLARE stem name.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = args.run_prefix or args.flare_prefix.name

    g_path = args.flare_prefix.parent / f"{args.flare_prefix.name}.global.anc.gz"
    m_path = args.flare_prefix.parent / f"{args.flare_prefix.name}.model"
    l_path = args.flare_prefix.parent / f"{args.flare_prefix.name}.log"
    if not g_path.exists():
        raise FileNotFoundError(f"FLARE global file missing: {g_path}")
    if not m_path.exists():
        raise FileNotFoundError(f"FLARE model file missing: {m_path}")

    print(f"Reading FLARE outputs from {args.flare_prefix.parent}/")
    model = parse_flare_model(m_path)
    log = parse_flare_log(l_path)
    print(f"  ancestries: {model['ancestries']}")
    print(f"  T: {model['T']:.3f}")
    print(f"  mu: {model['mu']}")
    if log:
        print(f"  target_samples: {log.get('target_samples', '?')}")
        print(f"  reference_samples: {log.get('reference_samples', '?')}")
        print(f"  markers: {log.get('markers', '?')}")

    out_global = args.out_dir / f"{run_prefix}.global.tsv"
    out_model = args.out_dir / f"{run_prefix}.model"
    out_summary = args.out_dir / f"{run_prefix}.summary.json"
    out_stdout = args.out_dir / "stdout"

    ancestries, n_samples = write_popout_global(g_path, out_global)
    write_popout_model(model, out_model)
    write_popout_summary(model, log, out_summary)
    write_stdout_shim(log, out_stdout)

    # Persist a tiny manifest with the FLARE→index mapping so analysis
    # scripts can reference ancestry NAMES later if they want to.
    manifest = args.out_dir / f"{run_prefix}.flare_manifest.json"
    manifest.write_text(json.dumps({
        "ancestries": ancestries,
        "n_samples": n_samples,
        "flare_prefix": str(args.flare_prefix),
    }, indent=2))

    print(f"\nWrote {out_global} ({n_samples} samples, {len(ancestries)} ancestries)")
    print(f"Wrote {out_model}")
    print(f"Wrote {out_summary}")
    print(f"Wrote {out_stdout}")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
