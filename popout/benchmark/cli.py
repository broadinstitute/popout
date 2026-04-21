"""CLI entry point for the LAI benchmark tool."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from popout.benchmark.parsers import get_parser
from popout.benchmark.report import build_report


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark LAI tool outputs against each other or ground truth."
    )
    parser.add_argument(
        "--tool",
        action="append",
        required=True,
        help="Tool output, format: 'name:path[:global_path]'. "
        "Can be specified multiple times.",
    )
    parser.add_argument(
        "--truth",
        help="Ground truth path (popout simulator .npz output)",
    )
    parser.add_argument(
        "--chrom",
        help="Chromosome to analyze (required for FLARE VCF filtering)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_report",
        help="Output directory (default: benchmark_report)",
    )
    parser.add_argument(
        "--label-map",
        help="CSV file with columns: tool,src_label,ref_label. "
        "Overrides automatic Hungarian matching.",
    )
    parser.add_argument(
        "--site-strategy",
        default="intersect",
        choices=["intersect", "project"],
        help="Site alignment strategy (default: intersect)",
    )

    args = parser.parse_args()

    # Parse label overrides
    label_overrides: dict[str, dict[int, int]] = {}
    if args.label_map:
        with open(args.label_map) as f:
            reader = csv.DictReader(f)
            for row in reader:
                tool = row["tool"]
                src = int(row["src_label"])
                ref = int(row["ref_label"])
                label_overrides.setdefault(tool, {})[src] = ref

    # Parse each tool output
    tracts = {}
    for tool_spec in args.tool:
        parts = tool_spec.split(":")
        if len(parts) < 2:
            parser.error(f"Invalid --tool format: {tool_spec!r}. Expected 'name:path[:global_path]'")
        name = parts[0]
        path = parts[1]
        global_path = parts[2] if len(parts) > 2 else None

        parse_fn = get_parser(name)
        kwargs = {}
        if name == "flare":
            kwargs["vcf_path"] = path
            if global_path:
                kwargs["global_path"] = global_path
            if args.chrom:
                kwargs["chrom"] = args.chrom
        elif name == "popout":
            kwargs["tracts_path"] = path
            if global_path:
                kwargs["global_path"] = global_path
        elif name == "truth":
            kwargs["truth_path"] = path
        else:
            # Generic: assume first positional is the main path
            kwargs["path"] = path

        tracts[name] = parse_fn(**kwargs)

    # Parse truth if provided separately
    truth = None
    if args.truth:
        from popout.benchmark.parsers.truth import parse_truth

        truth = parse_truth(args.truth)

    # Build report
    strategy = "project_a_onto_b" if args.site_strategy == "project" else "intersect"
    report_path = build_report(
        tracts=tracts,
        truth=truth,
        output_dir=args.output,
        site_strategy=strategy,
        label_overrides=label_overrides,
    )
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
