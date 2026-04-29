"""Build a per-population allele-frequency panel TSV from VCF source data.

Generates the reference frequency file used by ``popout label``. Supports
two source kinds via pluggable extractors:

- **genotype** — per-sample VCFs (e.g. 1000 Genomes Phase 3). Aggregates
  AC/AN across the samples that map to each output population.
- **info_af**  — sites-only VCFs with per-population ``AC_<pop>`` /
  ``AN_<pop>`` INFO fields (e.g. gnomAD v4).

Output (gzipped TSV):
    #chrom  pos  ref  alt  POP1  POP2  ...

Per-output-pop frequency is AC/AN-weighted across the source pops in that
rule:  AF_P = sum(AC_si) / sum(AN_si).

Usage::

    popout build-panel --source 1kg --download --out 1kg_superpop_freq.tsv.gz
    popout build-panel --source gnomad --vcf gnomad.chr22.vcf.bgz --out chr22.tsv.gz
    popout build-panel --source gnomad --vcf gnomad.chr22.vcf.bgz \\
        --pop-config configs/gnomad_5pop.json --out chr22.tsv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import shutil
import sys
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional

log = logging.getLogger(__name__)


# 1KG VCF download URLs per genome build
KG_VCF_URLS: dict[str, dict[str, str]] = {
    "GRCh37": {
        "base": "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502",
        "pattern": (
            "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b"
            ".20130502.genotypes.vcf.gz"
        ),
    },
    "GRCh38": {
        "base": (
            "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/"
            "1000G_2504_high_coverage/working/"
            "20220422_3202_phased_SNV_INDEL_SV"
        ),
        "pattern": (
            "1kGP_high_coverage_Illumina.chr{chrom}"
            ".filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        ),
    },
}

PANEL_URL = (
    "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/"
    "integrated_call_samples_v3.20130502.ALL.panel"
)


# ---------------------------------------------------------------------------
# Population configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PopConfig:
    """Population aggregation rules.

    output_order  -- column order of the output TSV.
    rules         -- output_pop -> tuple of source_pops to AC/AN-aggregate.
                     Pass-through is rules[pop] == (pop,).
    """

    output_order: tuple[str, ...]
    rules: dict[str, tuple[str, ...]]

    def source_pops(self) -> set[str]:
        return {sp for srcs in self.rules.values() for sp in srcs}


def _pop_config_from_dict(data: dict) -> PopConfig:
    if "output_order" not in data:
        raise ValueError("pop-config JSON must contain 'output_order'")
    if "rules" not in data:
        raise ValueError("pop-config JSON must contain 'rules'")
    order = tuple(data["output_order"])
    rules = {k: tuple(v) for k, v in data["rules"].items()}
    missing = [p for p in order if p not in rules]
    if missing:
        raise ValueError(
            f"output_order entries missing from rules: {missing}"
        )
    extra = [p for p in rules if p not in order]
    if extra:
        raise ValueError(
            f"rules contains entries not in output_order: {extra}"
        )
    return PopConfig(output_order=order, rules=rules)


def load_pop_config(path: str | Path) -> PopConfig:
    with open(path) as f:
        return _pop_config_from_dict(json.load(f))


def default_1kg_pop_config() -> PopConfig:
    text = (
        resources.files("popout.configs")
        .joinpath("1kg_superpops.json")
        .read_text()
    )
    return _pop_config_from_dict(json.loads(text))


def passthrough_pop_config(pops: list[str]) -> PopConfig:
    """Each source pop becomes its own output column (no collapsing)."""
    return PopConfig(
        output_order=tuple(pops),
        rules={p: (p,) for p in pops},
    )


# ---------------------------------------------------------------------------
# Source detection
# ---------------------------------------------------------------------------
GNOMAD_AF_RE = re.compile(r"^AF_([A-Za-z][A-Za-z0-9_]*)$")


def discover_gnomad_pops(vcf_header) -> list[str]:
    """Return the population suffixes present as AF_<pop> INFO IDs."""
    pops: list[str] = []
    for info_id in vcf_header.info:
        m = GNOMAD_AF_RE.match(info_id)
        if m:
            pops.append(m.group(1))
    return pops


def detect_source(vcf_path: str | Path) -> str:
    """Return 'genotype' if the VCF has samples, else 'info_af'.

    Raises ValueError if the VCF has neither samples nor AF_<pop> INFO fields.
    """
    import pysam

    vcf = pysam.VariantFile(str(vcf_path))
    try:
        n_samples = len(vcf.header.samples)
        gnomad_pops = discover_gnomad_pops(vcf.header)
    finally:
        vcf.close()
    if n_samples > 0:
        return "genotype"
    if gnomad_pops:
        return "info_af"
    raise ValueError(
        f"Cannot auto-detect source for {vcf_path}: no samples and no "
        "AF_<pop> INFO fields. Pass --source explicitly."
    )


# ---------------------------------------------------------------------------
# 1KG sample panel
# ---------------------------------------------------------------------------
def load_sample_panel(panel_path: str | Path | None = None) -> dict[str, str]:
    """Load sample -> pop mapping from a 1KG-format panel file.

    Format: tab-delimited with header; columns sample, pop, super_pop, gender.
    """
    if panel_path is None:
        log.info("Downloading sample panel from 1KG FTP...")
        data = urllib.request.urlopen(PANEL_URL).read().decode()
        lines = data.strip().split("\n")
    else:
        with open(panel_path) as f:
            lines = f.read().strip().split("\n")
    sample_to_pop: dict[str, str] = {}
    for line in lines[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            sample_to_pop[parts[0]] = parts[1]
    log.info("Loaded %d samples from panel", len(sample_to_pop))
    return sample_to_pop


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------
Row = tuple[str, int, str, str, list[float]]


def extract_genotype(
    vcf_path: str | Path,
    pop_config: PopConfig,
    sample_to_pop: dict[str, str],
    min_global_maf: float = 0.01,
) -> Iterator[Row]:
    """Per-sample GT extractor (1KG-style). AC/AN counted across sample indices."""
    import pysam

    vcf = pysam.VariantFile(str(vcf_path))
    samples = list(vcf.header.samples)

    # Validate that every source pop in the rules is present in the panel.
    panel_pops = set(sample_to_pop.values())
    unknown_srcs = pop_config.source_pops() - panel_pops
    if unknown_srcs:
        vcf.close()
        raise ValueError(
            f"pop-config references source pops not present in sample "
            f"panel: {sorted(unknown_srcs)}. Panel pops: {sorted(panel_pops)}"
        )

    out_indices: dict[str, list[int]] = {p: [] for p in pop_config.output_order}
    for i, s in enumerate(samples):
        sp = sample_to_pop.get(s)
        if sp is None:
            continue
        for out_pop, srcs in pop_config.rules.items():
            if sp in srcs:
                out_indices[out_pop].append(i)
                break

    for op in pop_config.output_order:
        log.info("  %s: %d samples", op, len(out_indices[op]))
        if not out_indices[op]:
            vcf.close()
            raise ValueError(
                f"Output pop {op!r} has zero samples in this VCF after "
                f"applying rules. Source pops: {pop_config.rules[op]}"
            )

    try:
        for rec in vcf:
            if len(rec.alts) != 1:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue

            freqs: list[float] = []
            total_ac, total_an = 0, 0
            for op in pop_config.output_order:
                ac, an = 0, 0
                for idx in out_indices[op]:
                    gt = rec.samples[samples[idx]]["GT"]
                    if gt is None:
                        continue
                    for allele in gt:
                        if allele is not None:
                            an += 1
                            if allele > 0:
                                ac += 1
                freqs.append(ac / an if an > 0 else 0.0)
                total_ac += ac
                total_an += an

            if total_an == 0:
                continue
            global_af = total_ac / total_an
            global_maf = min(global_af, 1.0 - global_af)
            if global_maf < min_global_maf:
                continue

            yield (rec.chrom, rec.pos, rec.ref, rec.alts[0], freqs)
    finally:
        vcf.close()


def extract_info_af(
    vcf_path: str | Path,
    pop_config: PopConfig,
    min_global_maf: float = 0.01,
    pass_only: bool = True,
) -> Iterator[Row]:
    """Sites-only AC/AN INFO-field extractor (gnomAD-style).

    For each output pop P with source pops s1..sk:
        AF_P = (sum AC_si) / (sum AN_si)

    Records with any missing AC_<sp> / AN_<sp> field, or where the summed
    AN is zero, are skipped (no per-record data to compute frequency).
    Multiallelic records (len(alts) != 1) are skipped, matching the
    genotype extractor.
    """
    import pysam

    vcf = pysam.VariantFile(str(vcf_path))

    # Validate header has every required AC_<sp> / AN_<sp>.
    src_pops = pop_config.source_pops()
    missing: list[str] = []
    for sp in src_pops:
        if f"AC_{sp}" not in vcf.header.info:
            missing.append(f"AC_{sp}")
        if f"AN_{sp}" not in vcf.header.info:
            missing.append(f"AN_{sp}")
    if missing:
        available = discover_gnomad_pops(vcf.header)
        vcf.close()
        raise ValueError(
            f"VCF header is missing INFO fields required for AC/AN-weighted "
            f"aggregation: {missing}. Discovered AF_<pop> in header: "
            f"{available}"
        )

    try:
        for rec in vcf:
            if len(rec.alts) != 1:
                continue
            if len(rec.ref) != 1 or len(rec.alts[0]) != 1:
                continue
            if pass_only:
                fset = set(rec.filter.keys())
                if fset and fset != {"PASS"}:
                    continue

            freqs: list[float] = []
            total_ac, total_an = 0, 0
            skip = False
            for op in pop_config.output_order:
                ac_sum, an_sum = 0, 0
                for sp in pop_config.rules[op]:
                    ac_val = rec.info.get(f"AC_{sp}")
                    an_val = rec.info.get(f"AN_{sp}")
                    if ac_val is None or an_val is None:
                        skip = True
                        break
                    # AC is Number=A -> tuple of length 1 (we filtered to biallelic).
                    ac = ac_val[0] if isinstance(ac_val, tuple) else ac_val
                    # AN is Number=1 -> scalar.
                    an = an_val
                    if ac is None or an is None:
                        skip = True
                        break
                    ac_sum += int(ac)
                    an_sum += int(an)
                if skip:
                    break
                if an_sum == 0:
                    skip = True
                    break
                freqs.append(ac_sum / an_sum)
                total_ac += ac_sum
                total_an += an_sum

            if skip or total_an == 0:
                continue
            global_af = total_ac / total_an
            global_maf = min(global_af, 1.0 - global_af)
            if global_maf < min_global_maf:
                continue

            yield (rec.chrom, rec.pos, rec.ref, rec.alts[0], freqs)
    finally:
        vcf.close()


# ---------------------------------------------------------------------------
# 1KG download
# ---------------------------------------------------------------------------
def download_kg_vcfs(
    genome: str = "GRCh38",
    dest_dir: Path | None = None,
    chromosomes: list[str] | None = None,
) -> list[Path]:
    """Download 1KG VCFs for the specified genome build."""
    if genome not in KG_VCF_URLS:
        raise ValueError(
            f"No VCF URLs for genome {genome!r}. "
            f"Available: {list(KG_VCF_URLS)}"
        )

    urls = KG_VCF_URLS[genome]
    if chromosomes is None:
        chromosomes = [str(i) for i in range(1, 23)]

    if dest_dir is None:
        dest_dir = Path(tempfile.mkdtemp(prefix="popout_kg_"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    paths: list[Path] = []
    for chrom in chromosomes:
        filename = urls["pattern"].format(chrom=chrom)
        url = f"{urls['base']}/{filename}"
        dest = dest_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            log.info("Already downloaded: %s", dest.name)
        else:
            log.info("Downloading chr%s from %s ...", chrom, urls["base"])
            with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f)
            log.info("  -> %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        paths.append(dest)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_panel_main(argv: list[str] | None = None) -> None:
    """CLI entry point: ``popout build-panel``."""
    parser = argparse.ArgumentParser(
        prog="popout build-panel",
        description=(
            "Build per-population allele-frequency panel TSV. "
            "Supports per-sample (1KG) and sites-only INFO-AF (gnomAD) sources."
        ),
    )
    parser.add_argument(
        "--source", choices=["auto", "1kg", "gnomad"], default="auto",
        help="Input VCF kind (default: auto-detect from header)",
    )
    parser.add_argument("--vcf", nargs="+", help="Input VCF file(s)")
    parser.add_argument("--vcf-dir", help="Directory of per-chromosome VCFs")
    parser.add_argument(
        "--download", action="store_true",
        help="Auto-download 1KG VCFs (only valid for --source 1kg or auto)",
    )
    parser.add_argument(
        "--genome", choices=["GRCh38", "GRCh37"], default="GRCh38",
        help="Genome build for --download (default: GRCh38)",
    )
    parser.add_argument(
        "--chromosomes", nargs="+", default=None,
        help="Chromosomes to download (default: all autosomes 1-22)",
    )
    parser.add_argument(
        "--panel", default=None,
        help="1KG sample panel file (only used for --source 1kg; "
             "auto-downloaded if not provided)",
    )
    parser.add_argument(
        "--pop-config", default=None,
        help="JSON file with output_order and rules. Default: 1KG superpops "
             "for --source 1kg; pass-through over discovered AF_<pop> for gnomad.",
    )
    parser.add_argument("--min-maf", type=float, default=0.01,
                        help="Minimum global MAF filter (default: 0.01)")
    parser.add_argument(
        "--pass-only", action=argparse.BooleanOptionalAction, default=True,
        help="For gnomad source, restrict to FILTER==PASS records (default: on)",
    )
    parser.add_argument("--out", required=True, help="Output panel TSV (.tsv.gz)")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ---- Collect VCF paths ----
    if args.download:
        if args.source not in ("auto", "1kg"):
            parser.error("--download is only valid with --source 1kg or auto")
        log.info("Downloading 1KG VCFs for %s ...", args.genome)
        vcf_paths: list[Path] = download_kg_vcfs(
            genome=args.genome, chromosomes=args.chromosomes,
        )
    elif args.vcf:
        vcf_paths = [Path(p) for p in args.vcf]
    elif args.vcf_dir:
        d = Path(args.vcf_dir)
        vcf_paths = sorted(d.glob("*.vcf.gz")) + sorted(d.glob("*.vcf.bgz"))
    else:
        parser.error("Provide --vcf, --vcf-dir, or --download")

    if not vcf_paths:
        log.error("No VCF files found")
        sys.exit(1)

    log.info("Processing %d VCF file(s)", len(vcf_paths))

    # ---- Resolve source ----
    if args.source == "auto":
        kind = detect_source(vcf_paths[0])
        source = "1kg" if kind == "genotype" else "gnomad"
        log.info("Auto-detected source: %s", source)
    else:
        source = args.source

    # ---- Resolve pop_config ----
    if args.pop_config:
        pop_config = load_pop_config(args.pop_config)
        log.info("Loaded pop config from %s: %s",
                 args.pop_config, list(pop_config.output_order))
    elif source == "1kg":
        pop_config = default_1kg_pop_config()
        log.info("Using default 1KG superpop config: %s",
                 list(pop_config.output_order))
    else:
        import pysam
        vcf = pysam.VariantFile(str(vcf_paths[0]))
        try:
            pops = discover_gnomad_pops(vcf.header)
        finally:
            vcf.close()
        if not pops:
            raise ValueError(
                f"No AF_<pop> INFO fields found in {vcf_paths[0]}; "
                "cannot run --source gnomad without a --pop-config"
            )
        pop_config = passthrough_pop_config(pops)
        log.info("Using pass-through gnomAD pops (from VCF header): %s", pops)

    # ---- 1KG sample panel ----
    sample_to_pop: Optional[dict[str, str]] = None
    if source == "1kg":
        sample_to_pop = load_sample_panel(args.panel)

    if source == "1kg" and not args.pass_only:
        log.info("Note: --no-pass-only has no effect for --source 1kg "
                 "(genotype extractor does not filter on FILTER).")

    # ---- Write output ----
    out_path = Path(args.out)
    opener = gzip.open if out_path.suffix == ".gz" else open
    total = 0
    with opener(out_path, "wt") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            ["#chrom", "pos", "ref", "alt"] + list(pop_config.output_order)
        )
        for vcf_path in vcf_paths:
            log.info("Processing %s", getattr(vcf_path, "name", vcf_path))
            if source == "1kg":
                rows = extract_genotype(
                    vcf_path, pop_config, sample_to_pop,
                    min_global_maf=args.min_maf,
                )
            else:
                rows = extract_info_af(
                    vcf_path, pop_config,
                    min_global_maf=args.min_maf,
                    pass_only=args.pass_only,
                )
            n = 0
            for chrom, pos, ref, alt, freqs in rows:
                writer.writerow(
                    [chrom, pos, ref, alt] + [f"{v:.4f}" for v in freqs]
                )
                n += 1
            log.info("  -> %d variants", n)
            total += n

    log.info("Total: %d variants written to %s", total, out_path)


def build_ref_main(argv: list[str] | None = None) -> None:
    """Deprecated alias: ``popout build-ref`` -> ``popout build-panel --source 1kg``."""
    warnings.warn(
        "popout build-ref is deprecated; use 'popout build-panel --source 1kg'",
        DeprecationWarning,
        stacklevel=2,
    )
    print(
        "WARNING: 'popout build-ref' is deprecated. "
        "Use 'popout build-panel --source 1kg' instead.",
        file=sys.stderr,
    )
    args = list(argv if argv is not None else [])
    if "--source" not in args:
        args = ["--source", "1kg"] + args
    build_panel_main(args)


if __name__ == "__main__":
    build_panel_main()
