# popout benchmark

A tool for comparing local ancestry inference (LAI) outputs from multiple tools against each other or against ground truth from simulations.

## Usage

```bash
python -m popout.benchmark \
    --tool popout:sim_results/popout_tracts.tsv.gz \
    --tool flare:sim_results/flare_chr1.anc.vcf.gz:sim_results/flare_chr1.global.anc.gz \
    --truth sim_results/truth_chr1.npz \
    --chrom chr1 \
    --output bench_report/
```

Produces `bench_report/report.md` with tables and `bench_report/plots/` with figures.

## Supported tools

| Tool | File format | Parser |
|------|-------------|--------|
| popout | `.tracts.tsv.gz` (BED-like tract intervals) | `popout.benchmark.parsers.popout` |
| FLARE | `.anc.vcf.gz` + optional `.global.anc.gz` | `popout.benchmark.parsers.flare` |
| truth | `.npz` with `true_ancestry` and `pos_bp` arrays | `popout.benchmark.parsers.truth` |

RFMix, Gnomix, SparsePainter, Orchestra, and Recomb-Mix parsers are planned but not yet implemented.

## Metrics

- **Per-ancestry r²**: Pearson r² between per-haplotype ancestry fractions. Primary metric in 2025 LAI benchmarking literature.
- **Per-site accuracy**: Fraction of (haplotype, site) pairs where tools agree.
- **Per-ancestry precision/recall**: Agreement broken down by ancestry label.
- **Global ancestry fraction error**: Per-sample L1 distance between genome-wide ancestry proportions.
- **Tract length statistics**: Min/max/median/mean/quartiles of tract lengths in bp and sites.
- **Wall time / Peak RSS**: Passed via tool metadata (not measured by the benchmark tool).

## Label alignment

popout produces arbitrary integer labels with no pre-defined correspondence to population names. The benchmark tool uses Hungarian optimization (`scipy.optimize.linear_sum_assignment`) to find the best label mapping from popout to the reference or ground truth.

Override automatic matching with `--label-map path/to/mapping.csv` (columns: `tool,src_label,ref_label`).

## Known limitations

- No ground-truth-free statistical tests (e.g., bootstrapped CIs).
- Parsers target specific output versions of each tool.
- Haplotype phase-swap detection is not implemented — tools must agree on which haplotype is 0 vs 1 within each sample.
- Designed for simulator-scale (thousands of haplotypes). Not optimized for biobank-scale.
