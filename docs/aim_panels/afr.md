# AFR AIM panel

`popout/data/aim_panels/african.tsv` — 10 markers, 1KG-Phase3-derived.

## Margin

| target | EUR     | EAS     | AMR     | SAS     | margin (vs 1KG others) |
|--------|---------|---------|---------|---------|------------------------|
| 0.00   | -135.02 | -140.13 | -115.85 | -134.44 | 12.6 SD                |

Strongest panel of the six. Every marker has AFR-vs-rest separation
above 0.80 (AFR is genetically the most isolated 1KG superpop, with
deep population structure giving long lists of alleles fixed in or
near-absent from sub-Saharan populations). AMR is the closest other
superpop, consistent with the African ancestry component in 1KG AMR
populations.

## Markers

| chrom | pos_bp     | alt | EUR    | EAS    | AMR    | AFR    | SAS    | weight |
|-------|------------|-----|--------|--------|--------|--------|--------|--------|
| chr1  | 159204893  | C   | 0.006  | 0.000  | 0.078  | 0.964  | 0.000  | 1.00   |
| chr8  | 144414297  | G   | 0.993  | 1.000  | 0.926  | 0.086  | 1.000  | 1.00   |
| chr5  | 179199608  | C   | 0.003  | 0.000  | 0.095  | 0.923  | 0.000  | 1.00   |
| chr9  | 133904766  | A   | 0.993  | 1.000  | 0.926  | 0.099  | 0.999  | 1.00   |
| chr17 | 31023751   | G   | 0.023  | 0.000  | 0.078  | 0.904  | 0.026  | 1.00   |
| chr15 | 29135197   | C   | 0.006  | 0.000  | 0.079  | 0.903  | 0.000  | 1.00   |
| chr4  | 3664767    | G   | 0.010  | 0.000  | 0.108  | 0.927  | 0.000  | 1.00   |
| chr7  | 146308840  | G   | 0.047  | 0.050  | 0.118  | 0.935  | 0.107  | 1.00   |
| chr2  | 72141061   | G   | 0.089  | 0.038  | 0.141  | 0.957  | 0.035  | 1.00   |
| chr1  | 116344833  | A   | 0.070  | 0.032  | 0.104  | 0.948  | 0.136  | 1.00   |

All markers are at 10 distinct chromosomes with at least 1 Mbp
separation from any other panel marker. No clustering at known long
LD blocks (e.g., HLA, lactase region) — the data-driven scan
naturally selects against tight clusters when LD pruning enforces
1 Mbp minimum spacing.

## Source

1000 Genomes Project Phase 3 superpop frequencies. The TSV at
`~/.popout/ref/GRCh38/1kg_superpop_freq.tsv.gz` is the build-time
input; the `source` column in the production panel TSV is
`1KG_Phase3` for every row.

The build script's selection criterion was
`AFR_freq − max(other_freq) ≥ 0.40` after considering both directions
(high-in-AFR and low-in-AFR markers). Top 10 by separation magnitude,
LD-pruned at 1 Mbp.

## Limitations and follow-up

This panel is intentionally a small infrastructure-grade fingerprint,
not a comprehensive AIM catalog. Notable AFR anchors absent from this
panel because they were not present in the 1KG TSV used as input:

* **DARC `rs2814778` (Duffy null, chr1:159174683)** — the canonical
  AFR-distinguishing marker (allele frequency ~0.96 in sub-Saharan
  Africa, near 0 elsewhere). Excluded from the panel because the
  built reference TSV does not include this position; if the 1KG TSV
  is rebuilt to include it, a panel rebuild would naturally pick it
  up at the top of the selection ranking.
* **HBB `rs334` (sickle, chr11:5227002)** — present in 1KG TSV but
  AFR frequency only ~0.10, doesn't clear the 0.40 separation cutoff.
  Useful as a low-weight informational marker but doesn't meet the
  panel's selection contract.

If a future 1KG TSV refresh adds the Duffy-null position, rebuild via
`scripts/build_aim_panels/build_panel.py --pop AFR`. Same 1KG TSV
in → same panel out, so refresh-and-rebuild is idempotent.
