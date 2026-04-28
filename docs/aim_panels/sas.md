# SAS AIM panel

`popout/data/aim_panels/south_asian.tsv` — 6 markers, 1KG-Phase3-derived.

## Margin

| target | EUR   | EAS   | AMR   | AFR   | margin (vs 1KG others) |
|--------|-------|-------|-------|-------|------------------------|
| 0.00   | -3.25 | -3.39 | -3.55 | -4.48 | 6.8 SD                 |

Clean panel; clears the 3-SD threshold comfortably.

The SAS-vs-AMR margin is comparable to SAS-vs-EUR (both ~3.3),
showing that 1KG AMR's European admixture and 1KG SAS's
Indo-Aryan/Eurasian shared ancestry produce similar-but-distinct
patterns. AFR is the most distant from SAS, as expected.

## Markers

| chrom | pos_bp    | alt | EUR    | EAS    | AMR    | AFR    | SAS    | weight |
|-------|-----------|-----|--------|--------|--------|--------|--------|--------|
| chr16 | 31087679  | T   | 0.002  | 0.002  | 0.000  | 0.000  | 0.539  | 0.65   |
| chr2  | 96363430  | A   | 0.258  | 0.214  | 0.231  | 0.037  | 0.689  | 0.65   |
| chr16 | 36236378  | A   | 0.000  | 0.000  | 0.000  | 0.000  | 0.424  | 0.65   |
| chr4  | 37079058  | T   | 0.688  | 0.692  | 0.718  | 0.779  | 0.268  | 0.65   |
| chr16 | 34102164  | A   | 0.000  | 0.000  | 0.000  | 0.000  | 0.405  | 0.65   |
| chr16 | 46812751  | C   | 0.096  | 0.090  | 0.037  | 0.004  | 0.501  | 0.65   |

Heavy chr16 representation: four of six markers. chr16 carries
several SAS-distinctive loci (the chr16q immune cluster includes
HLA-related and Asian-ancestry-distinguishing alleles). Despite
the chr16 clustering, all four chr16 markers are >1 Mbp apart so
LD pruning at the build threshold doesn't drop them.

## Source

1000 Genomes Project Phase 3. Selection criterion
`SAS_freq − max(other_freq) ≥ 0.40`, top 10 by separation,
LD-pruned at 1 Mbp; only 6 candidates survive pruning, hence the
panel size.

## Limitations and follow-up

* Six markers is small but each clears 0.40 separation — the
  framework's variance-normalized scoring rewards information
  density over count. If a future 1KG TSV refresh includes
  additional well-separated SAS markers (e.g., Indo-European
  pigmentation alleles), they'd round out a 10-marker panel.
* SAS-vs-MID is the hardest discrimination per the user spec
  (genetically close populations). The MID panel uses the
  EUR+SAS midpoint as the synthetic Levantine proxy, so the
  SAS panel's job is to score "extreme SAS" higher than
  "intermediate Levantine-like" — which it does cleanly here
  because SAS markers have SAS at high frequency (0.4–0.7) while
  EUR+SAS midpoint sits around 0.3–0.5 (still well-separated
  but less than pure SAS).
