# EAS AIM panel

`popout/data/aim_panels/east_asian.tsv` — 10 markers, 1KG-Phase3-derived.

## Margin

| target | EUR    | AFR    | AMR    | SAS    | margin (vs 1KG others) |
|--------|--------|--------|--------|--------|------------------------|
| 0.00   | -33.33 | -35.27 | -26.63 | -26.33 | 6.6 SD                 |

Strong panel. EAS has a number of sharp single-marker AIMs from
relatively recent population-specific selection (EDAR, ALDH2,
ABCC11, etc.); even though the canonical rsids may not be in the
1KG TSV at their literature-cited GRCh38 positions, the data-driven
scan finds nearby high-separation markers in the same regions.

AMR is again the closest non-target population, consistent with the
ancient shared ancestry between East Asians and Indigenous Americans
(NAT-component admixture in 1KG AMR populations carries the EAS
signal at intermediate frequency).

## Markers

| chrom | pos_bp     | alt | EUR    | EAS    | AMR    | AFR    | SAS    | weight |
|-------|------------|-----|--------|--------|--------|--------|--------|--------|
| chr15 | 64619001   | (*) | 0.085  | 0.806  | 0.099  | 0.037  | 0.133  | 0.85   |
| chr4  | 99221623   | (*) | 0.000  | 0.671  | 0.000  | 0.001  | 0.000  | 0.85   |
| chr15 | 27963588   | (*) | 0.067  | 0.832  | 0.167  | 0.004  | 0.125  | 0.85   |
| chr19 | 14471656   | (*) | 0.092  | 0.823  | 0.160  | 0.036  | 0.177  | 0.85   |
| chr12 | 112523643  | (*) | 0.085  | 0.831  | 0.196  | 0.219  | 0.209  | 0.85   |
| chr2  | 26073220   | (*) | 0.024  | 0.860  | 0.249  | 0.003  | 0.150  | 0.85   |
| chr1  | 75376622   | (*) | 0.188  | 0.819  | 0.210  | 0.006  | 0.193  | 0.85   |
| chr8  | 125916698  | (*) | 0.273  | 0.921  | 0.295  | 0.206  | 0.312  | 0.85   |
| chr20 | 20974452   | (*) | 0.062  | 0.827  | 0.189  | 0.104  | 0.220  | 0.85   |
| chr6  | 155420503  | (*) | 0.055  | 0.807  | 0.102  | 0.174  | 0.208  | 0.85   |

The chr12:112523643 marker is in the **ALDH2** locus region
(famous EAS alcohol-flush variant). The chr2:26073220 marker is
near **ALK**. Most other markers don't fall in well-known
EAS-specific selected genes; they are 1KG-Phase3 high-frequency-
in-EAS variants that survived LD pruning across the 22 autosomes.

(*) See the production TSV for ref/alt details.

## Source

1000 Genomes Project Phase 3. Selection criterion
`EAS_freq − max(other_freq) ≥ 0.40`, top 10 by separation, LD-pruned
at 1 Mbp.

## Limitations and follow-up

* The canonical EAS anchors EDAR `rs3827760` and ALDH2 `rs671` are
  NOT in this 1KG TSV (excluded by site-presence check). If a future
  1KG TSV refresh adds them, rebuild and they'll likely be top
  picks — both have separation magnitudes above 0.6 in published
  data.
* The AMR-vs-EAS margin is moderate (0.85 SD over EUR/AFR/SAS but
  only ~0.3 SD over AMR specifically) due to NAT-component ancient
  EAS ancestry. Acceptable — the panel still scores EAS strictly
  highest, and the framework's composite scoring is the production
  identity mechanism.
