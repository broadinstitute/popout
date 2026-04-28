# MID AIM panel

`popout/data/aim_panels/middle_east.tsv` — 10 markers, synthetic
Levantine-proxy.

## Margin

| target | AFR   | EUR   | EAS   | AMR   | SAS   | margin (vs all 1KG)    |
|--------|-------|-------|-------|-------|-------|------------------------|
| 0.00   | -1.74 | -0.87 | -1.45 | -1.06 | -0.87 | 2.5 SD                 |

**MID margin contract: 2.0 SD (relaxed).** The MID panel uses a
synthetic Levantine proxy, not real 1KG MID data (no MID superpop
exists in the 1KG release). The 2-SD threshold is documented in
both the panel test and the framework spec.

## Synthetic-proxy approach

MID is the only panel without a real ground-truth reference vector
in the 1KG TSV. The synthetic proxy is:

```
MID_proxy_freq = (EUR_freq + SAS_freq) / 2
```

Levantine populations cluster genetically between European and
South Asian ancestries in 1KG-derived PCA, with secondary
contributions from African and Caucasian groups. The midpoint
proxy is a coarse but principled model of that intermediate
position. Selection criteria for panel inclusion:

1. **EUR ≠ SAS by ≥ 0.40** at the locus, so the midpoint is
   meaningfully distinct from both pure populations.
2. **Midpoint ≥ 0.20 from AFR, EAS, and AMR**, so non-Levantine
   1KG superpops score visibly worse than the proxy.

Test reference: `tests/data/aim_panels/mid_reference.tsv`
ships per-locus MID frequencies for every panel position from
all six panels (not just MID's), each row carrying source
`synthetic_proxy_eur_sas_1KG_Phase3`. The test reference
**must cover all panels' positions**, not just MID's — otherwise
non-MID panels score 0 (no overlap) on MID reference, tying with
the target's perfect-fit score and breaking the argmax contract.

The test contract is "panel correctly identifies a component whose
frequencies match the synthetic Levantine signature," not "panel
identifies real Levantine ancestry from 1KG ground truth" — no
such ground truth exists in the dataset. The test still validates
the framework's ability to discriminate intermediate-between-EUR-
and-SAS components from extreme components, which is the
production behavior we want.

## Markers

| chrom | pos_bp    | alt | EUR    | EAS    | AMR    | AFR    | SAS    | MID-proxy | weight |
|-------|-----------|-----|--------|--------|--------|--------|--------|-----------|--------|
| chr16 | 31087679  | T   | 0.002  | 0.002  | 0.000  | 0.000  | 0.539  | 0.270     | 0.40   |
| chr2  | 96363430  | A   | 0.258  | 0.214  | 0.231  | 0.037  | 0.689  | 0.474     | 0.40   |
| chr12 | 54992285  | T   | 0.221  | 0.680  | 0.225  | 0.140  | 0.654  | 0.438     | 0.40   |
| chr16 | 36236378  | A   | 0.000  | 0.000  | 0.000  | 0.000  | 0.424  | 0.212     | 0.40   |
| chr4  | 37079058  | T   | 0.688  | 0.692  | 0.718  | 0.779  | 0.268  | 0.478     | 0.40   |
| chr14 | 49361649  | A   | 0.833  | 0.205  | 0.898  | 0.838  | 0.414  | 0.624     | 0.40   |
| chr15 | 64232781  | A   | 0.237  | 0.140  | 0.209  | 0.963  | 0.649  | 0.443     | 0.40   |
| chr8  | 41851552  | A   | 0.225  | 0.845  | 0.189  | 0.681  | 0.632  | 0.428     | 0.40   |
| chr16 | 34102164  | A   | 0.000  | 0.000  | 0.000  | 0.000  | 0.405  | 0.203     | 0.40   |
| chr16 | 46812751  | C   | 0.096  | 0.090  | 0.037  | 0.004  | 0.501  | 0.299     | 0.40   |

MID-proxy is `(EUR + SAS) / 2`. The expected_freq in the production
panel TSV is the MID-proxy column.

Notice the four chr16 markers shared with the SAS panel — these are
loci where SAS has a high allele frequency, EUR has near zero, and
the midpoint sits around 0.20–0.27. The MID panel scores a
synthetic-MID component (which has the proxy freqs) higher than a
pure SAS component (which has SAS freqs around 0.4–0.5) and much
higher than a pure EUR component (near zero).

## Source

* Panel positions: `1KG_Phase3` (data-driven scan with synthetic-MID
  selection criteria).
* Panel `expected_freq`: `synthetic_proxy_eur_sas_1KG_Phase3`
  (the (EUR+SAS)/2 midpoint computed from 1KG frequencies at each
  locus).

## Limitations and follow-up

* **No real MID ground truth.** The synthetic-proxy approach
  validates that the framework can identify intermediate-between-
  EUR-and-SAS components, but not that those components
  necessarily correspond to actual Levantine ancestry. In a
  production AoU run, components representing real Levantine
  participants will have empirical allele frequencies in the
  cohort that approximate the synthetic proxy IF Levantine
  participants in the cohort have ancestry profiles consistent
  with the EUR-SAS-midpoint model.
* **Refresh path:** if Behar 2010 / Haber 2013 / Reich AADR
  Levantine genotype data becomes available, the right rebuild
  is to:
  1. Compute real MID allele frequencies at every panel locus
     from those datasets.
  2. Replace the panel's `expected_freq` with the
     literature-derived values.
  3. Replace `tests/data/aim_panels/mid_reference.tsv` with the
     same literature values.
  4. Re-validate; the margin will likely change but the structure
     is the same.
* The MID panel's framework role is composite with F_ST scoring,
  except that there is no MID F_ST reference (MID is absent from
  the 1KG TSV). This means MID priors rely on the AIM panel
  alone — `configs/priors_v2.yaml` reflects this in the MID
  prior having only an `aims:` block, no `fst_reference:` block.
