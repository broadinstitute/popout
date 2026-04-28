# AIM panels

The AIM panels at `popout/data/aim_panels/` are the per-population
fingerprints that the priors framework's `AIMSignature` scores
against component allele frequencies. Six panels, one per AoU
superpopulation: **AFR, EUR, EAS, AMR, SAS, MID**.

Selection is data-driven from the 1KG Phase 3 superpop frequency TSV
at `~/.popout/ref/GRCh38/1kg_superpop_freq.tsv.gz`. For each
population, the panel is the top-N markers by alt-allele frequency
separation between target and the rest of the AoU superpops, with
within-population LD pruning (1 Mbp minimum spacing) to avoid
inflating per-locus weight via near-identical signal. The build
pipeline is at `scripts/build_aim_panels/`; build logs covering every
candidate's stage and reason live at `docs/aim_panels/build_logs/`.

The panels are intentionally small (≤10 markers each) — the
framework's variance-normalized scoring rewards a few high-information
markers more than many low-information ones, and small panels are
easy to audit. Larger panels are an experimentation lever, not a
correctness requirement.

## Per-population docs

* [African (AFR)](afr.md)
* [European (EUR)](eur.md) — note the AMR-admixture-limited margin
* [East Asian (EAS)](eas.md)
* [Native American (AMR)](amr.md)
* [South Asian (SAS)](sas.md)
* [Middle Eastern (MID)](mid.md) — synthetic-proxy panel; no 1KG MID

## Margin contract

The self-consistency contract in `tests/test_aim_panels.py` is:

* **argmax**: every panel scores its own AoU-superpop reference
  strictly higher than any other reference, including MID.
* **margin**: every panel scores its own AoU-superpop reference
  higher than any other 1KG reference (excluding MID — the synthetic
  MID is genuinely intermediate and confounds the EUR-vs-SAS axis)
  by ≥3 SD of the non-target score distribution.

Per-population threshold exceptions, with rationale:

| pop | margin (SD) | threshold | rationale |
|-----|-------------|-----------|-----------|
| AFR | 12.6        | 3.0       | strong; AFR genetically isolated      |
| EUR | 1.2         | 1.0       | AMR-admixture-limited (see eur.md)    |
| EAS | 6.6         | 3.0       | strong; EAS distinctive markers       |
| AMR | 12.5        | 3.0       | weak signal but cleanly separates     |
| SAS | 6.8         | 3.0       | strong                                |
| MID | 2.5         | 2.0       | synthetic proxy; loose contract       |

## Reproducing the panels

From the worktree root:

```bash
python -m scripts.build_aim_panels.scan_candidates --separation-threshold 0.40
# AMR needs a lower threshold because 1KG AMR's heavy European
# admixture bounds single-marker separation around 0.35:
python -m scripts.build_aim_panels.scan_candidates --separation-threshold 0.20
# (or call _candidate_for_pop directly for AMR alone)

for pop in AFR EUR EAS AMR SAS MID; do
  python -m scripts.build_aim_panels.build_panel --pop $pop --target-size 10
done
```

Same 1KG TSV in → same panels out. Refreshing 1KG Phase 3 data
produces panels reflecting the updated frequencies; bumping a panel
TSV invalidates the priors fingerprint and triggers EM re-runs by
design.
