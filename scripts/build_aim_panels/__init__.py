"""AIM panel build pipeline for the emergent-identity priors framework.

The pipeline consumes the 1KG superpop frequency TSV at
``~/.popout/ref/GRCh38/1kg_superpop_freq.tsv.gz`` (the same data the
priors framework's ``FSTReferenceSignature`` resolves at runtime) and
produces six panels at ``popout/data/aim_panels/``, one per AoU
superpopulation.

Selection is data-driven: candidate markers are chosen by their
target-vs-rest allele-frequency separation in the 1KG TSV, ranked,
LD-pruned within population, and written as the production panel
schema (chrom, pos_bp, ref, alt, expected_freq, weight, source).

Stages:

* ``scan_candidates`` — phases 1-3: stream the 1KG TSV; for each
  target population, emit candidates whose target-vs-max-others
  separation exceeds a configurable threshold; trivially satisfies
  the site-presence requirement (the TSV defines the cohort SNP
  set the priors framework will see).
* ``build_panel`` — phases 4-6: LD-prune (frequency-variance proxy),
  rank by separation, take top N, write production TSV plus build
  log.

The build is deterministic and reproducible: same 1KG TSV in →
same panels out. Re-running after a 1KG refresh produces panels
that reflect the updated data.
"""
