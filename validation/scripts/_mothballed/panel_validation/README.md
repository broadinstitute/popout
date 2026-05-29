# Mothballed: panel validation

The contents of this directory were originally rolled into the FLARE
output-validation pipeline as a preflight ("does the gt VCF actually
overlap the reference panel"). That was a category error: the question
is about **the panel**, not about FLARE's output. If FLARE emits ~500 k
calls on chr1, we already have site-level concordance — running an
O(panel × cluster) overlap audit alongside every output run was wasted
time and money.

This code is kept here because the question itself is real — it just
belongs in a separate `panel_validation` pipeline that runs once per
panel build, not per (cluster, chrom). When that pipeline lands it can
lift this code verbatim.

## Files

- `validate_ref_target_concordance.py` — original R6 audit. Computes
  exact `(chrom, pos, ref, alt)` overlap between a FLARE reference VCF
  and a per-cluster gt VCF, classifies misses into
  `absent_in_target / position_match_but_alleles_differ /
  exact_match_found_on_reinspection`, emits a wide-form summary TSV +
  pass/fail JSON.
- `ref_target_concordance.meta.json` — old baseline metadata
  (`exact_overlap_pct >= 94.5` was the pass threshold).

## When you revive this

- The script expects a `.tbi` next to each `.vcf.gz`. **Convention**:
  derive the tabix path as `vcf_path + ".tbi"` rather than carrying a
  separate path; if the two have different parents, that's an upstream
  bug (e.g. an output declaration with two separate `glob()` calls — see
  `workflows/plink2/wdl/plink2_export_clusters.wdl`).
- It does NOT depend on any FLARE output; only the FLARE reference VCF
  and the per-cluster gt VCF. So this lifts cleanly into a standalone
  WDL.
