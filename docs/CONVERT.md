# popout convert

Convert popout's native outputs into a FLARE-compatible ancestry VCF that downstream tools can consume directly.

## Quick start

```bash
# Run popout with --probs (implies --write-dense-decode)
popout --pgen data/ --map map.txt --out results/cohort --probs

# Convert to FLARE-compatible VCF
popout convert --to vcf \
    --popout-prefix results/cohort \
    --input-vcf data/cohort.phased.vcf.gz \
    --out results/cohort.anc.vcf.gz
```

## Inputs

The convert subcommand reads popout's native output files:

| File | Required | Description |
|------|----------|-------------|
| `{prefix}.model.npz` | yes | Model parameters, ancestry names |
| `{prefix}.chr{N}.decode.npz` | yes | Per-chromosome dense calls, pos_bp, max_post |
| `{prefix}.global.tsv` | no | Per-sample global ancestry proportions |

The `decode.npz` files are produced by `--write-dense-decode` or `--probs`. If you ran popout without either flag, re-run with `--write-dense-decode` to generate them.

## Outputs

| File | Description |
|------|-------------|
| `{out}.anc.vcf.gz` | Ancestry VCF with GT:AN1:AN2[:ANP1:ANP2] FORMAT |
| `{out-stem}.global.anc.gz` | Global ancestry fractions with named columns |

## Full invocation

```bash
popout convert --to vcf \
    --popout-prefix results/cohort \
    --input-vcf data/cohort.phased.vcf.gz \
    --out results/cohort.anc.vcf.gz \
    --probs \
    --ancestry-names "afr,eas,eur,sas,amr" \
    --thinned-sites fill-missing \
    --chroms "1,2,3,4,5"
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--to` | required | Target format (`vcf` is currently the only option) |
| `--popout-prefix` | required | Path prefix matching `--out` from the popout run |
| `--input-vcf` | required | Original phased VCF (provides GT and site alleles) |
| `--out` | required | Output path for `.anc.vcf.gz` |
| `--probs` | off | Emit ANP1/ANP2 posterior fields (requires `--probs` during popout run) |
| `--ancestry-names` | from model | Override ancestry names (comma-list or single-column TSV) |
| `--thinned-sites` | `skip` | How to handle input VCF sites popout did not process |
| `--chroms` | all | Comma-separated list of chromosomes to convert |

## `--thinned-sites` behavior

When popout was run with `--thin-cm`, it processes a subset of the input VCF sites. The converter needs to handle the mismatch:

- **`skip`** (default): Only popout-processed sites appear in the output VCF. The output has fewer sites than the input. This is the recommended mode for downstream analysis that only needs ancestry calls.

- **`fill-missing`**: All input VCF sites are written. Sites popout did not process get `AN1:AN2 = .:.` (missing). Use this when you need a 1:1 site correspondence with the input VCF.

## ANP1/ANP2: the posterior approximation

FLARE emits K floats per haplotype per site for ANP1/ANP2 — one posterior probability per ancestry. Popout stores only the max posterior (one float per haplotype per site in `decode.npz`).

To produce FLARE-compatible K-vectors, popout places `max_post` at the called ancestry's index and distributes the remaining probability mass uniformly:

```
ANP[called_ancestry] = max_post
ANP[other_ancestry]  = (1 - max_post) / (K - 1)
```

This produces a valid probability vector (sums to 1.0) where the max element is the confidence in the call and the off-ancestry mass is uniform. The approximation is exact when the true posterior is maximally concentrated (max_post = 1.0) and conservative otherwise.

**This is not the same as FLARE's full K-dimensional posterior.** It preserves the shape so downstream tools work, but the off-ancestry distribution is synthetic. If your analysis depends on the shape of the posterior across non-called ancestries, use FLARE directly with a popout-exported panel (see [PANEL.md](PANEL.md)).

The FORMAT Description field in the VCF documents this explicitly:
```
##FORMAT=<ID=ANP1,Number=.,Type=Float,Description="Posterior ancestry
probabilities for haplotype 1; popout emits max_post at the called
ancestry and (1-max_post)/(K-1) for others">
```

## Compatibility with FLARE

The output VCF matches FLARE's `.anc.vcf.gz` structure:

- `##ANCESTRY=<name1=0,name2=1,...>` as a single header line
- `GT:AN1:AN2[:ANP1:ANP2]` FORMAT fields
- Sample columns matching the input VCF

Popout's own `popout.benchmark.parsers.flare.parse_flare` can parse the output, as can any tool expecting FLARE ancestry VCFs.

## WDL

A separate `popout_convert.wdl` workflow runs the conversion on CPU (no GPU required). Wire popout's `decode_npz` output array into the convert workflow's input:

```json
{
  "popout_convert.output_prefix": "cohort",
  "popout_convert.tracts_tsv_gz": "gs://bucket/cohort.tracts.tsv.gz",
  "popout_convert.model_npz": "gs://bucket/cohort.model.npz",
  "popout_convert.global_tsv": "gs://bucket/cohort.global.tsv",
  "popout_convert.decode_npz": ["gs://bucket/cohort.chr1.decode.npz", ...],
  "popout_convert.input_vcf": "gs://bucket/cohort.phased.vcf.gz",
  "popout_convert.input_vcf_tbi": "gs://bucket/cohort.phased.vcf.gz.tbi"
}
```

Resources: 16 CPU, 64 GB memory, 500 GB disk (adjustable).
