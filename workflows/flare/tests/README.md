# flare_pipeline local smoke test

Synthetic end-to-end test for `workflows/flare/wdl/flare_pipeline.wdl`. Catches
WDL bugs that would otherwise only surface on Terra after paying biobank-scale
VCF localization costs (the original `bcftools +split --threads` bug burned $45
and ~3 hours of Cromwell compute before it tripped).

## What it covers

A single `miniwdl run` exercises every stage and helper in the pipeline:

- Stage A — `plink2 --pfile --keep --export vcf-4.2 bgz` (one task per chrom,
  loops over clusters; `bcftools index --tbi` on each per-cluster output)
- `find_chrom_index` helper task
- WDL `transpose()` on the `[chrom][cluster]` array
- Stage B — FLARE `em=true` (train, produces `.model`)
- Stage C — FLARE `em=false model=…` (apply, reuses train's model)
- `select_first([apply.anc_vcf[ci], train.anc_vcf])` model-chrom slot-back
- Stage D — `bcftools merge` per chromosome (column-wise across clusters)
- `magicwand` bootstrap (offline W&B, no `wandb_api_key` needed)

Dimensions: 2 chromosomes (`chr20`, `chr21`), 24 target samples in 4 clusters,
15 reference samples across 3 populations, ~500 variants per chrom. Whole
dataset is <5 MB and the run completes in **under 2 minutes** post-image-pull.

## Run it

```bash
python workflows/flare/tests/make_synthetic_flare_dataset.py --out data/synthetic_flare/

miniwdl run \
  workflows/flare/wdl/flare_pipeline.wdl \
  --input data/synthetic_flare/inputs.json \
  --dir /tmp/miniwdl_flare_smoke/
```

The generator overwrites `data/synthetic_flare/` (gitignored). A fresh
`--dir` is recommended each run so miniwdl's output paths don't collide.

## Verify outputs

Each chromosome should produce one merged anc VCF with:
- 24 samples (union of all 4 clusters)
- ~450–500 records (one chrom's worth)
- `AN1` and `AN2` FORMAT fields (FLARE hard-call ancestry)

```bash
F=/tmp/miniwdl_flare_smoke/<run-id>/out/chrom_anc_vcfs/0/chr20.anc.vcf.gz
bcftools query -l "$F" | wc -l                      # → 24
bcftools view -H "$F" | cut -f1 | sort -u           # → chr20
bcftools view -h "$F" | grep '^##FORMAT'            # → GT, AN1, AN2
```

## Prereqs

- `miniwdl` ≥ 1.13 on PATH
- Docker running, with access to `us-docker.pkg.dev/broad-dsde-methods/popout/{bcftools,flare,plink2}:latest`
  (run `gcloud auth configure-docker us-docker.pkg.dev` once if pulls fail)
- Host has `python3` + `pysam` (any version ≥ 0.22) and `plink2` on PATH
  for the generator (plink2 materializes the per-chrom PGEN triplets that
  Stage A consumes). Install plink2 from
  <https://www.cog-genomics.org/plink/2.0/> — the static prebuilt
  `plink2_macarm64_*` or `plink2_linux_avx2_*` binaries both work.
