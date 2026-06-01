# popout DX — run guide

End-to-end recipe for running the popout DX pipeline, from config to
cohort bundle. Read `SCHEMA.md` for the artifact contract and
`PERFORMANCE_CONTRACT.md` for the speed rules every collector must
satisfy.

---

## 1. Local fixture smoke test

Validates the whole pipeline (discover → 2× orchestrator → collate)
end-to-end on a synthetic 60-sample, 2-cluster fixture in ~1 s.

```bash
PYTHONPATH=$GPULAI:$POPOUT python -m validation.popout_dx.tests.test_e2e_fixture
# → ✓ popout DX e2e (global) PASS — bundle: /tmp/.../cohort_dx.smoke_e2e_global.v1.0.0.tar.gz
```

Pass `--keep <dir>` to retain the workspace and inspect artifacts.

## 2. Build the docker image

The popout DX scripts ride along with the existing `validation/` tree
into `lai-tools`. The Dockerfile already copies `validation/` from the
`validation` build context, so adding `validation/popout_dx/` requires
no Dockerfile change — just rebuild.

```bash
cd workflows/lai-tools
./push.sh
# → us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest
```

## 3. Generate a config

Use the `make_dx_config.py` helper rather than hand-writing JSON.

### Global mode (cheap, recommended first run)

```bash
PYTHONPATH=$GPULAI python validation/popout_dx/scripts/make_dx_config.py \
  --run-name popout_dx_aou_v9_chr1 \
  --tools popout,flare,rye,rf \
  --flare-cohort-bundle gs://.../cohort_bundle.flare_validate_chr1.v2.0.0.tar.gz \
  --rye-q gs://prod-drc-broad/aou-srwgs-foxtrot-aux/admixture_estimates/aou_admixture_estimates_rye_pruned_v9.Q \
  --rf-ancestry gs://prod-drc-broad/aou-srwgs-foxtrot-aux/ancestry/foxtrot_v4.ancestry_preds.tsv \
  --clusters 'cluster_*' \
  --chroms 'chr1' \
  --out scripts/popout_dx_config.chr1_all.json
```

popout's path is **not** in the config — it's the WDL input
`popout_dx.popout_outputs`. Set it per-submission in your inputs JSON.

### Global + local mode (adds per-cluster local-ancestry sampling)

Add the extra fields. `--flare-anc-vcf-root` is required because the
cohort bundle does not carry the raw `anc.vcf.gz`.

```bash
... \
  --mode global_local \
  --flare-anc-vcf-root gs://.../flare_pipeline/<wf_id>/call-fit_ancestry_model/ \
  --local-per-bucket-n 25 \
  --local-threshold 0.80 \
  --local-chroms chr1 \
  --out scripts/popout_dx_config.chr1_all_local.json
```

### Cluster / chrom subsetting

The `clusters` and `chroms` config keys are glob patterns matched
against the FLARE cohort bundle's `per_cluster/<cluster_id>/<chrom>/`
tree. Use them to limit a run:

```json
"clusters": ["cluster_000", "cluster_00[1-5]"],
"chroms":   ["chr1", "chr22"]
```

## 4. Submit to Cromwell / Terra

Inputs JSON:

```json
{
  "popout_dx.config_file":    "gs://.../popout_dx_config.chr1_all.json",
  "popout_dx.popout_outputs": "gs://fc-secure-.../submissions/<wf>/popout/<task>/call-popout_task/",
  "popout_dx.run_name":       "popout_dx_aou_v9_chr1_2026_06_01",
  "popout_dx.mode":           "global",
  "popout_dx.tools":          "popout,flare,rye,rf",
  "popout_dx.wandb_api_key":  "<secret>"
}
```

For local mode, set `popout_dx.mode = "global_local"` and (if you want
to override defaults) any of the `local_*` parameters:

```json
{
  "popout_dx.mode": "global_local",
  "popout_dx.local_per_bucket_n": 25,
  "popout_dx.local_threshold":    0.80,
  "popout_dx.local_rng_seed":     42,
  "popout_dx.local_coarse_grids_mb": "1 2 5 10 20"
}
```

Submit:

```bash
java -jar cromwell.jar submit \
  -i inputs.popout_dx.chr1_all.json \
  workflows/popout_dx/wdl/popout_dx.wdl
```

The workflow outputs:
- `runs_manifest_json` — what got selected from the cohort bundle
- `cohort_bundle`       — `cohort_dx.<run_name>.v1.0.0.tar.gz`
- `per_cluster_artifacts` — per-shard tarballs

## 5. Smoke-test budgets

For the first production run, target these wallclocks. Anything
significantly worse is a regression to investigate against
`PERFORMANCE_CONTRACT.md`.

| Run                                                            | Budget        |
|----------------------------------------------------------------|---------------|
| popout DX global, single cluster, chr1                         | total < 5 min |
| popout DX global, chr1 grid (15 clusters)                      | per-shard < 5 min; collate < 2 min |
| popout DX global+local, chr1 grid, per_bucket_n=25, chroms=chr1 | per-shard < 10 min (bcftools-bound; perf contracts 1+4+5 must hold) |

Per-step wallclocks land in W&B as `step.<name>.wallclock_seconds`
(see `PERFORMANCE_CONTRACT.md` #11). Tier-1 headlines:
`popout_dx.mean_ccc_vs_flare`, `popout_dx.local_bp_agreement_vs_flare`,
etc.

## 6. After the run

Grab the cohort bundle and render off-line:

```bash
gsutil cp gs://.../popout_dx_aou_v9_chr1.../cohort_dx.popout_dx_aou_v9_chr1_2026_05_30.v1.0.0.tar.gz .
tar tzf cohort_dx.popout_dx_aou_v9_chr1_2026_05_30.v1.0.0.tar.gz
# cohort_dx/cohort_manifest.json
# cohort_dx/cohort_summary.json
# cohort_dx/cohort/manifest.tsv
# cohort_dx/cohort/tier1_metrics.tsv
# cohort_dx/cohort/per_sample_mae.tsv
# cohort_dx/cohort/pairwise_soft_summary.tsv
# cohort_dx/cohort/popout_vs_{flare,rye,rf}.{confusion,metrics}.tsv
# (+ local/* tables when mode=global_local)
```

The off-line renderer (separate tooling, on the laptop) consumes the
unpacked `cohort_dx/` tree.

## Troubleshooting

- **`align_labels` step fails** with `RF probability vector has N entries
  but RF_LABEL_ORDER declares M` — the RF input file uses a non-canonical
  label set. Check that `--rf-ancestry` points at a foxtrot v4
  (afr/amr/eas/eur/mid/sas) file.

- **`discover_runs` errors with `popout.run_dir contained no *.global.tsv
  anchor`** — the popout run directory has zero or multiple `*.global.tsv`
  files. A run directory must have exactly one. Re-run popout, or split.

- **`flare.cohort_bundle missing global.tsv for N selected pairs`** — the
  cohort bundle's `per_cluster/` tree is missing one of the requested
  (cluster, chrom) pairs. The clusters/chroms globs over-selected; tighten
  them or regenerate the cohort bundle.

- **`mode=global_local requires config.flare.anc_vcf_root`** — local mode
  needs the raw FLARE `.anc.vcf.gz` (the bundle doesn't carry it). Pass
  `--flare-anc-vcf-root` to `make_dx_config.py`.

- **WDL fails at scatter with `Cannot coerce String to File`** — the
  optional File inputs (popout_summary, rye_q_path, rf_ancestry_path)
  come from conditionally-declared local Files: `if (row[N] != "") {
  File foo_opt = row[N] }`. This is the WDL 1.0 idiom for "Optional[File]
  from possibly-empty String". If your Cromwell build instead complains
  about the conditional declaration, fall back to declaring the input as
  String in the task and have the task script handle the empty case.
