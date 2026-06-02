# popout DX artifact schema

Schema version: **1.0.0** (semver — bump on any layout change)

## Version history

- **1.0.0** — initial schema. Two modes (`global`, `global_local`), config-driven, tool-gated per-cluster and cohort layouts.

This file is the contract between the three stages of popout DX:

| Stage | Producer | Consumer |
|---|---|---|
| 1. per-cluster DX (WDL scatter) | `validation/popout_dx/scripts/run_dx_cluster.py` | Stage 2, debugging |
| 2. cohort collation (WDL gather) | `validation/popout_dx/scripts/collate_dx.py` | Stage 3, laptop renderers |
| 3. reporting (off-line, on the laptop) | downstream tools, not part of this package | humans |

If you change a file's columns, you bump the schema. New optional file ⇒ minor bump. Removed/renamed file ⇒ major bump.

The schema lives in two places: this Markdown doc (prose contract) and `validation/popout_dx/schema.py` (executable dataclasses + tarball validator).

The companion document `PERFORMANCE_CONTRACT.md` lists the non-negotiable speed rules every collector in this package must satisfy. Read it before adding a new step.

---

## 0. Run shape

The pipeline is driven by a small JSON config — see `validation/popout_dx/scripts/make_dx_config.py` for the writer and `discover_runs.py` for the consumer.

popout's path is **not** in the config; it's a WDL input (`popout_dx.popout_outputs`). The config covers the comparison universe (FLARE bundle, rye / rf refs, glob filters, local-sampling knobs).

```json
{
  "run_name": "popout_dx_aou_v9_chr1",
  "schema_version": "1.0.0",
  "tools": ["popout", "flare", "rye", "rf"],
  "flare": {
    "cohort_bundle": "gs://.../cohort_bundle.flare_validate_chr1.v2.0.0.tar.gz",
    "anc_vcf": {
      "cluster_000.chr1":  "gs://.../cluster_000.chr1.anc.vcf.gz",
      "cluster_000.chr10": "gs://.../cluster_000.chr10.anc.vcf.gz",
      "...":                "..."
    }
  },
  "rye": { "q_path":        "gs://.../aou_admixture_estimates_rye_pruned_v9.Q" },
  "rf":  { "ancestry_path": "gs://.../foxtrot_v4.ancestry_preds.tsv" },
  "clusters": ["cluster_*"],
  "chroms":   ["chr*"],
  "local_sampling": {
    "per_bucket_n": 25,
    "threshold": 0.80,
    "rng_seed": 42,
    "chroms": ["chr1"],
    "coarse_grid_resolutions_mb": [1, 2, 5, 10, 20]
  }
}
```

Notes:
- `flare.anc_vcf` is an inline `{"<cluster_id>.<chrom>": "gs://...anc.vcf.gz"}` map. Required only for `mode=global_local` (cohort bundle does not carry the raw VCFs); every selected `(cluster_id, chrom)` pair must have an entry. Build it from the Terra `cluster_chrom` data table — see `scripts/cluster_chrom.tsv` for the AoU v9 canonical version and `make_dx_config.py --flare-anc-vcf-tsv` for the inliner.
- `local_sampling` is consumed only when `mode=global_local`.

### Data-model asymmetry

- **popout is whole-cohort.** One popout run produces a single `global.tsv` over every sample in the cohort, a single `tracts.tsv.gz` covering every sample × every chrom, and one model. Per-chrom decode parquets are optional siblings (only when `write_dense_decode=true`). There is no popout-side notion of "cluster".
- **FLARE is per-(cluster, chrom)** today. The current default modality scatters FLARE inference across clusters. Whole-cohort single-chrom FLARE VCFs are planned but not here yet; when they arrive this schema absorbs them by collapsing the cluster dimension in the config (a single synthetic `cluster_id="cohort"` entry in the bundle).

The scatter unit is `(cluster_id, chrom)` as defined by FLARE. Per shard, popout data is the **same whole-cohort files** (subset to the cluster's sample roster, which the shard derives from the FLARE per-cluster `global.tsv`). The popout paths in the manifest are therefore singletons — repeated across every row of the TSV for self-containment, but pointing at the same underlying objects.

A `discover_runs` task validates the config once and emits a deterministic, sorted `runs_manifest.json` + `runs_manifest.tsv` listing every selected `(cluster_id, chrom)` triple with resolved FLARE paths per row and the (constant) popout / rye / rf paths repeated for convenience. Scatter inputs come from the manifest — never from the raw config.

### Popout run catalog (future)

Out of scope for v1.0.0 but planned: a JSON catalog at a known GCS location maintained by a separate auditing tool. Each popout run registers itself (run_name → run_dir + metadata). DX configs would then reference `popout.run_name` (with `$LAST` as a magic alias for the most recent registered run) instead of `popout.run_dir`. The discover_runs config validator already reserves `popout.run_name` and `popout.catalog` to keep this addition non-breaking.

---

## 1. Per-cluster DX artifact tarball

Produced by Stage 1. One tarball per `(cluster_id, chrom)`.

**Tarball filename:** `<cluster_id>.<chrom>.popout_dx.v<schema_version>.tar.gz`

**Top-level prefix inside the tarball:** `<cluster_id>/<chrom>/`

```
<cluster_id>/<chrom>/
├── manifest.json
├── tier1_metrics.tsv
├── popout.global.tsv                       (popout-format, K_popout columns)
├── labels.json                             (popout ↔ RF alignment)
├── global/
│   ├── pairwise_hard/
│   │   ├── popout_vs_flare.confusion.tsv    (★ gated on tools includes flare)
│   │   ├── popout_vs_rye.confusion.tsv      (★ gated on tools includes rye)
│   │   └── popout_vs_rf.confusion.tsv       (★ gated on tools includes rf)
│   └── pairwise_soft/
│       ├── popout_vs_flare.metrics.tsv      (★ gated)
│       ├── popout_vs_rye.metrics.tsv        (★ gated)
│       ├── popout_vs_rf.metrics.tsv         (★ gated)
│       ├── per_sample_mae.tsv               (always present; null cols when a tool absent)
│       └── summary.json                     (always present)
├── local/                                  (★ gated on mode == global_local)
│   ├── selected_samples.tsv
│   ├── local_per_sample.tsv
│   ├── local_per_haplotype.tsv
│   ├── local_summary.json
│   └── views/
│       ├── bp_confusion_segments.tsv.gz
│       ├── boundary_localization.tsv
│       └── coarse_grid_summary.tsv
└── provenance/
    ├── schema_version.txt
    └── dx_config.json                       (the exact config file used)
```

### 1.1 `manifest.json`

Single JSON object. Keys:

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | `"1.0.0"` |
| `cluster_id` | str | e.g. `"cluster_007"` |
| `chrom` | str | e.g. `"chr1"` |
| `run_name` | str | propagated from config |
| `mode` | str | `"global"` or `"global_local"` |
| `tools` | array | subset of `{popout, flare, rye, rf}`; `popout` always present; at least one other |
| `n_samples` | int | sample count in `popout.global.tsv` |
| `n_ancestries_popout` | int | K_popout |
| `popout_run_dir` | str | resolved GCS path to the popout run for this cluster |
| `steps` | object | per-step `{wallclock_seconds: float, peak_rss_gb: float, exit: int, status}` |
| `total_wallclock_seconds` | float | sum |
| `peak_rss_gb` | float | max across steps |
| `cpu_wall_ratio` | float | total CPU s / total wall s |
| `generated_at` | str | ISO 8601 UTC |
| `optional_inputs` | object | `{flare: bool, rye: bool, rf: bool, local_mode: bool}` — must agree with `tools` and `mode` |

### 1.2 `tier1_metrics.tsv`

Two-column TSV: `key<TAB>value`. Consumed by the WDL command block, which emits one `magicwand log <key>=<value>` line per row.

```
popout_dx.cluster_id
popout_dx.chrom
popout_dx.mode
popout_dx.n_samples
popout_dx.n_ancestries_popout
popout_dx.peak_rss_gb
popout_dx.cpu_wall_ratio
popout_dx.global_ccc_vs_flare                 (NA if flare not in tools)
popout_dx.global_ccc_vs_rye                   (NA if rye not in tools)
popout_dx.global_hardcall_agree_vs_rf         (NA if rf not in tools)
popout_dx.local_bp_agreement_vs_flare         (NA if mode == global)
popout_dx.local_calibration_drift_fraction    (NA if mode == global)
step.<name>.wallclock_seconds                 (one row per orchestrator step — perf contract #11)
```

### 1.3 `popout.global.tsv`

Verbatim copy of the popout-side `.global.tsv` for this cluster. Columns: `sample_id<TAB>ancestry_0<TAB>...<TAB>ancestry_{K_popout-1}`.

### 1.4 `labels.json`

Output of the label-alignment step. Same shape as the FLARE-side `soft_correlation/labels.json` so downstream code can read either. Keys: `tool` (`"popout"`), `rf_ref_labels`, `popout_to_rf_label`, `rf_to_popout_components`, `correlations`, `slope_matrix`, `max_cal_matrix`, `merge_group_stats`, `n_overlapping_sites`.

### 1.5 `global/pairwise_hard/popout_vs_<tool>.confusion.tsv`

Hard-call confusion. Rows = popout dominant ancestry, columns = other tool's dominant ancestry, plus `total` row and column. Header: `popout_label<TAB>{other_tool_labels...}<TAB>total`.

### 1.6 `global/pairwise_soft/popout_vs_<tool>.metrics.tsv`

Per-ancestry concordance. One row per (popout, other-tool) ancestry pair after label alignment. Columns:

| col | type | notes |
|---|---|---|
| `popout_label` | str | aligned popout ancestry name |
| `other_label` | str | the other tool's ancestry name |
| `n_samples_compared` | int | non-null intersection |
| `pearson_r` | float | per-sample over the proportion column |
| `ccc` | float | Lin's concordance correlation coefficient |
| `mae_mean` | float | mean absolute error |
| `mae_median` | float | median |
| `mae_p95` | float | 95th-percentile |
| `jaccard_0.10` | float | Jaccard@τ=0.10 |
| `jaccard_0.25` | float | Jaccard@τ=0.25 |
| `jaccard_0.50` | float | Jaccard@τ=0.50 |
| `pass` | bool / null | μ-gated pass (null when cluster_μ < 0.01) |

### 1.7 `global/pairwise_soft/per_sample_mae.tsv`

One row per sample. Columns: `sample_id`, `mae_vs_flare`, `mae_vs_rye`, `mae_vs_rf`. Columns for absent tools are present but filled with empty strings (stable schema for cohort collation).

### 1.8 `global/pairwise_soft/summary.json`

Cohort-shaped summary even at the per-cluster level. Keys: `pairs` (list of `{popout_label, other_tool, ccc, pearson_r, mae_mean, pass}`), `n_pairs_passing`, `n_pairs_failing`, `n_pairs_null` (μ-gated). One entry per `(tool, ancestry-pair)`.

### 1.9 `local/` (mode = global_local only)

#### `selected_samples.tsv`

Output of the stratified sampler. Columns:

| col | type | notes |
|---|---|---|
| `sample_id` | str | |
| `bucket` | str | `high_<ancestry>` or `mixed` |
| `popout_dominant_anc` | str | argmax label |
| `popout_max_prop` | float | max proportion |

#### `local_per_sample.tsv`

One row per `(sample, chrom)`. Columns: `sample`, `chrom`, `n_sites_compared`, `agree_pct` (per-site agreement after `align_sites(strategy="project_a_onto_b")`), `jaccard_tracts`.

#### `local_per_haplotype.tsv`

One row per `(sample, hap)`. Columns: `sample`, `hap`, `chrom`, `agree_pct`, `per_ancestry_r2` (JSON-encoded dict `{label: r²}`).

#### `local_summary.json`

Per-chrom summary using the vocabulary defined in `popout/diagnostics/GLOSSARY.md`. Keys:

| Key | Type | Notes |
|---|---|---|
| `bp_agreement` | float | fraction of bp where popout and FLARE call the same RF-aligned label |
| `calibration_drift_fraction` | float | bp share of disagreements that do **not** resolve at any coarser grid (≠ boundary error) |
| `boundary_localization_error_fraction` | float | bp share of disagreements where the same label-pair appears within 5 Mb on both tools |
| `per_ancestry_r2_mean` | object | `{label: r²}` averaged across haplotypes |

#### `local/views/bp_confusion_segments.tsv.gz`

Output of `crosstool_merge_walk`. One row per `(start_bp, end_bp, sample, hap, flare_anc, popout_anc)` overlap segment. The compact "View A / View B / View C" inputs are all derived from this single artifact off-line.

#### `local/views/boundary_localization.tsv`

One row per FLARE switch. Columns: `sample`, `hap`, `chrom`, `flare_switch_pos`, `flare_left_label`, `flare_right_label`, `nearest_popout_switch_pos`, `distance_bp`, `flanking_label_match` (bool).

#### `local/views/coarse_grid_summary.tsv`

One row per `(sample, hap, chrom, resolution_mb)`. Columns: `sample`, `hap`, `chrom`, `resolution_mb`, `diagonal_fraction`, `off_diagonal_label_pairs` (JSON-encoded list of `{from, to, fraction}`).

### 1.10 `provenance/`

- `schema_version.txt` — single line `1.0.0`. Mirrors `manifest.json["schema_version"]`.
- `dx_config.json` — the exact JSON config consumed by `discover_runs`, copied byte-for-byte for reproducibility.

---

## 2. Cohort bundle

Produced by Stage 2. One tarball per run.

**Tarball filename:** `cohort_dx.<run_name>.v<schema_version>.tar.gz`

**Top-level prefix:** `cohort_dx/`

```
cohort_dx/
├── cohort_manifest.json
├── cohort_summary.json
├── cohort/
│   ├── manifest.tsv                            (one row per (cluster_id, chrom): n_samples, mode, tools, wallclock, peak_rss, ...)
│   ├── tier1_metrics.tsv                       (long-form: cluster_id, chrom, key, value)
│   ├── per_sample_mae.tsv                      (long-form: cluster_id, chrom, sample_id, mae_vs_flare, ...)
│   ├── pairwise_soft_summary.tsv               (long-form unpivot of every per-cluster summary.json entry)
│   ├── popout_vs_flare.{confusion,metrics}.tsv  (★ gated)
│   ├── popout_vs_rye.{confusion,metrics}.tsv    (★ gated)
│   ├── popout_vs_rf.{confusion,metrics}.tsv     (★ gated)
│   ├── local_per_sample.tsv                    (★ gated on local_mode)
│   ├── bp_confusion_segments.tsv.gz            (★ gated)
│   ├── boundary_localization.tsv               (★ gated)
│   └── coarse_grid_summary.tsv                 (★ gated)
└── per_cluster/                                (optional; unpacked per-cluster artifacts)
    └── <cluster_id>/<chrom>/                    (full tree as in §1)
```

### 2.1 `cohort_manifest.json`

Single JSON. Keys: `schema_version`, `run_name`, `mode`, `tools`, `n_clusters`, `n_chroms`, `n_artifacts`, `cluster_ids`, `chroms`, `generated_at`, `sha256_per_artifact` (`{artifact_path: sha256}`).

### 2.2 `cohort_summary.json`

Roll-ups across the cohort. Keys: `mean_ccc_per_pair` (`{tool: {popout_label: float}}`), `fraction_clusters_passing_per_pair`, `mean_bp_agreement` (gated on local_mode), `mean_calibration_drift_fraction` (gated).

### 2.3 `cohort/*.tsv`

Every per-cluster TSV is concatenated long-form with `cluster_id` and `chrom` columns prepended. Empty-column convention from §1.7 is preserved so absent-tool rows are explicit, not missing.
