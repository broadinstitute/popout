# FLARE validation artifact schema

Schema version: **2.0.0** (semver — bump on any layout change)

## Version history

- **2.0.0** — Mothball R6 ref/target site-concordance audit. The check was a panel-validation question, not a FLARE-output question — running it on every (cluster, chrom) was redundant once FLARE has emitted a usable .anc.vcf.gz. Removed files: `provenance/ref_target_concordance.{tsv,json}` (per-cluster) and `cohort/ref_target_concordance.tsv`. Removed Tier-1 row: `flare_validate.ref_target_exact_overlap_pct`. Removed dashboard dimension: `ref_target_concordance`. The script and baseline meta are preserved under `validation/scripts/_mothballed/panel_validation/` for a future standalone panel-validation pipeline.
- **1.1.0** — Adopt "Rye" naming (was ADMIXTURE); admixture_q optional gate renamed to `rye_q`. Add R6 ref/target concordance audit (`provenance/ref_target_concordance.{tsv,json}`). Expand R10 concordance (`concordance/concordance_metrics.tsv`, `concordance/concordance_summary.json`, `concordance/rye_*` family) with Pearson/CCC/cosine/MAE/Jaccard@τ per ancestry. Add three Tier-1 metrics (`global_ccc`, `ccc_<label>`, `ref_target_exact_overlap_pct`). Cohort bundle gains `cohort/concordance_metrics.tsv` and `cohort/ref_target_concordance.tsv`.
- **1.0.0** — Initial schema.

This file is the contract between the three stages of FLARE validation:

| Stage | Producer | Consumer |
|---|---|---|
| 1. per-cluster validation (WDL scatter) | `validation/scripts/run_cluster_validation.py` | Stage 2, debugging, ad-hoc downloads |
| 2. cohort collation (WDL gather) | `validation/scripts/collate_runs.py` | Stage 3, leaderboards |
| 3. reporting (local) | `validation/scripts/build_flare_validation_report.py` | humans |

If you change a file's columns, you bump the schema. If you add a new optional file, you bump the minor. If you remove or rename a file, you bump the major.

The schema lives in two places: this Markdown doc (the prose contract) and `validation/schema.py` (the executable dataclasses + tarball validator).

---

## 1. Per-cluster artifact tarball

Produced by Stage 1. One tarball per `(cluster_id, chrom)` pair.

**Tarball filename:** `<cluster_id>.<chrom>.validation.v<schema_version>.tar.gz`

**Top-level prefix inside the tarball:** `<cluster_id>/<chrom>/`

```
<cluster_id>/<chrom>/
├── manifest.json
├── tier1_metrics.tsv
├── global.tsv                          (popout-format per-sample ancestry — feeds cohort_global.tsv)
├── coverage/
│   ├── coverage_check.tsv
│   ├── per_chrom_consistency.tsv
│   └── per_chrom_consistency.png
├── model/
│   ├── mu_vs_global_diff.json
│   ├── fst_matrix.tsv                  (optional — FLARE-source artifacts skip this; see §1.4)
│   └── fst_tree.png                    (optional — paired with fst_matrix.tsv)
├── soft_correlation/
│   ├── labels.json
│   ├── rf_soft_correlation.tsv
│   ├── rf_merged_groups.tsv
│   └── popout_composition.tsv
├── confusion/
│   ├── rf_confusion_matrix.tsv
│   └── pca_by_rf_label.png
├── concordance/
│   ├── SUMMARY.md
│   ├── concordance_metrics.tsv          (★ v1.1 — per-ancestry r/CCC/cosine/MAE/Jaccard, gated on rye_q)
│   ├── concordance_summary.json         (★ v1.1 — global_ccc + pass/fail label lists, gated on rye_q)
│   ├── rye_full_matrix.tsv              (★ v1.1, optional — present iff rye_q provided; replaces v1.0 soft_correlation/admixture_correlations.tsv)
│   ├── rye_merged_groups.tsv            (★ v1.1, optional — per-ancestry summary)
│   ├── rye_confusion_matrix.tsv         (★ v1.1, optional — FLARE primary vs Rye primary)
│   ├── rye_scatter_<label>.png          (★ v1.1, optional — per RF label with mu ≥ 0.01)
│   └── rye_admixture_comparison.png     (★ v1.1, optional — replaces v1.0 soft_correlation/admixture_scatter.png)
├── calibration/
│   ├── slope_matrix.tsv
│   ├── calibration_curves.png
│   ├── calibration_slope_matrix.png
│   ├── soft_proportion_hexbin.png
│   ├── merged_confusion_matrix.png
│   ├── correlation_heatmap.png
│   ├── concordance_vs_confidence.png
│   ├── residual_violin.png
│   ├── l1_distance_violin.png
│   ├── entropy_scatter.png
│   ├── admixture_comparison.png
│   ├── calibration_<rf_label>_breakdown.png   (zero or more; per RF label with K_sub > 1)
│   └── notes.txt                         (present always; explains presence/absence of probs=true data)
├── structural/
│   ├── tract_length_summary.json
│   ├── switch_rate_summary.json
│   ├── tract_length_distribution.png
│   ├── switch_rate_distribution.png
│   └── switch_rate_distribution_log.png
├── hap_disagreement/
│   ├── per_sample.tsv
│   ├── summary.json
│   ├── by_rf_label.png
│   ├── per_region.tsv                    (optional — present iff region_bed supplied)
│   └── by_region.png                     (optional)
├── regional/
│   ├── windows.tsv.gz
│   ├── significant.bed
│   ├── summary.json
│   └── regional_qc_<chrom>.png
├── self_id/                              (optional — present iff self_id_tsv provided)
│   ├── check.tsv
│   └── summary.json
└── provenance/
    ├── flare_command_line.txt
    ├── flare_log_tail.txt
    ├── flare_qc.tsv                                       (optional — present iff flare_qc_tsv provided)
    ├── input_vcf_header.txt
    └── schema_version.txt
```

### 1.1 `manifest.json`

Single JSON object. Written by the orchestrator. Keys:

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | `"1.1.0"` |
| `cluster_id` | str | e.g. `"cluster_007"`, `"null_cluster"` |
| `chrom` | str | e.g. `"chr1"` |
| `run_name` | str | `magicwand` run name, includes pipeline run id |
| `flare_version` | str | parsed from FLARE log |
| `flare_command_line` | str | exact CLI used to invoke FLARE |
| `panel_id` | str \| null | ref-panel identifier (e.g. `"gnomad_90"`); null if unknown |
| `ref_panel_sha` | str \| null | sha256 of the ref-panel TSV used |
| `input_vcf_sha` | str | sha256 of the per-cluster input gt= VCF |
| `n_samples` | int | sample count in `global.tsv` |
| `n_markers` | int | markers parsed from FLARE log |
| `n_ancestries` | int | FLARE K |
| `coverage_passed` | bool | true iff all rows in `coverage/coverage_check.tsv` are PASS |
| `steps` | object | per-step `{wallclock_seconds: float, peak_rss_gb: float, exit: int, status: "ok"|"skipped"|"failed"}` |
| `total_wallclock_seconds` | float | sum of per-step wallclocks |
| `peak_rss_gb` | float | max of per-step peak_rss_gb |
| `cpu_wall_ratio` | float | total CPU seconds / total wallclock |
| `generated_at` | str | ISO 8601 UTC, e.g. `"2026-05-27T14:33:12Z"` |
| `optional_inputs` | object | `{rye_q: bool, self_id: bool, popout_secondary: bool, region_bed: bool, fst_tree: bool, flare_qc_tsv: bool}` (★ v1.1: `rye_q` replaces `admixture_q`; `flare_qc_tsv` added — false ⇒ qc-dependent coverage checks SKIP) |

### 1.1b `global.tsv`

Popout-format per-sample ancestry table written by `flare_to_popout_format.py`. Columns: `sample_id<TAB>ancestry_0<TAB>...<TAB>ancestry_K-1`. K is the FLARE-final ancestry count for this cluster (typically 5). Copied verbatim to the artifact root by the orchestrator so the collator can build `cohort_global.tsv` without re-running the format conversion.

### 1.2 `tier1_metrics.tsv`

Two-column TSV: `key<TAB>value`. Consumed by the WDL command block, which emits one `magicwand log <key>=<value>` line per row. The v1.0 set (16 rows) plus three v1.1 additions (★) plus per-ancestry CCC rows:

```
flare_validate.cluster_id
flare_validate.chrom
flare_validate.n_samples
flare_validate.n_markers
flare_validate.coverage_pass
flare_validate.merged_r_afr
flare_validate.merged_r_amr
flare_validate.merged_r_eas
flare_validate.merged_r_eur
flare_validate.merged_r_sas
flare_validate.calibration_slope_max_dev
flare_validate.switch_rate_p99
flare_validate.hap_disagreement_mean
flare_validate.regional_significant_n
flare_validate.peak_rss_gb
flare_validate.cpu_wall_ratio
flare_validate.global_ccc                       ★ v1.1 — cohort-wide CCC from concordance_summary.json
flare_validate.ccc_<label>                      ★ v1.1 — one row per ancestry in {afr,amr,eas,eur,sas}; NA when cluster_mu < 0.01
```

`cluster_id` and `chrom` are string-typed informational keys; the remaining rows are float/int/bool signals tuneable from the dashboard.

### 1.3 `coverage/`

#### `coverage_check.tsv`

Long-form. One row per check. Columns:

| col | type | notes |
|---|---|---|
| `check` | str | one of: `input_set_equals_output_set`, `qc_sample_count_consistent`, `output_site_count_matches_log`, `site_coverage_ge_95pct_of_intersection` |
| `status` | str | `PASS` or `FAIL` |
| `detail` | str | human-readable detail |

#### `per_chrom_consistency.tsv`

One row per chrom. Columns: `chrom`, `gt_records` (int), `out_records` (int). On a per-chrom validation task this has 1 row; rolled up at collation time across the cluster's chroms.

### 1.4 `model/`

#### `mu_vs_global_diff.json`

NEW emission from `validate_structural.py`. Object with keys:

| Key | Type | Notes |
|---|---|---|
| `max_abs_diff` | float | max(|global_mu_i − model_mu_i|) |
| `per_ancestry` | array of object | `{ancestry: int, name: str, global_mu: float, model_mu: float, abs_diff: float, pass: bool}` |
| `threshold` | float | the pass threshold used (0.01) |
| `overall_pass` | bool | all per-ancestry abs_diff < threshold |

#### `fst_matrix.tsv` *(optional)*

Square Hudson F_ST matrix. Header: `ancestry<TAB>name_0<TAB>...<TAB>name_K-1<TAB>mu`. **Skipped for FLARE-source artifacts in v1.0.0**: FLARE's `.model` file stores HMM parameters (T, μ, per-panel p/θ matrices) but not per-site ancestral allele frequencies, which the Hudson F_ST formula requires. Re-introducing this for FLARE means re-deriving allele frequencies from the reference VCF, which is out of scope for V1. The orchestrator's `manifest.json["optional_inputs"]["fst_tree"]` flag records whether the step ran. Popout-source artifacts (if this schema is reused for popout validation later) emit it normally.

### 1.5 `soft_correlation/`

#### `labels.json`

Direct copy from `compare_to_rf.py`. Object with keys: `tool` (`"FLARE"` or `"popout"`), `rf_ref_labels`, `popout_to_rf_label`, `rf_to_popout_components`, `n_overlapping_sites`, `correlations` (K × n_rf matrix), `slope_matrix`, `max_cal_matrix`, `merge_group_stats`.

#### `rf_soft_correlation.tsv`

Renamed from `compare_to_rf.py`'s `soft_correlation.tsv`. Columns: `popout_ancestry<TAB>afr<TAB>amr<TAB>eas<TAB>eur<TAB>mid<TAB>sas` (Pearson r per popout ancestry × each RF reference label).

#### `rf_merged_groups.tsv`

NEW. Derived by the orchestrator from `labels.json["merge_group_stats"]`. Long-form, one row per RF label. Columns: `rf_label<TAB>merged_r<TAB>summed_mu<TAB>component_indices<TAB>component_names`. `component_*` columns are comma-joined lists.

#### `popout_composition.tsv`

Direct copy from `compare_to_rf.py`. Per popout ancestry, the mean RF probability vector over high-confidence samples (FLARE posterior > 0.8). Columns: `popout_ancestry<TAB>n_samples<TAB>afr<TAB>amr<TAB>eas<TAB>eur<TAB>mid<TAB>sas`.

### 1.6 `confusion/`

#### `rf_confusion_matrix.tsv`

Renamed from `compare_to_rf.py`'s `confusion_matrix.tsv`. Rows = RF hard label (incl. `mixed`), columns = popout ancestries (named), plus a `total` row and column. Header: `rf_label<TAB>{popout_names...}<TAB>total`.

### 1.7 `concordance/`

#### `SUMMARY.md`

Direct copy from `compare_to_rf.py`. Markdown verdict text; consumed verbatim by the report builder.

#### `concordance_metrics.tsv` *(★ v1.1, optional: gated on `rye_q`)*

Long-form, one row per ancestry in the canonical Rye column set (`eur, eas, amr, afr, sas`). Columns:

| col | type | notes |
|---|---|---|
| `ancestry` | str | one of `eur`, `eas`, `amr`, `afr`, `sas` |
| `cluster_mu` | float | mean of this cluster's FLARE proportion on this ancestry; gates whether the row is a meaningful test |
| `n_samples` | int | size of the FLARE∩Rye intersection used for these stats |
| `pearson_r` | float | Pearson r between FLARE and Rye on this ancestry |
| `ccc` | float | Lin's concordance correlation coefficient |
| `cosine_mean` | float | mean of per-sample cosine similarity restricted to this ancestry column (NA if degenerate) |
| `mae_mean` | float | mean(|x − y|) |
| `mae_median` | float | median(|x − y|) |
| `mae_p95` | float | 95th percentile(|x − y|) |
| `jaccard_at_0.10` | float | Jaccard index of {samples with proportion ≥ 0.10} |
| `jaccard_at_0.25` | float | … at threshold 0.25 |
| `jaccard_at_0.50` | float | … at threshold 0.50 |
| `pass` | bool \| null | `true` iff `pearson_r ≥ 0.95 AND ccc ≥ 0.90`; `null` when `cluster_mu < 0.01` (degenerate non-test) |

#### `concordance_summary.json` *(★ v1.1, optional)*

| Key | Type | Notes |
|---|---|---|
| `global_ccc` | float | Lin's CCC on the flattened (n_samples × K) FLARE vs Rye matrices |
| `mean_pearson_r` | float | mean of per-ancestry pearson_r across labels with `cluster_mu ≥ 0.01` |
| `n_samples_overlap` | int | size of the FLARE∩Rye sample intersection |
| `labels_passing_r_ge_0.95` | array[str] | ancestries (with cluster_mu ≥ 0.01) where Pearson r ≥ 0.95 |
| `labels_passing_ccc_ge_0.90` | array[str] | … where CCC ≥ 0.90 |
| `labels_failing` | array[str] | ancestries (with cluster_mu ≥ 0.01) where either r < 0.95 OR CCC < 0.90 |

#### `rye_full_matrix.tsv` *(★ v1.1, optional)*

Pearson r per FLARE ancestry × Rye ancestry. Header: `flare_ancestry<TAB>eur<TAB>eas<TAB>amr<TAB>afr<TAB>sas`.

#### `rye_merged_groups.tsv` *(★ v1.1, optional)*

Long-form, one row per ancestry. Columns: `rf_label<TAB>n_samples<TAB>cluster_mu<TAB>pearson_r<TAB>ccc`. Direct collation source for `cohort/merged_groups_rye.tsv` if needed; otherwise consumed by the report layer.

#### `rye_confusion_matrix.tsv` *(★ v1.1, optional)*

Hard primary-call confusion matrix. Rows = FLARE primary (argmax of FLARE proportions), columns = Rye primary (argmax of Rye proportions), value = sample count. Plus `total` row and column.

#### `rye_scatter_<label>.png`, `rye_admixture_comparison.png` *(★ v1.1, optional)*

Per-ancestry FLARE-vs-Rye scatter (one per ancestry with `cluster_mu ≥ 0.01`) and a single mean-bar comparison plot. Layered into the report by Stage 3.

### 1.8 `calibration/`

#### `slope_matrix.tsv`

Direct copy from `plot_concordance.py`'s `calibration_slope_matrix.tsv`. Columns: `ancestry<TAB>{rf_label}_slope...<TAB>{rf_label}_max...`.

#### `notes.txt`

Plain text. One line: either `probs=true, per-bin calibration curves available` or `probs=false (FLARE default); calibration curves derived from hard calls only — slopes are still computed but error bars may be inflated`. Always present so the consumer can branch without checking for file existence.

### 1.9 `structural/`

#### `tract_length_summary.json`

NEW. Object with keys:

| Key | Type | Notes |
|---|---|---|
| `n_tracts_total` | int | |
| `per_ancestry` | array of object | `{ancestry: int, name: str, n_tracts: int, mean_Mb: float, median_Mb: float, exp_fit_rate: float \| null, implied_T_gen: float \| null, model_T_gen: float}` |
| `note` | str | If `exp_fit_rate` is null, this field explains why (e.g. "n_tracts < 100"). |

#### `switch_rate_summary.json`

NEW. Object with keys:

| Key | Type | Notes |
|---|---|---|
| `n_haplotypes` | int | |
| `mean` | float | switches per haplotype |
| `median` | float | |
| `p99` | float | 99th percentile |
| `min` | int | |
| `max` | int | |
| `histogram` | array of `{bin_lo, bin_hi, count}` | bins from the script's existing `bins_summary` (0, 3, 10, 20, 50, 100, max+1) |

### 1.10 `hap_disagreement/`

#### `per_sample.tsv`

Direct copy from `validate_hap_disagreement.py`'s `hap_disagreement.per_sample.tsv`. Columns: `sample_id<TAB>rf_hard_label<TAB>agreement_bp_frac<TAB>disagreement_bp_frac<TAB>total_bp<TAB>dominant_anc_h1<TAB>dominant_anc_h2`.

#### `summary.json`

NEW. Object with keys:

| Key | Type | Notes |
|---|---|---|
| `cohort_mean_disagreement` | float | bp-weighted mean across samples |
| `n_samples` | int | total |
| `n_samples_unjoined` | int | samples without an RF label |
| `per_rf_label` | array of object | `{rf_label: str, n: int, mean: float, median: float}` |

#### `per_region.tsv` *(optional)*

Direct copy from `hap_disagreement.per_region.tsv`. Present iff `--region-bed` supplied. Columns: `sample_id<TAB>rf_hard_label<TAB>region<TAB>chrom<TAB>start<TAB>end<TAB>agreement_bp_frac<TAB>disagreement_bp_frac<TAB>total_bp`.

### 1.11 `regional/`

#### `windows.tsv.gz`

Direct copy from `validate_regional.py`'s `regional_windows.tsv.gz`. Columns: `chrom<TAB>start<TAB>end<TAB>ancestry_name<TAB>mean_anc<TAB>z<TAB>p<TAB>q<TAB>mask_region`. Gzipped.

#### `significant.bed`

Direct copy from `regional_significant.bed`. 4-column BED: `chrom<TAB>start<TAB>end<TAB>name`. `name` encodes ancestry, z, q, and mask overlap.

#### `summary.json`

NEW. Object with keys:

| Key | Type | Notes |
|---|---|---|
| `n_windows_total` | int | |
| `n_windows_significant` | int | BH-FDR q < 0.05 |
| `fdr_q_threshold` | float | the q used (default 0.05) |
| `per_ancestry` | array of object | `{ancestry: int, name: str, n_significant: int, peak_window: {chrom, start, end, z, q, mask_region} \| null}` |
| `hla_overlap_n` | int | count of significant windows overlapping any mask named `hla` |
| `centromere_overlap_n` | int | count overlapping `centromere` |
| `segdup_overlap_n` | int | count overlapping `segdup` |
| `high_ld_overlap_n` | int | count overlapping `high_ld` |
| `outside_mask_n` | int | count overlapping no mask |

### 1.12 `self_id/` *(optional)*

#### `check.tsv`

Long-form. Columns: `self_id<TAB>n<TAB>ancestry<TAB>name<TAB>mean_mu`. One row per (self_id_class, ancestry).

#### `summary.json`

Object with keys: `n_samples_joined` (int), `n_self_id_classes` (int), `per_class` (array of `{self_id: str, n: int, dominant_ancestry_name: str, dominant_mean_mu: float}`).

### 1.13 `provenance/`

| File | Content |
|---|---|
| `flare_command_line.txt` | Plain text. Exact CLI used to invoke FLARE, parsed from the log. |
| `flare_log_tail.txt` | Last ~200 lines of the FLARE log. |
| `flare_qc.tsv` | *(optional, gated on `flare_qc_tsv`)* Copy of the in-WDL `qc.tsv`. |
| `input_vcf_header.txt` | Output of `bcftools view -h` on the input gt= VCF. |
| `schema_version.txt` | Plain text. One line: the schema version, mirrored from `manifest.json`. |

---

## 2. Cohort bundle

Produced by Stage 2. One bundle per Stage 2 run.

**Tarball filename:** `cohort_bundle.<run_name>.v<schema_version>.tar.gz`

```
cohort_bundle/
├── cohort_manifest.json
├── cohort_summary.json
├── cohort_qc_dashboard.json
├── cohort/
│   ├── cohort_global.tsv
│   ├── coverage.tsv
│   ├── manifest.tsv
│   ├── tier1_metrics.tsv
│   ├── soft_correlation_rf.tsv
│   ├── merged_groups_rf.tsv
│   ├── confusion_rf.tsv
│   ├── calibration_slope.tsv
│   ├── tract_length_stats.tsv
│   ├── switch_rate_stats.tsv
│   ├── hap_disagreement.tsv
│   ├── regional_windows.tsv.gz
│   ├── regional_meta.tsv
│   ├── concordance_metrics.tsv          (★ v1.1 — long unpivot of per-cluster concordance/concordance_metrics.tsv)
│   ├── fst_matrix.tsv                    (optional — popout-source artifacts only)
│   └── self_id.tsv                       (optional)
└── per_cluster/
    └── <cluster_id>/
        └── <chrom>/                       (un-tarred per-cluster artifacts, verbatim)
```

### 2.1 `cohort_manifest.json`

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | matches the per-cluster artifacts |
| `run_name` | str | from collation config |
| `collation_mode` | str | `single_run` \| `diff_runs` \| `leaderboard` |
| `n_clusters` | int | distinct cluster_ids |
| `n_chroms` | int | distinct chroms |
| `n_artifacts` | int | total per-cluster tarballs collated |
| `cluster_ids` | array[str] | |
| `chroms` | array[str] | |
| `generated_at` | str | ISO 8601 UTC |
| `sha256_per_artifact` | object | `{<cluster_id>.<chrom>: sha256str}` |
| `diff_against` | str \| null | path/name of the cohort bundle this was diffed against |
| `collation_config` | object | the input config, echoed verbatim |

### 2.2 Cohort long-form tables

All tables under `cohort/` are long-form with `cluster_id` and `chrom` as the first two columns (except `cohort_global.tsv`, which is per-sample). Schema for each table:

#### `cohort_global.tsv`

Per-sample. Columns: `cluster_id<TAB>chrom<TAB>sample_id<TAB>ancestry_0...<TAB>ancestry_K-1`. K per row may differ across clusters (FLARE K varies); use the cluster's `manifest.json` to interpret column count.

#### `coverage.tsv`

Columns: `cluster_id<TAB>chrom<TAB>check<TAB>status<TAB>detail`.

#### `manifest.tsv`

One row per (cluster_id, chrom). Flat columns from manifest.json: `cluster_id<TAB>chrom<TAB>n_samples<TAB>n_markers<TAB>n_ancestries<TAB>coverage_passed<TAB>total_wallclock_seconds<TAB>peak_rss_gb<TAB>cpu_wall_ratio<TAB>flare_version<TAB>panel_id<TAB>generated_at`.

#### `tier1_metrics.tsv`

Concatenation. Columns: `cluster_id<TAB>chrom<TAB>key<TAB>value`.

#### `soft_correlation_rf.tsv`

Columns: `cluster_id<TAB>chrom<TAB>flare_ancestry<TAB>rf_label<TAB>r`. Long unpivot of each cluster's `rf_soft_correlation.tsv`.

#### `merged_groups_rf.tsv`

Columns: `cluster_id<TAB>chrom<TAB>rf_label<TAB>merged_r<TAB>summed_mu<TAB>component_indices<TAB>component_names`. Direct concatenation of per-cluster `rf_merged_groups.tsv`.

#### `concordance_metrics.tsv` *(★ v1.1)*

Long unpivot of per-cluster `concordance/concordance_metrics.tsv` with `cluster_id` and `chrom` prepended. Columns:

`cluster_id<TAB>chrom<TAB>ancestry<TAB>cluster_mu<TAB>n_samples<TAB>pearson_r<TAB>ccc<TAB>cosine_mean<TAB>mae_mean<TAB>mae_median<TAB>mae_p95<TAB>jaccard_at_0.10<TAB>jaccard_at_0.25<TAB>jaccard_at_0.50<TAB>pass`

`pass` is empty string when degenerate (cluster_mu < 0.01). Rows from clusters where rye_q wasn't provided are absent.

#### `confusion_rf.tsv`

Columns: `cluster_id<TAB>chrom<TAB>rf_label<TAB>flare_call<TAB>n`. Long unpivot of each cluster's `rf_confusion_matrix.tsv`.

#### `calibration_slope.tsv`

Columns: `cluster_id<TAB>chrom<TAB>ancestry_name<TAB>rf_label<TAB>slope<TAB>max_cal`. Long unpivot of `slope_matrix.tsv` (one row per ancestry × RF label).

#### `tract_length_stats.tsv`

Columns: `cluster_id<TAB>chrom<TAB>ancestry<TAB>ancestry_name<TAB>n_tracts<TAB>mean_Mb<TAB>median_Mb<TAB>exp_fit_rate<TAB>implied_T_gen<TAB>model_T_gen`.

#### `switch_rate_stats.tsv`

Columns: `cluster_id<TAB>chrom<TAB>n_haplotypes<TAB>mean<TAB>median<TAB>p99<TAB>min<TAB>max`.

#### `hap_disagreement.tsv`

Columns: `cluster_id<TAB>chrom<TAB>rf_label<TAB>n<TAB>mean<TAB>median`. Long unpivot of `summary.json["per_rf_label"]`.

#### `regional_windows.tsv.gz`

Concatenation of per-cluster `regional/windows.tsv.gz` with `cluster_id<TAB>chrom_validated` prepended (where `chrom_validated` is the validation task's chrom — the `chrom` column inside the windows file is the per-window chrom, which equals `chrom_validated` in v1.0.0 since each task validates one chrom).

#### `regional_meta.tsv`

Cross-cluster meta-analysis. One row per (window). Columns: `chrom<TAB>start<TAB>end<TAB>ancestry_name<TAB>n_clusters_flagged<TAB>n_clusters_total<TAB>stouffer_z<TAB>stouffer_p<TAB>stouffer_q<TAB>mask_region`.

#### `fst_matrix.tsv`

Long-form. Columns: `cluster_id<TAB>chrom<TAB>ancestry_i<TAB>ancestry_j<TAB>name_i<TAB>name_j<TAB>fst<TAB>mu_i<TAB>mu_j`.

#### `self_id.tsv` *(optional)*

Columns: `cluster_id<TAB>chrom<TAB>self_id<TAB>n<TAB>ancestry<TAB>name<TAB>mean_mu`. Present iff at least one cluster had `self_id/`.

### 2.3 `cohort_summary.json`

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | |
| `run_name` | str | |
| `n_clusters` | int | |
| `n_artifacts` | int | |
| `n_clusters_pass_coverage` | int | clusters where ALL their (cluster, chrom) artifacts have `coverage_passed: true` |
| `mean_merged_r_per_rf_label` | object | `{afr: float, amr: float, eas: float, eur: float, mid: float, sas: float}` — cohort-wide mean |
| `n_clusters_with_hla_flagged` | int | regional summary's `hla_overlap_n > 0` |
| `n_regional_outliers_outside_mask` | int | sum of `outside_mask_n` across clusters |
| `traffic_light_per_cluster` | object | `{<cluster_id>: "green"|"yellow"|"red"}` |
| `total_wallclock_hours` | float | sum across all tasks |
| `total_peak_rss_gb_max` | float | max peak_rss_gb across all tasks |

### 2.4 `cohort_qc_dashboard.json`

| Key | Type | Notes |
|---|---|---|
| `schema_version` | str | |
| `dimensions` | array[str] | `["coverage", "calibration", "concordance", "structural", "hap_disagreement", "regional"]` |
| `per_cluster` | object | `{<cluster_id>: {<dimension>: "green"|"yellow"|"red"}}` |
| `thresholds` | object | echoed from collation config (e.g. `calibration_slope_outside: [0.85, 1.15]`) |

---

## 3. Stage 3 reporting

The canonical consumer of the cohort bundle is `validation/scripts/build_flare_validation_report.py`. It produces one PDF rolling up:

- **Cohort front-matter** — run overview, QC traffic-light dashboard (PIL-rendered widget), mean-merged-r bar, per-ancestry Rye concordance box, ref/target site-overlap bars, top-20 regional meta-analysis windows.
- **Per-cluster sections** — header (traffic lights + manifest + Tier-1 metrics), coverage, Rye concordance (gated on `optional_inputs.rye_q`), calibration (figure grid + slope matrix), structural (tract / switch distributions + summary), hap disagreement + regional QC.
- **Provenance appendix** — schema version, sha256 of every per-cluster artifact, diff-against link.

Usage:

```
python validation/scripts/build_flare_validation_report.py \
    --cohort-bundle <dir>                     # unpacked cohort_bundle/
    [--tarball-dir <dir>]                     # per-cluster *.validation.*.tar.gz, used when per_cluster/ is absent
    --out <report.pdf>
    [--clusters cluster_001,cluster_007]
    [--max-clusters 10]
    [--no-per-cluster]
    [--keep-md]
```

Architecture: builds a single markdown document (using existing per-cluster PNGs as embedded assets, PIL for the traffic-light dashboard widget, matplotlib `savefig` for cohort stat charts) and shells out to `pandoc --pdf-engine=xelatex`. The intermediate `.md` is removed unless `--keep-md` is passed; the per-asset PNGs land in `<out>.stem_assets/` next to the PDF.

Required system dependencies: `pandoc`, `xelatex` (any TeX Live distribution).
