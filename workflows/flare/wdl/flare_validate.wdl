version 1.0

## FLARE per-cluster validation — scatter (validate) + gather (collate).
##
## One Cromwell task per (cluster_id, chrom). Each task runs the
## 11-script diagnostic DAG via validation/scripts/run_cluster_validation.py
## and emits a versioned per-cluster artifact tarball matching
## validation/SCHEMA.md §1. A second single task gathers all artifacts
## and emits a cohort bundle matching validation/SCHEMA.md §2.
##
## Reporting (PDF, run-diff, leaderboard) lives outside this WDL — it
## consumes the cohort bundle locally. See my_notes/validation/PLAN.md
## for the architecture.
##
## Observability: magicwand wires per-task metrics to W&B (project
## flare-validate). The orchestrator emits a tier1_metrics.tsv next to
## the tarball; this WDL reads that file and replays it as
## `magicwand log` lines so every per-task metric is queryable from the
## dashboard without untarring artifacts.

struct FlareClusterRun {
  String cluster_id
  String chrom
  File   anc_vcf
  File   global_anc
  File   flare_model
  File   flare_log
  File?  flare_qc_tsv              # ★ v1.1: optional — pre-pipeline fixtures have no qc.tsv
  File?  flare_summary             # FLARE does not emit one today; reserved
  File   input_vcf                 # per-cluster gt VCF; header-only reads in coverage + provenance
}

task validate_cluster {
  input {
    FlareClusterRun cluster_run
    File   rf_ancestry
    File   chrom_sizes
    File?  rye_q                   # ★ v1.1: Rye Q TSV (was admixture_q); cohort-wide singleton
    File?  self_id
    File?  popout_secondary_global
    File?  popout_secondary_labels
    File?  ref_panel               # for sha256-in-manifest only

    String run_name                # magicwand run-name prefix
    String? panel_id
    String schema_version = "3.0.0"   # ★ v3.0.0 (Phase 6 label-space retrofit)

    # Resource overrides (auto-scaled by anc_vcf size by default).
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 0

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  # Auto-scale on anc_vcf size: the .anc.vcf.gz size is monotone in
  # (n_samples × n_markers), which is what drives memory pressure for
  # flare_vcf_to_tracts (step 1, dominant). For the production grid
  # (cluster sizes 500–30k samples × ~600k markers/chr), anc_vcf ranges
  # ~150 MB (null cluster, 506 samp) to ~10+ GB (largest cluster).
  #
  # First-Terra-run calibration (chr1, 506 / 1297 / 1607 sample clusters):
  # peak_rss observed at 0.91 GB across all three, ~8× headroom below the
  # original 8 GB floor and 17× below the 16 GB floor. Dropped the small
  # bucket memory floor to 8 GB. Mid + large buckets stay generous until
  # we see a >5 GB anc_vcf cluster on Terra.
  Float anc_gb = size(cluster_run.anc_vcf, "GB")
  Int bucket_cpu = if anc_gb > 5.0 then 16
                   else if anc_gb > 1.0 then 8
                   else 4
  Int bucket_mem_gb = if anc_gb > 5.0 then 64
                      else if anc_gb > 1.0 then 32
                      else 8
  Int bucket_disk_gb = ceil(anc_gb * 4) + 20

  # Predicted sizing — used only when it exceeds the bucket value.
  # The orchestrator's peak working set is the union of step 1's tract
  # streaming (~3× anc_gb in pysam decompressed buffer) and step 4's
  # numpy correlation matrices (~1.5× n_samples × K). Empirical
  # multiplier from the first Terra run: peak_rss ≈ 0.91 GB at anc_gb
  # ≈ 0.15 GB (null cluster), so the linear coefficient is ~6× anc_gb
  # plus a ~1 GB baseline. Tightened from the pre-run guess of 4.5× + 8 GB.
  Float predicted_mem_gb = anc_gb * 6.0 + 1.0
  Int   sized_mem_gb     = ceil(predicted_mem_gb * 1.5)   # 50% headroom
  Int   sized_cpu        = ceil(sized_mem_gb * 1.0 / 8.0)

  Int    auto_cpu       = if sized_cpu    > bucket_cpu    then sized_cpu    else bucket_cpu
  Int    auto_mem_int   = if sized_mem_gb > bucket_mem_gb then sized_mem_gb else bucket_mem_gb
  String auto_memory    = "~{auto_mem_int} GB"

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  Int    disk_size_gb = select_first([disk_size_gb_override, bucket_disk_gb])

  String run_id = "~{run_name}.~{cluster_run.cluster_id}.~{cluster_run.chrom}"
  String out_tarball = "~{cluster_run.cluster_id}.~{cluster_run.chrom}.validation.v~{schema_version}.tar.gz"

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare-validate
    export WANDB_RUN_NAME="~{run_id}"
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    # Static inputs logged upfront — small enough that we want them
    # queryable in the dashboard even if the task crashes mid-DAG.
    magicwand log \
      flare_validate.cluster_id="~{cluster_run.cluster_id}" \
      flare_validate.chrom="~{cluster_run.chrom}" \
      flare_validate.schema_version="~{schema_version}" \
      flare_validate.anc_vcf_bytes="$(stat -c %s ~{cluster_run.anc_vcf})" \
      flare_validate.input_vcf_bytes="$(stat -c %s ~{cluster_run.input_vcf})" \
      flare_validate.cpu="~{cpu}" \
      flare_validate.memory_gb="~{auto_mem_int}" \
      flare_validate.disk_gb="~{disk_size_gb}" \
      flare_validate.disk_type="~{disk_type}" \
      flare_validate.optional_rye_q="~{defined(rye_q)}" \
      flare_validate.optional_flare_qc_tsv="~{defined(cluster_run.flare_qc_tsv)}" \
      flare_validate.optional_self_id="~{defined(self_id)}" \
      flare_validate.optional_popout_secondary="~{defined(popout_secondary_global)}"

    # ---- run the orchestrator ----------------------------------------
    # The orchestrator's DAG runner sizes its thread pool to --max-workers;
    # we pass through the WDL-allocated CPU count so within-task parallelism
    # tracks the resource grant.
    python3 /opt/validation/scripts/run_cluster_validation.py \
      --cluster-id     ~{cluster_run.cluster_id} \
      --chrom          ~{cluster_run.chrom} \
      --anc-vcf        ~{cluster_run.anc_vcf} \
      --global-anc     ~{cluster_run.global_anc} \
      --flare-model    ~{cluster_run.flare_model} \
      --flare-log      ~{cluster_run.flare_log} \
      ~{"--flare-qc-tsv "            + cluster_run.flare_qc_tsv} \
      ~{"--flare-summary "           + cluster_run.flare_summary} \
      --input-vcf      ~{cluster_run.input_vcf} \
      --rf-ancestry    ~{rf_ancestry} \
      --chrom-sizes    ~{chrom_sizes} \
      --region-masks-dir /opt/region_masks \
      ~{"--rye-q "                   + rye_q} \
      ~{"--self-id "                 + self_id} \
      ~{"--popout-secondary-global " + popout_secondary_global} \
      ~{"--popout-secondary-labels " + popout_secondary_labels} \
      ~{"--ref-panel "               + ref_panel} \
      ~{"--panel-id "                + panel_id} \
      --schema-version ~{schema_version} \
      --max-workers    ~{cpu} \
      --run-name       "~{run_id}" \
      --out-tarball    ~{out_tarball}

    # ---- replay tier1 metrics into magicwand -------------------------
    # The orchestrator writes tier1_metrics.tsv inside the tarball; pull
    # it back out for the W&B log. tar -x to a temp dir, no untarring
    # the whole artifact.
    mkdir -p .t1
    tar -xzf ~{out_tarball} -C .t1 --strip-components=2 \
      "~{cluster_run.cluster_id}/~{cluster_run.chrom}/tier1_metrics.tsv"
    while IFS=$'\t' read -r key value; do
      [ -n "$key" ] || continue
      magicwand log "$key=$value"
    done < .t1/tier1_metrics.tsv
    rm -rf .t1

    # Also emit final resource numbers from /proc/self/status — useful
    # for bucket tuning when magicwand's system metrics don't survive.
    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')
    magicwand log flare_validate.task_peak_rss_gb="$PEAK_RSS_GB" || true
  >>>

  output {
    File artifact_tarball = "~{out_tarball}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task collate_cohort {
  input {
    Array[File] cluster_artifacts
    File?       collation_config
    File?       previous_cohort_bundle    # cohort bundle to diff against
    String      run_name
    String      schema_version = "3.0.0"     # ★ v3.0.0 (Phase 6 label-space retrofit)

    # MID-handling rule for cohort/confusion_rf.tsv (Phase 6). FLARE's
    # panel has no MID component; RF emits SP6 including MID. Pick:
    #   "none"        — pass-through (legacy v2 behavior)
    #   "drop"        — drop every RF-MID row from the cohort confusion
    #   "fold_to_eur" — sum MID counts into the EUR row per cluster
    # The chosen rule is recorded in cohort_manifest.json.provenance.mid_rule
    # and surfaces in every figure footer's shorthand tag.
    String      mid_rule       = "none"

    Int     cpu          = 4
    String  memory       = "16 GB"
    Int     disk_size_gb = 50
    String  disk_type    = "HDD"
    Int     preemptible  = 0

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  String out_bundle  = "cohort_bundle.~{run_name}.v~{schema_version}.tar.gz"
  String out_summary = "cohort_summary.~{run_name}.json"

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare-validate
    export WANDB_RUN_NAME="~{run_name}.collate"
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    magicwand log \
      flare_collate.run_name="~{run_name}" \
      flare_collate.n_artifacts="~{length(cluster_artifacts)}" \
      flare_collate.schema_version="~{schema_version}" \
      flare_collate.mid_rule="~{mid_rule}"

    python3 /opt/validation/scripts/collate_runs.py \
      --cluster-artifacts ~{sep=" " cluster_artifacts} \
      ~{"--collation-config " + collation_config} \
      ~{"--diff-against "     + previous_cohort_bundle} \
      --schema-version    ~{schema_version} \
      --mid-rule          ~{mid_rule} \
      --run-name          ~{run_name} \
      --out-bundle        ~{out_bundle} \
      --out-summary       ~{out_summary}

    # Bubble up a few cohort-level metrics for the dashboard.
    python3 -c "
import json, sys
d = json.load(open('~{out_summary}'))
print(f\"flare_collate.n_clusters_pass_coverage={d['n_clusters_pass_coverage']}\")
print(f\"flare_collate.n_clusters_with_hla_flagged={d['n_clusters_with_hla_flagged']}\")
print(f\"flare_collate.n_regional_outliers_outside_mask={d['n_regional_outliers_outside_mask']}\")
print(f\"flare_collate.total_wallclock_hours={d['total_wallclock_hours']}\")
print(f\"flare_collate.total_peak_rss_gb_max={d['total_peak_rss_gb_max']}\")
for label, val in d.get('mean_merged_r_per_rf_label', {}).items():
    print(f\"flare_collate.mean_merged_r_{label}={val:.4f}\")
" | while read line; do magicwand log "$line" || true; done

    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')
    magicwand log flare_collate.peak_rss_gb="$PEAK_RSS_GB" || true
  >>>

  output {
    File cohort_bundle       = "~{out_bundle}"
    File cohort_summary_json = "~{out_summary}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow flare_validate {
  input {
    Array[FlareClusterRun] cluster_runs
    File   rf_ancestry
    File   chrom_sizes
    File?  rye_q                       # ★ v1.1: Rye Q TSV (was admixture_q); cohort singleton
    File?  self_id
    File?  popout_secondary_global
    File?  popout_secondary_labels
    File?  ref_panel
    File?  collation_config
    File?  previous_cohort_bundle

    String run_name
    String? panel_id
    String schema_version = "3.0.0"     # ★ v3.0.0 (Phase 6 label-space retrofit)

    # MID-handling for cohort/confusion_rf.tsv. See collate_cohort task
    # input docstring above. Recommended for the v3 cutover:
    # "fold_to_eur" — preserves the RF-MID signal as an EUR contribution.
    String mid_rule       = "none"

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 0

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  scatter (cr in cluster_runs) {
    call validate_cluster {
      input:
        cluster_run             = cr,
        rf_ancestry             = rf_ancestry,
        chrom_sizes             = chrom_sizes,
        rye_q                   = rye_q,
        self_id                 = self_id,
        popout_secondary_global = popout_secondary_global,
        popout_secondary_labels = popout_secondary_labels,
        ref_panel               = ref_panel,
        run_name                = run_name,
        panel_id                = panel_id,
        schema_version          = schema_version,
        cpu_override            = cpu_override,
        memory_override         = memory_override,
        disk_size_gb_override   = disk_size_gb_override,
        disk_type               = disk_type,
        preemptible             = preemptible,
        wandb_api_key           = wandb_api_key,
        docker_image            = docker_image,
    }
  }

  call collate_cohort {
    input:
      cluster_artifacts      = validate_cluster.artifact_tarball,
      collation_config       = collation_config,
      previous_cohort_bundle = previous_cohort_bundle,
      run_name               = run_name,
      schema_version         = schema_version,
      mid_rule               = mid_rule,
      wandb_api_key          = wandb_api_key,
      docker_image           = docker_image,
  }

  output {
    Array[File] cluster_artifacts = validate_cluster.artifact_tarball
    File        cohort_bundle     = collate_cohort.cohort_bundle
    File        cohort_summary    = collate_cohort.cohort_summary_json
  }
}
