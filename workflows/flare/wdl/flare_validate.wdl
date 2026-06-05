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
##
## ★ v4.0.0: input contract is now a single `File config_file` (Cromwell
## localizes from gs://). A new `discover_runs` task opens it and emits a
## headerless TSV of per-shard inputs; the workflow scatters via
## `read_tsv()` and lets Cromwell auto-localize per-shard files via the
## standard String -> File coercion. See validation/scripts/discover_runs.py
## for the config schema.

task discover_runs {
  input {
    File   config_file
    Int    cpu          = 2
    String memory       = "8 GB"
    Int    disk_size_gb = 20
    String disk_type    = "HDD"
    Int    preemptible  = 1
    String docker_image
  }

  command <<<
    set -euo pipefail
    mkdir -p out
    python3 /opt/validation/scripts/discover_runs.py \
      --config  ~{config_file} \
      --out-dir out
    ls -lh out/
  >>>

  output {
    File   runs_manifest_json        = "out/runs_manifest.json"
    File   runs_manifest_tsv         = "out/runs_manifest.tsv"
    File   runs_manifest_with_header = "out/runs_manifest.tsv.with_header.tsv"
    String run_name                   = read_string("out/run_name.txt")
    String schema_version             = read_string("out/schema_version.txt")
    String mid_rule                   = read_string("out/mid_rule.txt")
    String panel_id                   = read_string("out/panel_id.txt")
    String collation_config_uri       = read_string("out/collation_config_uri.txt")
    String previous_cohort_bundle_uri = read_string("out/previous_cohort_bundle_uri.txt")
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task validate_cluster {
  input {
    # Per-shard inputs (flattened from the v3 FlareClusterRun struct).
    String cluster_id
    String chrom
    File   anc_vcf
    File   global_anc
    File   flare_model
    File   flare_log
    File?  flare_qc_tsv              # ★ v1.1: optional — pre-pipeline fixtures have no qc.tsv
    File   input_vcf                 # per-cluster gt VCF; header-only reads in coverage + provenance

    # Cohort singletons (repeated per row in the TSV; Cromwell auto-localizes
    # at the scatter call site via String -> File coercion).
    File   rf_ancestry
    File   chrom_sizes
    File?  rye_q                   # ★ v1.1: Rye Q TSV (was admixture_q); cohort-wide singleton
    File?  self_id
    File?  popout_secondary_global
    File?  popout_secondary_labels
    File?  ref_panel               # for sha256-in-manifest only

    String run_name                # magicwand run-name prefix
    String? panel_id
    String schema_version = "4.0.0"   # ★ v4.0.0 (input contract refactor)

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
  Float anc_gb = size(anc_vcf, "GB")
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

  String run_id = "~{run_name}.~{cluster_id}.~{chrom}"
  String out_tarball = "~{cluster_id}.~{chrom}.validation.v~{schema_version}.tar.gz"

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
      flare_validate.cluster_id="~{cluster_id}" \
      flare_validate.chrom="~{chrom}" \
      flare_validate.schema_version="~{schema_version}" \
      flare_validate.anc_vcf_bytes="$(stat -c %s ~{anc_vcf})" \
      flare_validate.input_vcf_bytes="$(stat -c %s ~{input_vcf})" \
      flare_validate.cpu="~{cpu}" \
      flare_validate.memory_gb="~{auto_mem_int}" \
      flare_validate.disk_gb="~{disk_size_gb}" \
      flare_validate.disk_type="~{disk_type}" \
      flare_validate.optional_rye_q="~{defined(rye_q)}" \
      flare_validate.optional_flare_qc_tsv="~{defined(flare_qc_tsv)}" \
      flare_validate.optional_self_id="~{defined(self_id)}" \
      flare_validate.optional_popout_secondary="~{defined(popout_secondary_global)}"

    # ---- run the orchestrator ----------------------------------------
    # The orchestrator's DAG runner sizes its thread pool to --max-workers;
    # we pass through the WDL-allocated CPU count so within-task parallelism
    # tracks the resource grant.
    python3 /opt/validation/scripts/run_cluster_validation.py \
      --cluster-id     ~{cluster_id} \
      --chrom          ~{chrom} \
      --anc-vcf        ~{anc_vcf} \
      --global-anc     ~{global_anc} \
      --flare-model    ~{flare_model} \
      --flare-log      ~{flare_log} \
      ~{"--flare-qc-tsv "            + flare_qc_tsv} \
      --input-vcf      ~{input_vcf} \
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
      "~{cluster_id}/~{chrom}/tier1_metrics.tsv"
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
    String      schema_version = "4.0.0"     # ★ v4.0.0 (input contract refactor)

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
    # ★ v4.0.0 input contract: a single gs:// JSON config (Cromwell-localized).
    # All cohort singletons (rf_ancestry, chrom_sizes, rye_q, etc.) and the
    # full per-(cluster, chrom) cluster_runs list live inside this config.
    # See validation/scripts/discover_runs.py for the schema.
    File config_file

    # Per-shard resource overrides (orthogonal to the config). Defaults are
    # auto-sized by anc_vcf size in validate_cluster.
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 0

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  call discover_runs {
    input:
      config_file  = config_file,
      docker_image = docker_image,
  }

  # Headerless TSV; column indices match
  # validation/scripts/discover_runs.py:TSV_COLUMNS exactly. Keep in sync.
  Array[Array[String]] rows = read_tsv(discover_runs.runs_manifest_tsv)

  scatter (row in rows) {
    # WDL 1.0 has no None literal; the idiomatic way to lift a possibly-empty
    # String column into Optional[File] is a conditional declaration.
    if (row[6]  != "") { File flare_qc_tsv_opt          = row[6]  }
    if (row[10] != "") { File rye_q_opt                 = row[10] }
    if (row[11] != "") { File self_id_opt               = row[11] }
    if (row[12] != "") { File popout_secondary_global_o = row[12] }
    if (row[13] != "") { File popout_secondary_labels_o = row[13] }
    if (row[14] != "") { File ref_panel_opt             = row[14] }

    call validate_cluster {
      input:
        cluster_id              = row[0],
        chrom                   = row[1],
        anc_vcf                 = row[2],
        global_anc              = row[3],
        flare_model             = row[4],
        flare_log               = row[5],
        flare_qc_tsv            = flare_qc_tsv_opt,
        input_vcf               = row[7],
        rf_ancestry             = row[8],
        chrom_sizes             = row[9],
        rye_q                   = rye_q_opt,
        self_id                 = self_id_opt,
        popout_secondary_global = popout_secondary_global_o,
        popout_secondary_labels = popout_secondary_labels_o,
        ref_panel               = ref_panel_opt,

        run_name                = discover_runs.run_name,
        panel_id                = discover_runs.panel_id,
        schema_version          = discover_runs.schema_version,

        cpu_override            = cpu_override,
        memory_override         = memory_override,
        disk_size_gb_override   = disk_size_gb_override,
        disk_type               = disk_type,
        preemptible             = preemptible,
        wandb_api_key           = wandb_api_key,
        docker_image            = docker_image,
    }
  }

  # Lift the discover-emitted gs:// strings to Optional[File] so Cromwell
  # localizes them for the collate task only when actually present.
  if (discover_runs.collation_config_uri       != "") { File collation_config_opt       = discover_runs.collation_config_uri }
  if (discover_runs.previous_cohort_bundle_uri != "") { File previous_cohort_bundle_opt = discover_runs.previous_cohort_bundle_uri }

  call collate_cohort {
    input:
      cluster_artifacts      = validate_cluster.artifact_tarball,
      collation_config       = collation_config_opt,
      previous_cohort_bundle = previous_cohort_bundle_opt,
      run_name               = discover_runs.run_name,
      schema_version         = discover_runs.schema_version,
      mid_rule               = discover_runs.mid_rule,
      wandb_api_key          = wandb_api_key,
      docker_image           = docker_image,
  }

  output {
    Array[File] cluster_artifacts = validate_cluster.artifact_tarball
    File        cohort_bundle     = collate_cohort.cohort_bundle
    File        cohort_summary    = collate_cohort.cohort_summary_json
    File        runs_manifest_tsv = discover_runs.runs_manifest_with_header
    File        runs_manifest_json = discover_runs.runs_manifest_json
  }
}
