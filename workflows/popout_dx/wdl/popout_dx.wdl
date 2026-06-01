version 1.0

## popout diagnostics (DX) — discover, scatter, gather.
##
## Single config-driven entry point. ``discover_runs`` reads the YAML
## config + the FLARE-validate cohort bundle, enumerates the
## ``(cluster_id, chrom)`` shards, and emits a headerless TSV the WDL
## scatters on directly via ``read_tsv()``. One Cromwell task per shard
## runs ``run_dx_cluster.py``, emits a versioned per-cluster tarball
## matching ``validation/popout_dx/SCHEMA.md`` §1. A final ``collate_dx``
## task gathers all artifacts and emits the cohort bundle (§2).
##
## Reporting (PDF, dashboards) lives outside this WDL — it consumes the
## cohort bundle off-line. See ``validation/popout_dx/SCHEMA.md``.

task discover_runs {
  input {
    File   config_file
    String popout_outputs
    String mode

    Int    cpu          = 2
    String memory       = "8 GB"
    Int    disk_size_gb = 20
    String disk_type    = "HDD"
    Int    preemptible  = 1

    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  command <<<
    set -euo pipefail
    mkdir -p out
    python3 /opt/validation/popout_dx/scripts/discover_runs.py \
      --config         ~{config_file} \
      --popout-outputs ~{popout_outputs} \
      --mode           ~{mode} \
      --out-dir        out
    ls -lh out/
  >>>

  output {
    File        runs_manifest_json = "out/runs_manifest.json"
    File        runs_manifest_tsv  = "out/runs_manifest.tsv"   # headerless; for read_tsv()
    File        runs_manifest_with_header = "out/runs_manifest.tsv.with_header.tsv"
    # The per-cluster FLARE slices extracted from the cohort bundle.
    # Each scatter shard picks its slice by basename match.
    Array[File] flare_slices = glob("out/flare_slices/**/*")
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task popout_dx_per_cluster {
  input {
    # Per-shard scalars (from the manifest row).
    String cluster_id
    String chrom

    # Per-shard files (Cromwell auto-localizes via String → File coercion
    # at the call site).
    File   flare_global_tsv
    File   flare_labels_json
    String flare_anc_vcf            # may be empty in global mode

    # Whole-cohort singletons.
    File   popout_global_tsv
    File   popout_tracts
    File   popout_model
    File   popout_model_npz
    File?  popout_summary
    File?  rye_q_path
    File?  rf_ancestry_path

    # Run-level scalars.
    File   config_file
    String run_name
    String mode
    String tools                    # comma-separated, e.g. "popout,flare,rye,rf"

    # Local-mode parameters (ignored when mode=global).
    Int     local_per_bucket_n = 25
    Float   local_threshold    = 0.80
    Int     local_rng_seed     = 42
    String  local_coarse_grids_mb = "1 2 5 10 20"

    # Resource grant. The popout DX shard is light (subset arithmetic +
    # bcftools query for local mode); the small bucket fits most runs.
    Int    cpu          = 4
    String memory       = "16 GB"
    Int    disk_size_gb = 40
    String disk_type    = "HDD"
    Int    preemptible  = 1

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  String out_tarball = "~{cluster_id}.~{chrom}.popout_dx.v1.0.0.tar.gz"
  String run_id      = "~{run_name}.~{cluster_id}.~{chrom}"

  command <<<
    set -euo pipefail

    # ---- magicwand wiring (run-level tier-1 metrics → W&B) -----------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=popout-dx
    export WANDB_RUN_NAME="~{run_id}"
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    magicwand log \
      popout_dx.cluster_id="~{cluster_id}" \
      popout_dx.chrom="~{chrom}" \
      popout_dx.run_name="~{run_name}" \
      popout_dx.mode="~{mode}" \
      popout_dx.tools="~{tools}"

    # ---- build a fake manifest TSV with just this shard --------------
    # The orchestrator looks rows up by (cluster_id, chrom). For the WDL
    # shard's needs, a single-row TSV is enough.
    mkdir -p shard_inputs
    {
      printf "%s\t" "~{cluster_id}" "~{chrom}" \
        "~{flare_global_tsv}" "~{flare_labels_json}" "~{flare_anc_vcf}" \
        "~{popout_global_tsv}" "~{popout_tracts}" "~{popout_model}" \
        "~{popout_model_npz}" "~{default='' popout_summary}" \
        "~{default='' rye_q_path}" "~{default='' rf_ancestry_path}"
      printf "\n"
    } > shard_inputs/manifest.tsv

    # ---- run the orchestrator ----------------------------------------
    python3 /opt/validation/popout_dx/scripts/run_dx_cluster.py \
      --runs-manifest-tsv shard_inputs/manifest.tsv \
      --cluster-id        ~{cluster_id} \
      --chrom             ~{chrom} \
      --mode              ~{mode} \
      --run-name          "~{run_id}" \
      --tools             "~{tools}" \
      --config-file       ~{config_file} \
      --work-dir          work \
      --max-workers       ~{cpu} \
      --local-per-bucket-n ~{local_per_bucket_n} \
      --local-threshold    ~{local_threshold} \
      --local-rng-seed     ~{local_rng_seed} \
      --local-coarse-grids-mb ~{local_coarse_grids_mb} \
      --emit-tarball      ~{out_tarball}

    # ---- replay tier1 metrics into magicwand -------------------------
    mkdir -p .t1
    tar -xzf ~{out_tarball} -C .t1 --strip-components=2 \
      "~{cluster_id}/~{chrom}/tier1_metrics.tsv"
    while IFS=$'\t' read -r key value; do
      [ -n "$key" ] || continue
      magicwand log "$key=$value"
    done < .t1/tier1_metrics.tsv
    rm -rf .t1

    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')
    magicwand log popout_dx.task_peak_rss_gb="$PEAK_RSS_GB" || true
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

task collate_dx {
  input {
    Array[File] cluster_artifacts
    String      run_name
    String      mode
    String      tools

    Int     cpu          = 4
    String  memory       = "16 GB"
    Int     disk_size_gb = 50
    String  disk_type    = "HDD"
    Int     preemptible  = 1

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  String out_bundle = "cohort_dx.~{run_name}.v1.0.0.tar.gz"

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=popout-dx
    export WANDB_RUN_NAME="~{run_name}.collate"
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    magicwand log \
      popout_dx_collate.run_name="~{run_name}" \
      popout_dx_collate.n_artifacts="~{length(cluster_artifacts)}" \
      popout_dx_collate.mode="~{mode}" \
      popout_dx_collate.tools="~{tools}"

    python3 /opt/validation/popout_dx/scripts/collate_dx.py \
      --tarballs    ~{sep=" " cluster_artifacts} \
      --run-name    ~{run_name} \
      --mode        ~{mode} \
      --tools       "~{tools}" \
      --out-dir     collate \
      --out-tarball ~{out_bundle}

    # Bubble up cohort-summary headlines for the dashboard.
    python3 -c "
import json
s = json.load(open('collate/cohort_dx/cohort_summary.json'))
print(f\"popout_dx_collate.n_clusters={s['n_clusters']}\")
print(f\"popout_dx_collate.n_artifacts={s['n_artifacts']}\")
for p in s.get('pairs', []):
    tool, lbl = p['tool'], p['rf_label']
    ccc = p.get('mean_ccc_across_clusters')
    if ccc is not None:
        print(f\"popout_dx_collate.mean_ccc_{tool}_{lbl}={ccc:.4f}\")
" | while read line; do magicwand log "$line" || true; done

    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')
    magicwand log popout_dx_collate.peak_rss_gb="$PEAK_RSS_GB" || true
  >>>

  output {
    File cohort_bundle = "~{out_bundle}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow popout_dx {
  input {
    File   config_file                 # FLARE + rye/rf + globs + local-sampling knobs
    String popout_outputs              # GCS path containing one popout run's outputs
    String run_name
    String mode = "global"             # "global" or "global_local"
    String tools = "popout,flare,rye,rf"

    # Per-shard resource overrides (optional; defaults are small-bucket).
    Int?    shard_cpu
    String? shard_memory
    Int?    shard_disk_gb

    # Local-mode parameters.
    Int     local_per_bucket_n = 25
    Float   local_threshold    = 0.80
    Int     local_rng_seed     = 42
    String  local_coarse_grids_mb = "1 2 5 10 20"

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  call discover_runs {
    input:
      config_file    = config_file,
      popout_outputs = popout_outputs,
      mode           = mode,
      docker_image   = docker_image,
  }

  Array[Array[String]] rows = read_tsv(discover_runs.runs_manifest_tsv)

  scatter (row in rows) {
    # Column order matches discover_runs.TSV_COLUMNS:
    #   0=cluster_id 1=chrom
    #   2=flare_global_tsv 3=flare_labels_json 4=flare_anc_vcf
    #   5=popout_global_tsv 6=popout_tracts 7=popout_model
    #   8=popout_model_npz 9=popout_summary
    #   10=rye_q_path 11=rf_ancestry_path
    call popout_dx_per_cluster {
      input:
        cluster_id          = row[0],
        chrom               = row[1],
        flare_global_tsv    = row[2],
        flare_labels_json   = row[3],
        flare_anc_vcf       = row[4],
        popout_global_tsv   = row[5],
        popout_tracts       = row[6],
        popout_model        = row[7],
        popout_model_npz    = row[8],
        popout_summary      = if row[9]  == "" then None else row[9],
        rye_q_path          = if row[10] == "" then None else row[10],
        rf_ancestry_path    = if row[11] == "" then None else row[11],

        config_file         = config_file,
        run_name            = run_name,
        mode                = mode,
        tools               = tools,

        local_per_bucket_n  = local_per_bucket_n,
        local_threshold     = local_threshold,
        local_rng_seed      = local_rng_seed,
        local_coarse_grids_mb = local_coarse_grids_mb,

        cpu          = select_first([shard_cpu, 4]),
        memory       = select_first([shard_memory, "16 GB"]),
        disk_size_gb = select_first([shard_disk_gb, 40]),
        wandb_api_key = wandb_api_key,
        docker_image  = docker_image,
    }
  }

  call collate_dx {
    input:
      cluster_artifacts = popout_dx_per_cluster.artifact_tarball,
      run_name          = run_name,
      mode              = mode,
      tools             = tools,
      wandb_api_key     = wandb_api_key,
      docker_image      = docker_image,
  }

  output {
    File runs_manifest_json = discover_runs.runs_manifest_json
    File cohort_bundle      = collate_dx.cohort_bundle
    Array[File] per_cluster_artifacts = popout_dx_per_cluster.artifact_tarball
  }
}
