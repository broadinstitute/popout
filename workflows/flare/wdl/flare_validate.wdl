version 1.0

## FLARE per-cluster tract extraction — scatter (extract) + gather (collate).
##
## v7 rewrite: replaces the fused 11-script per-cluster validation DAG with
## a single lean pass that emits raw parquet tract events per shard, plus a
## single-node collate step that concatenates shards into a hive-partitioned
## cohort bundle. See my_notes/waterfall/PLAN.md for the rationale — every
## downstream statistic (tract length dist, hap agreement, crossover hotspot
## map, transition matrix, admixture, genomic-structure overlap, diversity)
## is now a local DuckDB query against the bundle, not a per-shard summary
## the WDL pre-cooks.
##
## Contract (unchanged from v4):
##   - Input:  `File config_file` (gs://JSON, per discover_runs.py schema).
##   - Scatter: one task per (cluster_id, chrom) row.
##   - Output: per-shard tarball + cohort bundle tarball.
##
## Only column [0]=cluster_id, [1]=chrom, [2]=anc_vcf are referenced from
## the discover_runs manifest; the remaining columns (global_anc,
## flare_model, flare_log, rf_ancestry, chrom_sizes, rye_q, etc.) are left
## in the TSV for backwards-compatibility with the config schema but are
## not localized by Cromwell — the extractor doesn't need them.
##
## Resource posture: cpu 2, mem 4 GB, HDD, preemptible 3. The extractor is
## a single-core bcftools stream + numpy state machine; there is nothing to
## parallelize within a shard.

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
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task extract_shard {
  input {
    String cluster_id
    String chrom
    File   anc_vcf

    String run_name

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 3

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  # bcftools streams the anc.vcf.gz once; the state machine holds
  # O(n_samples) ints in memory. Disk = anc_vcf itself + tarball out.
  Float anc_gb        = size(anc_vcf, "GB")
  Int   auto_disk_gb  = ceil(anc_gb) + 5
  Int   cpu           = select_first([cpu_override, 2])
  String memory       = select_first([memory_override, "4 GB"])
  Int   disk_size_gb  = select_first([disk_size_gb_override, auto_disk_gb])

  String run_id      = "~{run_name}.~{cluster_id}.~{chrom}"
  String out_dirname = "~{cluster_id}.~{chrom}"
  String out_tarball = "~{cluster_id}.~{chrom}.tracts.tar.gz"

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare-validate
    export WANDB_RUN_NAME="~{run_id}"
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh) || true
    magicwand init || true

    magicwand log \
      flare_extract.cluster_id="~{cluster_id}" \
      flare_extract.chrom="~{chrom}" \
      flare_extract.anc_vcf_bytes="$(stat -c %s ~{anc_vcf})" \
      flare_extract.cpu="~{cpu}" \
      flare_extract.memory="~{memory}" \
      flare_extract.disk_gb="~{disk_size_gb}"

    mkdir -p "~{out_dirname}"
    python3 /opt/validation/scripts/extract_tract_events.py \
      ~{anc_vcf} \
      --out-dir "~{out_dirname}" \
      --cluster-id "~{cluster_id}" \
      -v

    # Log key extractor metrics from provenance.json.
    python3 - <<'PY' | while read line; do magicwand log "$line" || true; done
import json, pathlib
p = json.loads(pathlib.Path("~{out_dirname}/provenance.json").read_text())
print(f"flare_extract.n_samples={p['n_samples']}")
print(f"flare_extract.n_sites={sum(p['n_sites_per_chrom'].values())}")
print(f"flare_extract.n_tracts={p['n_tracts_emitted']}")
print(f"flare_extract.n_transitions={p['n_transitions_emitted']}")
print(f"flare_extract.wall_s={p['wall_s']:.2f}")
for name, ok in p["checks_passed"].items():
    print(f"flare_extract.check.{name}={int(ok)}")
PY

    tar -C "~{out_dirname}" -czf "~{out_tarball}" .

    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')
    magicwand log flare_extract.task_peak_rss_gb="$PEAK_RSS_GB" || true
  >>>

  output {
    File shard_tarball = "~{out_tarball}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task collate_bundle {
  input {
    Array[File] shard_tarballs
    String      run_name

    Int     cpu          = 4
    String  memory       = "16 GB"
    Int     disk_size_gb = 100
    String  disk_type    = "HDD"
    Int     preemptible  = 1

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  String out_dirname = "cohort_bundle.~{run_name}"
  String out_tarball = "cohort_bundle.~{run_name}.tar.gz"

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare-validate
    export WANDB_RUN_NAME="~{run_name}.collate"
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh) || true
    magicwand init || true

    magicwand log \
      flare_collate.run_name="~{run_name}" \
      flare_collate.n_shards="~{length(shard_tarballs)}"

    # Build the --shard-tarball arg list.
    ARGS=()
    for t in ~{sep=" " shard_tarballs}; do
      ARGS+=("--shard-tarball" "$t")
    done

    mkdir -p "~{out_dirname}"
    python3 /opt/validation/scripts/collate_tract_bundle.py \
      "${ARGS[@]}" \
      --out-dir "~{out_dirname}" \
      --run-name "~{run_name}"

    python3 - <<'PY' | while read line; do magicwand log "$line" || true; done
import json, pathlib
m = json.loads(pathlib.Path("~{out_dirname}/cohort_manifest.json").read_text())
print(f"flare_collate.n_samples={m['n_samples']}")
print(f"flare_collate.n_chroms={len(m['chroms'])}")
for k, v in m["row_counts"].items():
    print(f"flare_collate.row_counts.{k}={v}")
PY

    tar -C "~{out_dirname}" -czf "~{out_tarball}" .

    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')
    magicwand log flare_collate.peak_rss_gb="$PEAK_RSS_GB" || true
  >>>

  output {
    File cohort_bundle = "~{out_tarball}"
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
    # v4 config-file contract kept intact (per discover_runs.py schema).
    File config_file

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 3

    String?  wandb_api_key
    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  call discover_runs {
    input:
      config_file  = config_file,
      docker_image = docker_image,
  }

  # discover_runs still emits the full 15-column TSV. The v7 extractor
  # only needs cluster_id, chrom, anc_vcf; Cromwell will not localize the
  # other columns because we never coerce them to File here.
  Array[Array[String]] rows = read_tsv(discover_runs.runs_manifest_tsv)

  scatter (row in rows) {
    call extract_shard {
      input:
        cluster_id            = row[0],
        chrom                 = row[1],
        anc_vcf               = row[2],
        run_name              = discover_runs.run_name,

        cpu_override          = cpu_override,
        memory_override       = memory_override,
        disk_size_gb_override = disk_size_gb_override,
        disk_type             = disk_type,
        preemptible           = preemptible,
        wandb_api_key         = wandb_api_key,
        docker_image          = docker_image,
    }
  }

  call collate_bundle {
    input:
      shard_tarballs = extract_shard.shard_tarball,
      run_name       = discover_runs.run_name,
      wandb_api_key  = wandb_api_key,
      docker_image   = docker_image,
  }

  output {
    Array[File] shard_tarballs   = extract_shard.shard_tarball
    File        cohort_bundle    = collate_bundle.cohort_bundle
    File        runs_manifest_tsv  = discover_runs.runs_manifest_with_header
    File        runs_manifest_json = discover_runs.runs_manifest_json
  }
}
