version 1.0

## FLARE finalize — cross-cluster per-chromosome merge of FLARE outputs.
##
## For each chromosome:
##   anc VCFs   : bcftools merge (column-wise, union of samples at shared sites)
##   global TSVs: row-concat, byte-identical header enforced across clusters
##
## Two input modes:
##   Manifest mode : File manifest_tsv (schema pinned by
##                   validation/make_flare_validate_config.py). Preflight
##                   parses it into per-chrom URI lists.
##   Direct  mode  : Array[String] chromosomes + two Array[Array[File]] inputs
##                   already bucketed by chrom (same shape flare_pipeline.wdl
##                   constructs at line 184). Skips the preflight.
##
## Exactly one mode must be supplied. The manifest mode is the primary path
## and lines up with the stats-gather system's manifest contract; direct
## mode exists so a caller that already has the per-chrom arrays (e.g.
## flare_cleanup.wdl output plumbed straight through) does not have to
## round-trip through a manifest write/read.
##
## WDL-surface justification (CLAUDE.md): separate workflow from
## flare_pipeline.wdl because that pipeline cannot run merge-only, and
## flare_pipeline.wdl's Stage D handles only VCFs — global TSV concat has no
## counterpart there. User explicitly OK'd the addition.

import "../../bcftools/wdl/bcftools_merge.wdl" as merge_wf

task preflight_finalize {
  input {
    File   manifest_tsv

    Int    cpu          = 2
    String memory       = "8 GB"
    Int    disk_size_gb = 20
    String disk_type    = "HDD"
    Int    preemptible  = 1

    String? wandb_api_key
    String  docker_image
  }

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare_finalize
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    mkdir -p out
    python3 /opt/flare/scripts/plan_finalize.py \
      --manifest ~{manifest_tsv} \
      --out-dir  out

    magicwand log \
      flare_finalize.num_chroms="$(jq -r .num_chroms out/stats.json)" \
      flare_finalize.num_rows="$(jq -r .num_rows out/stats.json)"

    ls -lh out/ out/anc_lists/ out/global_lists/
  >>>

  # plan_finalize.py names per-chrom list files with a zero-padded numeric
  # prefix (e.g. 0000__chr1.txt, 0001__chr2.txt, ...) so a lexicographic
  # glob returns them in the same order as chroms.txt. WDL 1.0 lacks the
  # array combinators (suffix / map) needed to derive these paths from the
  # chrom names directly.
  output {
    File          chroms_file  = "out/chroms.txt"
    Array[String] chroms       = read_lines("out/chroms.txt")
    Array[File]   anc_lists    = glob("out/anc_lists/*.txt")
    Array[File]   global_lists = glob("out/global_lists/*.txt")
    File          stats_json   = "out/stats.json"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task concat_global {
  input {
    Array[File] cluster_global_ancs
    String      output_prefix

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 1

    String? wandb_api_key
    String  docker_image
  }

  Float in_gb     = size(cluster_global_ancs, "GB")
  Int   auto_disk = ceil(in_gb * 3.0) + 20

  Int    cpu          = select_first([cpu_override, 2])
  String memory       = select_first([memory_override, "4 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk])

  String out_name = "~{output_prefix}.global.anc.gz"

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare_finalize
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    # Build the --input flag list. Inputs are consumed in the array order,
    # which the preflight already sorted by cluster_id for determinism.
    IN_ARGS=()
    while IFS= read -r f; do
      IN_ARGS+=(--input "$f")
    done < ~{write_lines(cluster_global_ancs)}

    N_IN=${#IN_ARGS[@]}
    N_IN=$((N_IN / 2))
    magicwand log \
      flare_finalize.output_prefix="~{output_prefix}" \
      flare_finalize.num_input_globals="$N_IN"

    python3 /opt/flare/scripts/concat_global_anc.py \
      --out "~{out_name}" \
      "${IN_ARGS[@]}"

    N_OUT=$(( $(gunzip -c "~{out_name}" | wc -l | tr -d ' ') - 1 ))
    magicwand log \
      flare_finalize.n_global_rows_out="$N_OUT" \
      flare_finalize.global_bytes_out="$(stat -c %s "~{out_name}")"
  >>>

  output {
    File merged_global_anc = "~{out_name}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow flare_finalize {
  input {
    # --- Mode A: manifest-driven ---
    File? manifest_tsv

    # --- Mode B: direct per-chrom arrays ---
    # Position-aligned. anc_vcfs_by_chrom[i] and global_ancs_by_chrom[i]
    # are the K cluster artifacts covering chromosomes[i].
    Array[String]?      chromosomes
    Array[Array[File]]? anc_vcfs_by_chrom
    Array[Array[File]]? global_ancs_by_chrom

    String?       wandb_api_key
    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  # --- Mode A: run preflight if a manifest was supplied ------------------
  if (defined(manifest_tsv)) {
    call preflight_finalize {
      input:
        manifest_tsv  = select_first([manifest_tsv]),
        wandb_api_key = wandb_api_key,
        docker_image  = docker_image
    }
  }

  # --- Resolve the working chrom list (both modes converge here) ---------
  Array[String] chroms_resolved = select_first([preflight_finalize.chroms, chromosomes])

  scatter (i in range(length(chroms_resolved))) {
    String chrom_i = chroms_resolved[i]

    # Per-chrom cluster input arrays. In manifest mode, read_lines the
    # preflight-emitted list file for this chrom and let Cromwell coerce
    # Array[String] URIs to Array[File]. In direct mode, index the caller-
    # supplied Array[Array[File]] directly.
    if (defined(manifest_tsv)) {
      Array[File] anc_from_manifest    = read_lines(select_first([preflight_finalize.anc_lists])[i])
      Array[File] global_from_manifest = read_lines(select_first([preflight_finalize.global_lists])[i])
    }

    Array[File] anc_this_chrom    = select_first([
      anc_from_manifest,
      select_first([anc_vcfs_by_chrom])[i]
    ])
    Array[File] global_this_chrom = select_first([
      global_from_manifest,
      select_first([global_ancs_by_chrom])[i]
    ])

    call merge_wf.bcftools_merge as merge_anc {
      input:
        vcfs          = anc_this_chrom,
        output_prefix = chrom_i + ".anc",
        output_type   = "z",
        write_index   = true,
        wandb_api_key = wandb_api_key
    }

    call concat_global {
      input:
        cluster_global_ancs = global_this_chrom,
        output_prefix       = chrom_i,
        wandb_api_key       = wandb_api_key,
        docker_image        = docker_image
    }
  }

  output {
    Array[String] chromosomes_out        = chroms_resolved
    Array[File]   chrom_anc_vcfs         = merge_anc.merged_vcf
    Array[File?]  chrom_anc_vcf_indices  = merge_anc.merged_index
    Array[File]   chrom_global_anc       = concat_global.merged_global_anc

    File?         preflight_stats_json   = preflight_finalize.stats_json
    File?         preflight_chroms_file  = preflight_finalize.chroms_file
  }
}
