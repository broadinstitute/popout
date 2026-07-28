version 1.0

## FLARE cleanup — remove a supplied list of research IDs from per-cluster
## FLARE outputs before the final chromosome-level merge. Bookkeeping-first:
## only the (cluster, chrom) pairs whose cluster actually contains one or
## more drop-list samples are re-materialized.
##
## Inputs mirror the manifest that make_flare_validate_config.py already
## produces for the stats-gather system (validation/scripts/discover_runs.py
## consumes the same schema). See workflows/flare/scripts/plan_cleanup.py
## for the exact contract.
##
## The workflow emits the cleaned artifacts as Array[File] workflow outputs;
## Cromwell/Terra delocalize them to a submission-specific prefix that this
## WDL cannot predict. Assembly of a cleaned manifest for flare_finalize
## happens post-hoc via workflows/flare/scripts/emit_cleaned_manifest.py:
## point it at a text file listing the cleaned URIs (from a Terra data
## table export or a `gcloud storage ls`) plus the original manifest, and
## it joins by basename to produce the finalize input.
##
## WDL-surface justification (CLAUDE.md): separate workflow rather than a
## mode of flare_pipeline.wdl because the pipeline cannot be invoked
## Stage-D-only without a mutually-exclusive mode flag, and cleanup is
## orthogonal to FLARE inference. User explicitly OK'd the addition.
##
## Observability: magicwand -> W&B project `flare_cleanup`.

task preflight {
  input {
    File          manifest_tsv
    File          drop_samples
    Array[File]   cluster_sample_lists

    Int    cpu          = 2
    String memory       = "8 GB"
    Int    disk_size_gb = 50
    String disk_type    = "HDD"
    Int    preemptible  = 1

    String? wandb_api_key
    String  docker_image
  }

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare_cleanup
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    mkdir -p out

    # Build the --cluster-sample-list flag list. Cromwell localizes each
    # sample-list into an arbitrary directory; plan_cleanup.py derives
    # cluster_id from the basename before the first '.' — the same rule
    # plink2_export_clusters.wdl already enforces upstream.
    SL_ARGS=()
    while IFS= read -r sl; do
      SL_ARGS+=(--cluster-sample-list "$sl")
    done < ~{write_lines(cluster_sample_lists)}

    python3 /opt/flare/scripts/plan_cleanup.py \
      --manifest ~{manifest_tsv} \
      --drop-ids ~{drop_samples} \
      "${SL_ARGS[@]}" \
      --out-dir out

    magicwand log \
      flare_cleanup.num_clusters_total="$(jq -r .num_clusters_total  out/stats.json)" \
      flare_cleanup.num_clusters_affected="$(jq -r .num_clusters_affected out/stats.json)" \
      flare_cleanup.num_drop_ids="$(jq -r .num_drop_ids out/stats.json)" \
      flare_cleanup.num_drops_matched="$(jq -r .num_drops_matched out/stats.json)" \
      flare_cleanup.num_manifest_rows="$(jq -r .num_manifest_rows out/stats.json)" \
      flare_cleanup.num_affected_rows="$(jq -r .num_affected_rows out/stats.json)"

    # Emit a headerless shard TSV for the scatter site: 4 columns —
    # cluster_id, chrom, anc_vcf URI, global_anc URI. Cromwell auto-coerces
    # the URI strings to File when the scatter body binds them to File inputs
    # (same pattern flare_validate.wdl uses at line 74-89).
    awk -F'\t' 'NR>1 {print $1"\t"$2"\t"$3"\t"$4}' \
        out/manifest_affected_rows.tsv \
        > out/shards.tsv

    ls -lh out/
  >>>

  output {
    File        audit_tsv                 = "out/cleanup_audit.tsv"
    File        affected_clusters_tsv     = "out/affected_clusters.tsv"
    File        affected_manifest_tsv     = "out/manifest_affected_rows.tsv"
    File        shards_tsv                = "out/shards.tsv"
    File        stats_json                = "out/stats.json"
    Array[File] per_cluster_drop_files    = glob("out/drops_*.txt")
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

task cleanup_pair {
  input {
    String      cluster_id
    String      chrom
    File        anc_vcf
    File        global_anc
    # All per-cluster drop files from the preflight. Task selects the one
    # named `drops_<cluster_id>.txt` at runtime. Small (few KB each) — the
    # O(K) localization per shard is negligible vs the anc VCF payload.
    Array[File] per_cluster_drop_files

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type    = "HDD"
    Int      preemptible  = 1

    String? wandb_api_key
    String  docker_image
  }

  Float in_gb     = size(anc_vcf, "GB") + size(global_anc, "GB")
  Int   auto_disk = ceil(in_gb * 3.0) + 20

  Int    cpu          = select_first([cpu_override, 4])
  String memory       = select_first([memory_override, "8 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk])

  String anc_out    = "~{cluster_id}.~{chrom}.anc.vcf.gz"
  String global_out = "~{cluster_id}.~{chrom}.global.anc.gz"

  command <<<
    set -euo pipefail

    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare_cleanup
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init

    # Select this cluster's drop file by basename. Hard-fail if not present —
    # would indicate an inconsistency between the preflight's affected set
    # and the shard iteration.
    DROP_FILE=""
    while IFS= read -r f; do
      if [ "$(basename "$f")" = "drops_~{cluster_id}.txt" ]; then
        DROP_FILE="$f"
        break
      fi
    done < ~{write_lines(per_cluster_drop_files)}
    if [ -z "$DROP_FILE" ]; then
      echo "ERROR: no drop file found for cluster ~{cluster_id} in localized set" >&2
      exit 1
    fi

    N_DROPS=$(awk 'NF && $1 !~ /^#/' "$DROP_FILE" | wc -l | tr -d ' ')
    N_ANC_IN=$(bcftools query -l ~{anc_vcf} | wc -l | tr -d ' ')
    N_GLOBAL_IN=$(( $(gunzip -c ~{global_anc} | wc -l | tr -d ' ') - 1 ))

    magicwand log \
      flare_cleanup.cluster_id="~{cluster_id}" \
      flare_cleanup.chrom="~{chrom}" \
      flare_cleanup.n_drops="$N_DROPS" \
      flare_cleanup.n_anc_samples_in="$N_ANC_IN" \
      flare_cleanup.n_global_samples_in="$N_GLOBAL_IN" \
      flare_cleanup.anc_vcf_bytes_in="$(stat -c %s ~{anc_vcf})" \
      flare_cleanup.global_anc_bytes_in="$(stat -c %s ~{global_anc})"

    # ---- VCF: bcftools view -S ^drops -----------------------------------
    # The '^' prefix on --samples-file inverts to exclude semantics. The
    # preflight has already scoped drops to samples this cluster contains,
    # so --force-samples is not needed.
    bcftools view \
      --samples-file "^$DROP_FILE" \
      --output-type z \
      --output "~{anc_out}" \
      --threads ~{cpu} \
      ~{anc_vcf}
    bcftools index --tbi --threads ~{cpu} "~{anc_out}"

    # ---- global.anc.gz: header-preserving row filter --------------------
    python3 /opt/flare/scripts/filter_global_anc.py \
      --in  ~{global_anc} \
      --out "~{global_out}" \
      --drop-ids "$DROP_FILE" \
      --strict-present

    N_ANC_OUT=$(bcftools query -l "~{anc_out}" | wc -l | tr -d ' ')
    N_GLOBAL_OUT=$(( $(gunzip -c "~{global_out}" | wc -l | tr -d ' ') - 1 ))

    # Both artifacts must lose exactly N_DROPS samples.
    if [ "$((N_ANC_IN - N_ANC_OUT))" -ne "$N_DROPS" ]; then
      echo "ERROR: anc VCF sample delta $((N_ANC_IN - N_ANC_OUT)) != N_DROPS $N_DROPS" >&2
      exit 1
    fi
    if [ "$((N_GLOBAL_IN - N_GLOBAL_OUT))" -ne "$N_DROPS" ]; then
      echo "ERROR: global.anc sample delta $((N_GLOBAL_IN - N_GLOBAL_OUT)) != N_DROPS $N_DROPS" >&2
      exit 1
    fi

    magicwand log \
      flare_cleanup.n_anc_samples_out="$N_ANC_OUT" \
      flare_cleanup.n_global_samples_out="$N_GLOBAL_OUT" \
      flare_cleanup.anc_vcf_bytes_out="$(stat -c %s "~{anc_out}")" \
      flare_cleanup.global_anc_bytes_out="$(stat -c %s "~{global_out}")"
  >>>

  output {
    File cleaned_anc_vcf       = "~{anc_out}"
    File cleaned_anc_vcf_index = "~{anc_out}.tbi"
    File cleaned_global_anc    = "~{global_out}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow flare_cleanup {
  input {
    # Manifest TSV (same schema as validation/make_flare_validate_config.py):
    # required cols cluster_id, chrom, anc_vcf, global_anc, flare_model,
    # flare_log, input_vcf; optional flare_qc_tsv. Non-schema columns pass
    # through untouched into the cleaned manifest.
    File          manifest_tsv

    # Per-cluster sample lists. Basename before the first '.' MUST equal
    # the cluster_id used in manifest_tsv (same convention plink2 uses in
    # plink2_export_clusters.wdl and flare_pipeline.wdl).
    Array[File]   cluster_sample_lists

    # Research IDs to drop. One per line; '#' comments allowed.
    File          drop_samples

    String?       wandb_api_key
    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  # --- Preflight: figure out which clusters need work -----------------
  call preflight {
    input:
      manifest_tsv         = manifest_tsv,
      drop_samples         = drop_samples,
      cluster_sample_lists = cluster_sample_lists,
      wandb_api_key        = wandb_api_key,
      docker_image         = docker_image
  }

  # --- Scatter over affected (cluster, chrom) rows -------------------
  # shards.tsv is headerless with columns cluster_id, chrom, anc_vcf,
  # global_anc. Cromwell coerces the URI strings to File via the task
  # input binding (same pattern flare_validate.wdl uses).
  Array[Array[String]] shards = read_tsv(preflight.shards_tsv)

  scatter (row in shards) {
    call cleanup_pair {
      input:
        cluster_id             = row[0],
        chrom                  = row[1],
        anc_vcf                = row[2],
        global_anc             = row[3],
        per_cluster_drop_files = preflight.per_cluster_drop_files,
        wandb_api_key          = wandb_api_key,
        docker_image           = docker_image
    }
  }

  output {
    File        cleanup_audit_tsv       = preflight.audit_tsv
    File        affected_clusters_tsv   = preflight.affected_clusters_tsv
    File        stats_json              = preflight.stats_json

    Array[File] cleaned_anc_vcfs        = cleanup_pair.cleaned_anc_vcf
    Array[File] cleaned_anc_vcf_indices = cleanup_pair.cleaned_anc_vcf_index
    Array[File] cleaned_global_anc      = cleanup_pair.cleaned_global_anc
  }
}
