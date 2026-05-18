version 1.0

## Per-chromosome, per-cluster VCF export from a PGEN triplet.
##
## Replaces the streaming-split scatter/gather that the FLARE pipeline used
## to run for Stage A. The key insight is that plink2 reads the genotype
## matrix once from PGEN (column-major binary), then `--keep --export` per
## cluster is cheap. Running K clusters in one task amortizes the single
## expensive read across all outputs:
##
##   for cluster in clusters:
##     plink2 --pfile <chrom> --keep <cluster> --export vcf-4.2 bgz id-paste=iid
##     bcftools index --tbi <cluster>.<chrom>.vcf.gz
##
## Inputs:
##   pgen/pvar/psam   — per-chromosome triplet (three parallel File inputs)
##   sample_groups    — per-cluster sample-list files (one IID per line,
##                      blanks + `#`-prefixed lines ignored, no FID column)
##   cluster_ids      — parallel array of cluster output basenames
##
## Outputs (returned via glob() so Cromwell delocalizes them):
##   subset_vcfs[i]     = <NNN>_<cluster_ids[i]>.<chrom>.vcf.gz
##   subset_indices[i]  = <NNN>_<cluster_ids[i]>.<chrom>.vcf.gz.tbi
##
## The NNN prefix is the zero-padded input index, present only to anchor
## glob()'s lexicographic order to cluster_ids[] input order — downstream
## consumers use the array position, not the filename.
##
## Resources auto-scale on PGEN size (matches pgen_to_vcf.wdl). Magicwand
## instrumented so the first Terra run validates the size→(cpu,mem,disk)
## table.

task plink2_export_clusters_task {
  input {
    File          pgen
    File          pvar
    File          psam
    Array[File]   sample_groups
    Array[String] cluster_ids        # parallel to sample_groups

    # Extra args appended to every plink2 invocation (escape hatch).
    String        extra_args = ""

    # Resource overrides — leave unset for auto-scaling by PGEN size.
    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override

    String? wandb_api_key
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
  }

  Float pgen_gb = size(pgen, "GB")

  # Auto-scale to match pgen_to_vcf.wdl:47-55. K cluster exports off one
  # PGEN read are cheaper than the unconditional VCF write that task does,
  # but memory + disk are dominated by the same factors.
  Int auto_cpu = if pgen_gb > 60.0 then 64
                 else if pgen_gb > 30.0 then 32
                 else if pgen_gb > 10.0 then 16
                 else 8

  String auto_memory = if pgen_gb > 60.0 then "256 GB"
                       else if pgen_gb > 30.0 then "128 GB"
                       else if pgen_gb > 10.0 then "64 GB"
                       else "32 GB"

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  # Each cluster output is a fraction of the full chrom VCF size, so total
  # output bytes scale with sum(cluster_sizes)/N_total — but we don't know
  # the cluster sizes until task-time. Worst case (all samples in one
  # cluster) is the full PGEN→VCF expansion; budget for it.
  Int    auto_disk    = ceil(pgen_gb * 5) + 100
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk])

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    magicwand() { :; }
    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=plink2_export_clusters
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    # Co-locate pfile triplet — same pattern as pgen_to_vcf.wdl:66-69.
    PREFIX="input_pfile"
    ln -sf ~{pgen} "${PREFIX}.pgen"
    ln -sf ~{pvar} "${PREFIX}.pvar"
    ln -sf ~{psam} "${PREFIX}.psam"

    OUT_DIR=out
    mkdir -p "$OUT_DIR"

    CLUSTER_IDS=(~{sep=' ' cluster_ids})
    SAMPLE_GROUPS=(~{sep=' ' sample_groups})
    NUM_CLUSTERS=${#CLUSTER_IDS[@]}
    NUM_GROUPS=${#SAMPLE_GROUPS[@]}
    if [ "$NUM_CLUSTERS" != "$NUM_GROUPS" ]; then
      echo "ERROR: cluster_ids ($NUM_CLUSTERS) and sample_groups ($NUM_GROUPS) length mismatch" >&2
      exit 2
    fi

    # plink2 `--keep` needs either a 2-col (FID, IID) file or a 1-col file
    # with a `#IID` header line. Our cluster sample lists are bare IIDs.
    # Rewrite each list into a plink2-friendly form with the header line
    # prepended (and `#`/blank lines stripped) before --keep sees it.
    KEEP_DIR=keep_files
    mkdir -p "$KEEP_DIR"
    PER_CLUSTER_SIZES=()
    TOTAL_SAMPLES=0
    for i in "${!CLUSTER_IDS[@]}"; do
      cid="${CLUSTER_IDS[$i]}"
      src="${SAMPLE_GROUPS[$i]}"
      keep="$KEEP_DIR/${cid}.keep"
      {
        echo "#IID"
        awk 'NF && $1 !~ /^#/ { print $1 }' "$src"
      } > "$keep"
      n=$(awk 'NF && $1 !~ /^#/' "$src" | wc -l | tr -d ' ')
      PER_CLUSTER_SIZES+=("$n")
      TOTAL_SAMPLES=$((TOTAL_SAMPLES + n))
    done

    # Derive a chromosome label for log + filenames. Use the basename of
    # the PGEN with .pgen stripped — Cromwell localizes per file so the
    # chrom name comes through unchanged.
    CHROM_LABEL=$(basename "~{pgen}" .pgen)

    INPUT_PGEN_BYTES=$(stat -c %s ~{pgen})

    # min/median/max of per-cluster sample counts via sort.
    SAMPLES_MIN=$(printf '%s\n' "${PER_CLUSTER_SIZES[@]}" | sort -n | head -n1)
    SAMPLES_MAX=$(printf '%s\n' "${PER_CLUSTER_SIZES[@]}" | sort -n | tail -n1)
    SAMPLES_MEDIAN=$(printf '%s\n' "${PER_CLUSTER_SIZES[@]}" | sort -n | \
      awk 'BEGIN {} {a[NR]=$1} END {if (NR%2==1) print a[(NR+1)/2]; else printf "%.1f", (a[NR/2]+a[NR/2+1])/2}')

    magicwand log \
      plink2_export_clusters.chrom_label="$CHROM_LABEL" \
      plink2_export_clusters.input_pgen_bytes="$INPUT_PGEN_BYTES" \
      plink2_export_clusters.num_clusters="$NUM_CLUSTERS" \
      plink2_export_clusters.total_samples="$TOTAL_SAMPLES" \
      plink2_export_clusters.samples_per_cluster_min="$SAMPLES_MIN" \
      plink2_export_clusters.samples_per_cluster_median="$SAMPLES_MEDIAN" \
      plink2_export_clusters.samples_per_cluster_max="$SAMPLES_MAX" \
      plink2_export_clusters.cpu="~{cpu}" \
      plink2_export_clusters.memory_gb="$(echo '~{memory}' | awk '{print $1}')" \
      plink2_export_clusters.disk_gb="~{disk_size_gb}"

    printf 'cluster_id\tsamples\twall_s\n' > per_cluster_timings.tsv

    # Output basenames are prefixed with the zero-padded input index so glob()
    # at task-output time returns files in cluster_ids[] order (which the
    # workflow relies on via transpose(stage_a.subset_vcfs) → by_cluster_vcfs[c]
    # ↔ cluster_ids[c]). Without the prefix, glob's lexicographic order would
    # silently desync from input order whenever cluster_ids is non-alphabetical.
    for i in "${!CLUSTER_IDS[@]}"; do
      cid="${CLUSTER_IDS[$i]}"
      idx=$(printf '%03d' "$i")
      keep="$KEEP_DIR/${cid}.keep"
      n_samples="${PER_CLUSTER_SIZES[$i]}"
      out_prefix="$OUT_DIR/${idx}_${cid}.${CHROM_LABEL}"

      echo "===== plink2 export: cluster=$cid samples=$n_samples ====="
      cluster_start=$(date +%s.%N)
      plink2 \
        --pfile "$PREFIX" \
        --keep "$keep" \
        --export vcf-4.2 bgz id-paste=iid \
        --output-chr chrM \
        --out "$out_prefix" \
        --threads ~{cpu} \
        ~{extra_args}
      cluster_end=$(date +%s.%N)
      wall_s=$(awk -v a="$cluster_end" -v b="$cluster_start" 'BEGIN {printf "%.3f", a-b}')

      bcftools index --tbi --threads ~{cpu} "${out_prefix}.vcf.gz"

      printf '%s\t%s\t%s\n' "$cid" "$n_samples" "$wall_s" >> per_cluster_timings.tsv

      magicwand log \
        "plink2_export_clusters.cluster_wall_s.${cid}=$wall_s"
    done

    OUTPUT_COUNT=$(ls -1 "$OUT_DIR"/*.vcf.gz | wc -l | tr -d ' ')
    TOTAL_OUTPUT_BYTES=$(stat -c %s "$OUT_DIR"/*.vcf.gz | awk '{s+=$1} END {print s+0}')
    DISK_USED_BYTES=$(du -sb "$OUT_DIR" 2>/dev/null | awk '{print $1}')
    # /proc/self/status VmHWM is in KiB (man proc(5)).
    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')

    magicwand log \
      plink2_export_clusters.output_count="$OUTPUT_COUNT" \
      plink2_export_clusters.total_output_bytes="$TOTAL_OUTPUT_BYTES" \
      plink2_export_clusters.disk_used_bytes="$DISK_USED_BYTES" \
      plink2_export_clusters.peak_rss_gb="$PEAK_RSS_GB"

    echo "===== per-cluster timings ====="
    cat per_cluster_timings.tsv
    echo "==============================="
    ls -lh "$OUT_DIR"
  >>>

  output {
    # glob() is the WDL output pattern Cromwell handles via dynamic post-task
    # delocalization. `Array[File] = read_lines(manifest)` does NOT — the
    # delocalize script is a static array baked pre-task, so manifest contents
    # are invisible to it and the listed files silently never reach GCS.
    Array[File] subset_vcfs         = glob("out/*.vcf.gz")
    Array[File] subset_indices      = glob("out/*.vcf.gz.tbi")
    File        per_cluster_timings = "per_cluster_timings.tsv"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow plink2_export_clusters {
  input {
    File          pgen
    File          pvar
    File          psam
    Array[File]   sample_groups
    Array[String] cluster_ids

    String        extra_args = ""

    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override

    String? wandb_api_key
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
  }

  call plink2_export_clusters_task {
    input:
      pgen                  = pgen,
      pvar                  = pvar,
      psam                  = psam,
      sample_groups         = sample_groups,
      cluster_ids           = cluster_ids,
      extra_args            = extra_args,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      wandb_api_key         = wandb_api_key,
      docker_image          = docker_image
  }

  output {
    Array[File] subset_vcfs         = plink2_export_clusters_task.subset_vcfs
    Array[File] subset_indices      = plink2_export_clusters_task.subset_indices
    File        per_cluster_timings = plink2_export_clusters_task.per_cluster_timings
  }
}
