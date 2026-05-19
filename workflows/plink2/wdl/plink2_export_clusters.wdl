version 1.0

## Per-(chromosome, cluster) VCF export from a PGEN triplet.
##
## Single plink2 invocation: read PGEN, --keep one cluster's sample list,
## --export vcf-4.2 bgz, index the resulting VCF with bcftools index --tbi.
##
##   plink2 --pfile <chrom> --keep <cluster> --export vcf-4.2 bgz id-paste=iid
##   bcftools index --tbi <cluster>.<chrom>.vcf.gz
##
## Earlier this task looped K clusters inside one big VM to amortize the
## PGEN read. That design didn't survive contact with biobank-scale data:
## the bottleneck wasn't the PGEN read but the single-stream bgz writer,
## peak memory was ~250× below the allocation, and the serial loop turned
## chr1 into a ~50 wall-hour task at full AoU scale. Pulling the loop into
## a WDL scatter at the workflow level (one task per (chrom, cluster) pair)
## gives parallelism + per-task right-sizing for a >20× cost cut.
##
## Inputs:
##   pgen/pvar/psam       — per-chromosome triplet
##   cluster_sample_list  — one IID per line, blanks + `#`-prefixed ignored
##   cluster_id           — output basename component (matches the parallel
##                          cluster_ids[] array in the caller workflow)
##
## Outputs (delocalized via glob() so dynamic post-task tagging works):
##   subset_vcf   — <cluster_id>.<chrom>.vcf.gz
##   subset_index — <cluster_id>.<chrom>.vcf.gz.tbi
##
## Resources: flat 4 CPU / 16 GB memory; disk scales on pgen_gb. HDD by
## default (sequential I/O, ~14 MB/s observed write rate well below
## pd-standard's throughput ceiling). Magicwand-instrumented.

task plink2_export_clusters_task {
  input {
    File   pgen
    File   pvar
    File   psam
    File   cluster_sample_list
    String cluster_id

    # Extra args appended to the plink2 invocation (escape hatch).
    String extra_args = ""

    # Resource overrides — leave unset for the right-sized defaults below.
    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override
    String  disk_type   = "HDD"
    Int     preemptible = 1

    String? wandb_api_key
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
  }

  # Right-sized defaults. W&B from the previous (looped) shape showed:
  #   peak process memory ~235 MB on 64 GB VMs (250-1000× headroom)
  #   process CPU bursts 20-80% on 16/32/64 cores (idle most of the time)
  #   sustained disk write ~14 MB/s (bgz single-thread is the bottleneck)
  # 4 CPU / 16 GB is plenty; throwing more cores at plink2 doesn't help.
  Int    cpu          = select_first([cpu_override, 4])
  String memory       = select_first([memory_override, "16 GB"])

  Float pgen_gb = size(pgen, "GB")
  # Disk = PGEN + worst-case per-cluster VCF.gz output. The per-cluster
  # output/pgen ratio on chr20 was ~40×; 50× is a generous upper bound.
  # Override per-task after the first green run trims this against actual
  # output_bytes from W&B.
  Int    auto_disk    = ceil(pgen_gb * 50) + 30
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

    # Co-locate pfile triplet so plink2 can find them via --pfile <prefix>.
    PREFIX="input_pfile"
    ln -sf ~{pgen} "${PREFIX}.pgen"
    ln -sf ~{pvar} "${PREFIX}.pvar"
    ln -sf ~{psam} "${PREFIX}.psam"

    OUT_DIR=out
    mkdir -p "$OUT_DIR"

    # plink2 `--keep` needs either a 2-col (FID, IID) file or a 1-col file
    # with a `#IID` header line. The cluster sample list is bare IIDs.
    KEEP=keep.tsv
    {
      echo "#IID"
      awk 'NF && $1 !~ /^#/ { print $1 }' ~{cluster_sample_list}
    } > "$KEEP"
    N_SAMPLES=$(awk 'NF && $1 !~ /^#/' ~{cluster_sample_list} | wc -l | tr -d ' ')

    # CHROM_LABEL = basename of the PGEN with .pgen stripped (e.g. chr1).
    CHROM_LABEL=$(basename "~{pgen}" .pgen)
    OUT_PREFIX="$OUT_DIR/~{cluster_id}.${CHROM_LABEL}"

    PGEN_BYTES=$(stat -c %s ~{pgen})

    magicwand log \
      plink2_export_clusters.chrom_label="$CHROM_LABEL" \
      plink2_export_clusters.cluster_id="~{cluster_id}" \
      plink2_export_clusters.cluster_samples="$N_SAMPLES" \
      plink2_export_clusters.pgen_bytes="$PGEN_BYTES" \
      plink2_export_clusters.cpu="~{cpu}" \
      plink2_export_clusters.memory_gb="$(echo '~{memory}' | awk '{print $1}')" \
      plink2_export_clusters.disk_gb="~{disk_size_gb}" \
      plink2_export_clusters.disk_type="~{disk_type}"

    cluster_start=$(date +%s.%N)
    plink2 \
      --pfile "$PREFIX" \
      --keep "$KEEP" \
      --export vcf-4.2 bgz id-paste=iid \
      --output-chr chrM \
      --out "$OUT_PREFIX" \
      --threads ~{cpu} \
      ~{extra_args}
    cluster_end=$(date +%s.%N)
    wall_s=$(awk -v a="$cluster_end" -v b="$cluster_start" 'BEGIN {printf "%.3f", a-b}')

    bcftools index --tbi --threads ~{cpu} "${OUT_PREFIX}.vcf.gz"

    OUTPUT_BYTES=$(stat -c %s "${OUT_PREFIX}.vcf.gz")
    # /proc/self/status VmHWM is in KiB (man proc(5)).
    PEAK_RSS_KB=$(awk '/^VmHWM:/ {print $2}' /proc/self/status 2>/dev/null || echo 0)
    PEAK_RSS_GB=$(awk -v k="$PEAK_RSS_KB" 'BEGIN {printf "%.3f", k/1048576}')

    magicwand log \
      plink2_export_clusters.wall_s="$wall_s" \
      plink2_export_clusters.output_bytes="$OUTPUT_BYTES" \
      plink2_export_clusters.peak_rss_gb="$PEAK_RSS_GB"

    ls -lh "$OUT_DIR"
  >>>

  output {
    # glob() so Cromwell's dynamic post-task delocalization picks the files
    # up. The directory holds exactly one VCF + one .tbi.
    File subset_vcf   = glob("out/*.vcf.gz")[0]
    File subset_index = glob("out/*.vcf.gz.tbi")[0]
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow plink2_export_clusters {
  input {
    File   pgen
    File   pvar
    File   psam
    File   cluster_sample_list
    String cluster_id

    String  extra_args = ""

    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override
    String  disk_type   = "HDD"
    Int     preemptible = 1

    String? wandb_api_key
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/plink2:latest"
  }

  call plink2_export_clusters_task {
    input:
      pgen                  = pgen,
      pvar                  = pvar,
      psam                  = psam,
      cluster_sample_list   = cluster_sample_list,
      cluster_id            = cluster_id,
      extra_args            = extra_args,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      wandb_api_key         = wandb_api_key,
      docker_image          = docker_image
  }

  output {
    File subset_vcf   = plink2_export_clusters_task.subset_vcf
    File subset_index = plink2_export_clusters_task.subset_index
  }
}
