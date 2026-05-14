version 1.0

## Byte-balanced genomic partitions for streaming scatter.
##
## Walks a tabix index (.tbi) and emits roughly equal-byte work units that
## downstream tasks (e.g. bcftools_split_streaming) consume to scatter over
## `gs://` URLs without ever localizing the whole VCF. High-density loci
## (HLA, KIR) get small-bp but normal-byte partitions automatically — see
## `scripts/generate_partitions.py` for the algorithm.
##
## The .tbi *is* localized (it's small, ~300 KB for AoU chr1); only the
## main VCF stays remote.
##
## Why MB inputs (not bytes): WDL Int is 32-bit in Cromwell/Rawls
## (max ~2.1 GB). 10 GB and 30 GB literal byte counts overflow. The task
## converts MB -> bytes via bash arithmetic (which is 64-bit-safe) before
## invoking the Python helper.

task generate_genomic_partitions {
  input {
    File          vcf_index                       # localized .tbi
    Array[String] chromosomes = []                # empty = all contigs in the index
    Int           target_mb_per_partition = 10240     # 10 GB
    Int           max_mb_per_partition    = 30720     # 30 GB safety valve
    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  command <<<
    set -euo pipefail

    TARGET_BYTES=$(( ~{target_mb_per_partition} * 1048576 ))
    MAX_BYTES=$(( ~{max_mb_per_partition} * 1048576 ))

    generate_partitions.py \
      --tbi-path "~{vcf_index}" \
      ~{if length(chromosomes) > 0 then "--chromosomes" else ""} ~{sep=' ' chromosomes} \
      --target-bytes-per-partition "$TARGET_BYTES" \
      --max-bytes-per-partition "$MAX_BYTES" \
      --out-manifest partitions.tsv \
      --out-regions regions.txt \
      --out-region-ids region_ids.txt
  >>>

  output {
    File          manifest    = "partitions.tsv"
    Array[String] regions     = read_lines("regions.txt")
    Array[String] region_ids  = read_lines("region_ids.txt")
  }

  runtime {
    docker: docker_image
    cpu:    1
    memory: "2 GB"
    disks:  "local-disk 5 HDD"
  }
}

workflow generate_genomic_partitions_workflow {
  input {
    File          vcf_index
    Array[String] chromosomes = []
    Int           target_mb_per_partition = 10240
    Int           max_mb_per_partition    = 30720
    String        docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  call generate_genomic_partitions {
    input:
      vcf_index               = vcf_index,
      chromosomes             = chromosomes,
      target_mb_per_partition = target_mb_per_partition,
      max_mb_per_partition    = max_mb_per_partition,
      docker_image            = docker_image
  }

  output {
    File          manifest    = generate_genomic_partitions.manifest
    Array[String] regions     = generate_genomic_partitions.regions
    Array[String] region_ids  = generate_genomic_partitions.region_ids
  }
}
