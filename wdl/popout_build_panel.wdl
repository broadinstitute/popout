version 1.0

## Build a per-population allele-frequency panel TSV.
##
## Two source kinds are supported, dispatched by `source`:
##   - "1kg":    per-sample VCFs (1000 Genomes Phase 3 genotype data).
##               Auto-downloads from public FTP unless `vcfs` is set.
##   - "gnomad": sites-only VCFs with AC_<pop>/AN_<pop> INFO fields.
##               Requires `vcfs` (no auto-download).
##
## Both paths scatter across input shards (chromosomes for 1KG --download,
## one task per file for everything else) and gather into a single panel.
##
## Inputs:
##   - source:       "1kg" | "gnomad"  (default: "1kg")
##   - vcfs:         optional Array[File]; for 1KG, providing this skips
##                   the auto-download path. Required for gnomad.
##   - chromosomes:  used by 1KG --download path (default: 1..22)
##   - pop_config:   optional JSON with output_order + rules
##   - kg_panel:     optional 1KG sample-panel file (1KG only)
##   - min_maf:      global MAF filter (default 0.01)
##   - pass_only:    gnomad PASS-only filter (default true; ignored for 1KG)
##
## Output:
##   - panel: gathered #chrom/pos/ref/alt/<POP>... TSV.gz

# ---------------------------------------------------------------------------
# Task: process one shard (one chromosome download or one provided VCF)
# ---------------------------------------------------------------------------
task build_panel_chrom_task {
  input {
    String chrom
    String genome   = "GRCh38"
    File?  kg_panel
    File?  pop_config
    Float  min_maf  = 0.01
    String docker_image
  }

  command <<<
    set -euo pipefail
    popout build-panel \
      --source 1kg \
      --download \
      --genome ~{genome} \
      --chromosomes ~{chrom} \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      ~{if defined(pop_config) then '--pop-config ~{pop_config}' else ''} \
      --min-maf ~{min_maf} \
      --out shard_panel.tsv.gz
  >>>

  output {
    File shard_panel = "shard_panel.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "8 GB"
    disks:  "local-disk 10 SSD"
  }
}

task build_panel_vcf_task {
  input {
    String source
    File   vcf_file
    File?  kg_panel
    File?  pop_config
    Float  min_maf      = 0.01
    Boolean pass_only   = true
    String docker_image
  }

  Int disk_gb = ceil(size(vcf_file, "GB")) + 5

  command <<<
    set -euo pipefail
    popout build-panel \
      --source ~{source} \
      --vcf ~{vcf_file} \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      ~{if defined(pop_config) then '--pop-config ~{pop_config}' else ''} \
      --min-maf ~{min_maf} \
      ~{if pass_only then '--pass-only' else '--no-pass-only'} \
      --out shard_panel.tsv.gz
  >>>

  output {
    File shard_panel = "shard_panel.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "8 GB"
    disks:  "local-disk ~{disk_gb} SSD"
  }
}

# ---------------------------------------------------------------------------
# Task: concatenate per-shard panel files into one
# ---------------------------------------------------------------------------
task gather_panel_task {
  input {
    Array[File] shard_panels
    String      docker_image
  }

  command <<<
    set -euo pipefail
    SHARDS=(~{sep=' ' shard_panels})

    # Header from first file, then data rows from all files
    zgrep -m1 '^' "${SHARDS[0]}" > combined.tsv
    for f in "${SHARDS[@]}"; do
      zcat "$f" | tail -n +2 >> combined.tsv
    done

    gzip combined.tsv
    mv combined.tsv.gz built_panel.tsv.gz
    echo "Gathered ${#SHARDS[@]} files -> $(wc -l < <(zcat built_panel.tsv.gz)) lines"
  >>>

  output {
    File panel = "built_panel.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "4 GB"
    disks:  "local-disk 5 SSD"
  }
}

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
workflow popout_build_panel {
  input {
    String       source        = "1kg"
    Array[File]  vcfs          = []
    String       genome        = "GRCh38"
    Array[String] chromosomes  = [
      "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10",
      "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
      "21", "22"
    ]

    File?   pop_config
    File?   kg_panel
    Float   min_maf   = 0.01
    Boolean pass_only = true

    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  Boolean use_download = source == "1kg" && length(vcfs) == 0

  if (use_download) {
    scatter (chrom in chromosomes) {
      call build_panel_chrom_task {
        input:
          chrom        = chrom,
          genome       = genome,
          kg_panel     = kg_panel,
          pop_config   = pop_config,
          min_maf      = min_maf,
          docker_image = docker_image
      }
    }

    call gather_panel_task as gather_download {
      input:
        shard_panels = build_panel_chrom_task.shard_panel,
        docker_image = docker_image
    }
  }

  if (!use_download) {
    scatter (vcf in vcfs) {
      call build_panel_vcf_task {
        input:
          source       = source,
          vcf_file     = vcf,
          kg_panel     = kg_panel,
          pop_config   = pop_config,
          min_maf      = min_maf,
          pass_only    = pass_only,
          docker_image = docker_image
      }
    }

    call gather_panel_task as gather_vcfs {
      input:
        shard_panels = build_panel_vcf_task.shard_panel,
        docker_image = docker_image
    }
  }

  output {
    File panel = select_first([gather_download.panel, gather_vcfs.panel])
  }
}
