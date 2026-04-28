version 1.0

## Label inferred ancestries using 1KG superpopulation allele frequencies.
##
## One-stop workflow: just provide popout outputs and the task handles the rest.
## It will auto-download 1KG data and build the superpop-freqs TSV if needed.
##
## The build is scattered across chromosomes for parallelism — each
## chromosome is downloaded and processed independently, then gathered
## into a single superpop-freqs file before labeling.
##
## Optionally provide a pre-built `superpop_freqs` file to skip the build
## step, or `kg_vcfs` to build from specific VCF files.
##
## Inputs (required):
##   - .model.npz, .global.tsv, .tracts.tsv.gz from a popout run
##
## Inputs (optional — zero-config by default):
##   - `superpop_freqs`: pre-built superpop allele-frequency TSV (skips build, fastest)
##   - `kg_vcfs`:        specific 1KG VCFs to build from
##
## Outputs: labeled versions of global and tracts files, plus a labels.json
## metadata report with correlation scores and assignment details.

# ---------------------------------------------------------------------------
# Task: download + process a single chromosome from 1KG
# ---------------------------------------------------------------------------
task build_superpop_freqs_chrom_task {
  input {
    String chrom
    String genome   = "GRCh38"
    File?  kg_panel
    Float  min_maf  = 0.01
    String docker_image
  }

  command <<<
    set -euo pipefail
    popout build-superpop-freqs \
      --download \
      --genome ~{genome} \
      --chromosomes ~{chrom} \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      --min-maf ~{min_maf} \
      --out chrom_freqs.tsv.gz
  >>>

  output {
    File chrom_freqs = "chrom_freqs.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "8 GB"
    disks:  "local-disk 10 SSD"
  }
}

# ---------------------------------------------------------------------------
# Task: process a single provided VCF file
# ---------------------------------------------------------------------------
task build_superpop_freqs_vcf_task {
  input {
    File   vcf_file
    File?  kg_panel
    Float  min_maf  = 0.01
    String docker_image
  }

  Int disk_gb = ceil(size(vcf_file, "GB")) + 5

  command <<<
    set -euo pipefail
    popout build-superpop-freqs \
      --vcf ~{vcf_file} \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      --min-maf ~{min_maf} \
      --out chrom_freqs.tsv.gz
  >>>

  output {
    File chrom_freqs = "chrom_freqs.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "8 GB"
    disks:  "local-disk ~{disk_gb} SSD"
  }
}

# ---------------------------------------------------------------------------
# Task: concatenate per-chromosome superpop-freqs files into one
# ---------------------------------------------------------------------------
task gather_superpop_freqs_task {
  input {
    Array[File] chrom_freqs
    String      docker_image
  }

  command <<<
    set -euo pipefail
    REFS=(~{sep=' ' chrom_freqs})

    # Header from first file, then data rows from all files
    zgrep -m1 '^' "${REFS[0]}" > combined.tsv
    for f in "${REFS[@]}"; do
      zcat "$f" | tail -n +2 >> combined.tsv
    done

    gzip combined.tsv
    mv combined.tsv.gz built_superpop_freqs.tsv.gz
    echo "Gathered ${#REFS[@]} files -> $(wc -l < <(zcat built_superpop_freqs.tsv.gz)) lines"
  >>>

  output {
    File superpop_freqs = "built_superpop_freqs.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "4 GB"
    disks:  "local-disk 5 SSD"
  }
}

# ---------------------------------------------------------------------------
# Task: label ancestries using a superpop-freqs TSV
# ---------------------------------------------------------------------------
task label_task {
  input {
    File   model_npz
    File   global_ancestry
    File   tracts
    File   superpop_freqs
    String genome        = "GRCh38"
    String output_prefix = "popout.labeled"

    Int    cpu           = 4
    String memory        = "16 GB"
    Int    extra_disk_gb = 10
    String docker_image
  }

  Int tracts_size_gb         = ceil(size(tracts, "GB"))
  Int superpop_freqs_size_gb = ceil(size(superpop_freqs, "GB"))
  Int disk_size_gb           = 2 * tracts_size_gb + superpop_freqs_size_gb + extra_disk_gb

  command <<<
    set -euo pipefail

    echo "=== Labeling ancestries ==="
    ls -lh ~{model_npz} ~{global_ancestry} ~{tracts} ~{superpop_freqs}

    popout label \
      --model ~{model_npz} \
      --global ~{global_ancestry} \
      --tracts ~{tracts} \
      --superpop-freqs ~{superpop_freqs} \
      --genome ~{genome} \
      --out ~{output_prefix}

    echo "=== Output files ==="
    ls -lh ~{output_prefix}.*
  >>>

  output {
    File labeled_global = "~{output_prefix}.global.tsv"
    File labeled_tracts = "~{output_prefix}.tracts.tsv.gz"
    File labels_json    = "~{output_prefix}.labels.json"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------
workflow popout_label {
  input {
    # popout outputs (required)
    File   model_npz
    File   global_ancestry
    File   tracts

    # Superpop frequencies (all optional — auto-builds from 1KG by default)
    File?        superpop_freqs
    Array[File]  kg_vcfs  = []
    File?        kg_panel

    String output_prefix = "popout.labeled"
    String genome        = "GRCh38"
    Float  min_maf       = 0.01

    # Runtime
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  Array[String] autosomes = [
    "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22"
  ]

  Boolean need_download  = !defined(superpop_freqs) && length(kg_vcfs) == 0
  Boolean need_vcf_build = !defined(superpop_freqs) && length(kg_vcfs) > 0

  # Path A: auto-download — scatter across chromosomes
  if (need_download) {
    scatter (chrom in autosomes) {
      call build_superpop_freqs_chrom_task {
        input:
          chrom        = chrom,
          genome       = genome,
          kg_panel     = kg_panel,
          min_maf      = min_maf,
          docker_image = docker_image
      }
    }

    call gather_superpop_freqs_task as gather_download {
      input:
        chrom_freqs  = build_superpop_freqs_chrom_task.chrom_freqs,
        docker_image = docker_image
    }
  }

  # Path B: provided VCFs — scatter across files
  if (need_vcf_build) {
    scatter (vcf in kg_vcfs) {
      call build_superpop_freqs_vcf_task {
        input:
          vcf_file     = vcf,
          kg_panel     = kg_panel,
          min_maf      = min_maf,
          docker_image = docker_image
      }
    }

    call gather_superpop_freqs_task as gather_vcf {
      input:
        chrom_freqs  = build_superpop_freqs_vcf_task.chrom_freqs,
        docker_image = docker_image
    }
  }

  # Path C: pre-built superpop-freqs — used directly
  File effective_freqs = select_first([
    superpop_freqs, gather_download.superpop_freqs, gather_vcf.superpop_freqs
  ])

  call label_task {
    input:
      model_npz       = model_npz,
      global_ancestry = global_ancestry,
      tracts          = tracts,
      superpop_freqs  = effective_freqs,
      genome          = genome,
      output_prefix   = output_prefix,
      docker_image    = docker_image
  }

  output {
    File  labeled_global         = label_task.labeled_global
    File  labeled_tracts         = label_task.labeled_tracts
    File  labels_json            = label_task.labels_json
    File? built_superpop_freqs   = if need_download then gather_download.superpop_freqs
                                    else gather_vcf.superpop_freqs
  }
}
