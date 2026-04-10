version 1.0

## Label inferred ancestries using 1KG superpopulation reference frequencies.
##
## One-stop workflow: just provide popout outputs and the task handles the rest.
## It will auto-download 1KG data and build the reference if needed.
##
## The reference build is scattered across chromosomes for parallelism —
## each chromosome is downloaded and processed independently, then gathered
## into a single reference file before labeling.
##
## Optionally provide a pre-built `reference` file to skip the build step,
## or `kg_vcfs` to build from specific VCF files.
##
## Inputs (required):
##   - .model.npz, .global.tsv, .tracts.tsv.gz from a popout run
##
## Inputs (optional — zero-config by default):
##   - `reference`: pre-built superpop freq TSV (skips build, fastest)
##   - `kg_vcfs`:   specific 1KG VCFs to build from
##
## Outputs: labeled versions of global and tracts files, plus a labels.json
## metadata report with correlation scores and assignment details.

# ---------------------------------------------------------------------------
# Task: download + process a single chromosome from 1KG
# ---------------------------------------------------------------------------
task build_ref_chrom_task {
  input {
    String chrom
    String genome   = "GRCh38"
    File?  kg_panel
    Float  min_maf  = 0.01
    String docker_image
  }

  command <<<
    set -euo pipefail
    popout build-ref \
      --download \
      --genome ~{genome} \
      --chromosomes ~{chrom} \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      --min-maf ~{min_maf} \
      --out chrom_ref.tsv.gz
  >>>

  output {
    File chrom_ref = "chrom_ref.tsv.gz"
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
task build_ref_vcf_task {
  input {
    File   vcf_file
    File?  kg_panel
    Float  min_maf  = 0.01
    String docker_image
  }

  Int disk_gb = ceil(size(vcf_file, "GB")) + 5

  command <<<
    set -euo pipefail
    popout build-ref \
      --vcf ~{vcf_file} \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      --min-maf ~{min_maf} \
      --out chrom_ref.tsv.gz
  >>>

  output {
    File chrom_ref = "chrom_ref.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "8 GB"
    disks:  "local-disk ~{disk_gb} SSD"
  }
}

# ---------------------------------------------------------------------------
# Task: concatenate per-chromosome reference files into one
# ---------------------------------------------------------------------------
task gather_ref_task {
  input {
    Array[File] chrom_refs
    String      docker_image
  }

  command <<<
    set -euo pipefail
    REFS=(~{sep=' ' chrom_refs})

    # Header from first file, then data rows from all files
    zcat "${REFS[0]}" | head -1 > combined.tsv
    for f in "${REFS[@]}"; do
      zcat "$f" | tail -n +2 >> combined.tsv
    done

    gzip combined.tsv
    mv combined.tsv.gz built_ref.tsv.gz
    echo "Gathered ${#REFS[@]} files -> $(wc -l < <(zcat built_ref.tsv.gz)) lines"
  >>>

  output {
    File reference = "built_ref.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "4 GB"
    disks:  "local-disk 5 SSD"
  }
}

# ---------------------------------------------------------------------------
# Task: label ancestries using a reference
# ---------------------------------------------------------------------------
task label_task {
  input {
    File   model_npz
    File   global_ancestry
    File   tracts
    File   reference
    String genome        = "GRCh38"
    String output_prefix = "popout.labeled"

    Int    cpu           = 4
    String memory        = "16 GB"
    Int    extra_disk_gb = 10
    String docker_image
  }

  Int tracts_size_gb = ceil(size(tracts, "GB"))
  Int ref_size_gb    = ceil(size(reference, "GB"))
  Int disk_size_gb   = 2 * tracts_size_gb + ref_size_gb + extra_disk_gb

  command <<<
    set -euo pipefail

    echo "=== Labeling ancestries ==="
    ls -lh ~{model_npz} ~{global_ancestry} ~{tracts} ~{reference}

    popout label \
      --model ~{model_npz} \
      --global ~{global_ancestry} \
      --tracts ~{tracts} \
      --reference ~{reference} \
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

    # Reference (all optional — auto-builds from 1KG by default)
    File?        reference
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

  Boolean need_download  = !defined(reference) && length(kg_vcfs) == 0
  Boolean need_vcf_build = !defined(reference) && length(kg_vcfs) > 0

  # Path A: auto-download — scatter across chromosomes
  if (need_download) {
    scatter (chrom in autosomes) {
      call build_ref_chrom_task {
        input:
          chrom        = chrom,
          genome       = genome,
          kg_panel     = kg_panel,
          min_maf      = min_maf,
          docker_image = docker_image
      }
    }

    call gather_ref_task as gather_download {
      input:
        chrom_refs   = build_ref_chrom_task.chrom_ref,
        docker_image = docker_image
    }
  }

  # Path B: provided VCFs — scatter across files
  if (need_vcf_build) {
    scatter (vcf in kg_vcfs) {
      call build_ref_vcf_task {
        input:
          vcf_file     = vcf,
          kg_panel     = kg_panel,
          min_maf      = min_maf,
          docker_image = docker_image
      }
    }

    call gather_ref_task as gather_vcf {
      input:
        chrom_refs   = build_ref_vcf_task.chrom_ref,
        docker_image = docker_image
    }
  }

  # Path C: pre-built reference — used directly
  File effective_ref = select_first([reference, gather_download.reference, gather_vcf.reference])

  call label_task {
    input:
      model_npz       = model_npz,
      global_ancestry = global_ancestry,
      tracts          = tracts,
      reference       = effective_ref,
      genome          = genome,
      output_prefix   = output_prefix,
      docker_image    = docker_image
  }

  output {
    File  labeled_global   = label_task.labeled_global
    File  labeled_tracts   = label_task.labeled_tracts
    File  labels_json      = label_task.labels_json
    File? built_reference  = if need_download then gather_download.reference
                             else gather_vcf.reference
  }
}
