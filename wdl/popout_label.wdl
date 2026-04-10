version 1.0

## Label inferred ancestries using 1KG superpopulation reference frequencies.
##
## One-stop workflow: just provide popout outputs and the task handles the rest.
## It will auto-download 1KG data and build the reference if needed.
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

task popout_label_task {
  input {
    File   model_npz
    File   global_ancestry
    File   tracts
    String output_prefix = "popout.labeled"
    String genome        = "GRCh38"

    # Optional — if omitted, 1KG VCFs are auto-downloaded and reference is built
    File?        reference
    Array[File]  kg_vcfs  = []
    File?        kg_panel
    Float        min_maf  = 0.01

    # Runtime
    Int    cpu           = 4
    String memory        = "16 GB"
    Int    extra_disk_gb = 50
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  Int tracts_size_gb = ceil(size(tracts, "GB"))
  Int vcf_size_gb    = ceil(size(kg_vcfs, "GB"))
  Int disk_size_gb   = 2 * tracts_size_gb + vcf_size_gb + extra_disk_gb

  command <<<
    set -euo pipefail

    # ---- Resolve reference ----
    REF_PATH="~{default='' reference}"

    if [ -n "$REF_PATH" ]; then
      echo "=== Using provided reference ==="
      ls -lh "$REF_PATH"

    elif [ ~{length(kg_vcfs)} -gt 0 ]; then
      echo "=== Building reference from ~{length(kg_vcfs)} provided VCF(s) ==="
      popout build-ref \
        --vcf ~{sep=' ' kg_vcfs} \
        ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
        --min-maf ~{min_maf} \
        --out built_ref.tsv.gz
      REF_PATH="built_ref.tsv.gz"

    else
      echo "=== Auto-downloading 1KG data and building reference ==="
      popout build-ref \
        --download \
        --genome ~{genome} \
        --min-maf ~{min_maf} \
        --out built_ref.tsv.gz
      REF_PATH="built_ref.tsv.gz"
    fi

    ls -lh "$REF_PATH"

    # ---- Label ----
    echo "=== Labeling ancestries ==="
    ls -lh ~{model_npz} ~{global_ancestry} ~{tracts}

    popout label \
      --model ~{model_npz} \
      --global ~{global_ancestry} \
      --tracts ~{tracts} \
      --reference "$REF_PATH" \
      --genome ~{genome} \
      --out ~{output_prefix}

    echo "=== Output files ==="
    ls -lh ~{output_prefix}.*
  >>>

  output {
    File labeled_global   = "~{output_prefix}.global.tsv"
    File labeled_tracts   = "~{output_prefix}.tracts.tsv.gz"
    File labels_json      = "~{output_prefix}.labels.json"
    File? built_reference = "built_ref.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }

  meta {
    description: "Label inferred ancestries using 1KG superpopulation reference"
  }
}

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

  call popout_label_task {
    input:
      model_npz       = model_npz,
      global_ancestry = global_ancestry,
      tracts          = tracts,
      reference       = reference,
      kg_vcfs         = kg_vcfs,
      kg_panel        = kg_panel,
      min_maf         = min_maf,
      output_prefix   = output_prefix,
      genome          = genome,
      docker_image    = docker_image
  }

  output {
    File labeled_global   = popout_label_task.labeled_global
    File labeled_tracts   = popout_label_task.labeled_tracts
    File labels_json      = popout_label_task.labels_json
    File? built_reference = popout_label_task.built_reference
  }
}
