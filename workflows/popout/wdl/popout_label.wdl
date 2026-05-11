version 1.0

## Label inferred ancestries against a pre-built population-frequency panel.
##
## This workflow is now build-free. To produce the `reference` panel input,
## run `popout_build_panel.wdl` (or pass any compatible TSV with the
## #chrom/pos/ref/alt/<POP>... schema).
##
## Inputs (required):
##   - .model.npz, .global.tsv, .tracts.tsv.gz from a popout run
##   - reference: panel TSV (e.g. output of popout_build_panel.wdl)
##
## Outputs: labeled global and tracts files, plus a labels.json metadata
## report with correlation scores and assignment details.

# ---------------------------------------------------------------------------
# Task: label ancestries using a reference panel
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
    File   model_npz
    File   global_ancestry
    File   tracts
    File   reference

    String output_prefix = "popout.labeled"
    String genome        = "GRCh38"

    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  call label_task {
    input:
      model_npz       = model_npz,
      global_ancestry = global_ancestry,
      tracts          = tracts,
      reference       = reference,
      genome          = genome,
      output_prefix   = output_prefix,
      docker_image    = docker_image
  }

  output {
    File labeled_global = label_task.labeled_global
    File labeled_tracts = label_task.labeled_tracts
    File labels_json    = label_task.labels_json
  }
}
