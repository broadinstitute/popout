version 1.0

## Label inferred ancestries using 1KG superpopulation reference frequencies.
##
## Inputs: the .model.npz, .global.tsv, and .tracts.tsv.gz from a popout run,
## plus a reference frequency file (built by scripts/build_1kg_ref.py).
##
## Outputs: labeled versions of global and tracts files, plus a labels.json
## metadata report with correlation scores and assignment details.
##
## This is a CPU-only task — no GPU needed.

task popout_label_task {
  input {
    File   model_npz
    File   global_ancestry
    File   tracts
    File   reference
    String output_prefix = "popout.labeled"
    String genome        = "GRCh38"

    # Runtime
    Int    cpu          = 2
    String memory       = "8 GB"
    Int    disk_size_gb = 50
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  command <<<
    set -euo pipefail

    echo "=== Input files ==="
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

  meta {
    description: "Label inferred ancestries using 1KG superpopulation reference"
  }
}

workflow popout_label {
  input {
    File   model_npz
    File   global_ancestry
    File   tracts
    File   reference
    String output_prefix = "popout.labeled"
    String genome        = "GRCh38"

    # Runtime
    Int    cpu          = 2
    String memory       = "8 GB"
    Int    disk_size_gb = 50
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  call popout_label_task {
    input:
      model_npz       = model_npz,
      global_ancestry = global_ancestry,
      tracts          = tracts,
      reference       = reference,
      output_prefix   = output_prefix,
      genome          = genome,
      cpu             = cpu,
      memory          = memory,
      disk_size_gb    = disk_size_gb,
      docker_image    = docker_image
  }

  output {
    File labeled_global = popout_label_task.labeled_global
    File labeled_tracts = popout_label_task.labeled_tracts
    File labels_json    = popout_label_task.labels_json
  }
}
