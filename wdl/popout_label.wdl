version 1.0

## Label inferred ancestries using 1KG superpopulation reference frequencies.
##
## One-stop workflow: provide either a pre-built reference frequency file OR
## 1000 Genomes VCF(s) and the workflow will build the reference automatically.
##
## Inputs:
##   - .model.npz, .global.tsv, .tracts.tsv.gz from a popout run
##   - EITHER `reference` (pre-built superpop freq TSV, reusable across runs)
##     OR `kg_vcfs` (1KG Phase 3 VCFs to build it from)
##
## Outputs: labeled versions of global and tracts files, plus a labels.json
## metadata report with correlation scores and assignment details.

task build_reference_task {
  input {
    Array[File]+ kg_vcfs
    File?        kg_panel
    Float        min_maf      = 0.01
    String       genome       = "GRCh38"

    # Runtime
    Int    cpu          = 4
    String memory       = "16 GB"
    Int    extra_disk_gb = 20
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  Int vcf_size_gb = ceil(size(kg_vcfs, "GB"))
  Int disk_size_gb = vcf_size_gb + extra_disk_gb

  command <<<
    set -euo pipefail

    echo "=== Building 1KG superpopulation reference ==="

    # Write VCF paths to a file list for reliable shell handling
    cat > vcf_list.txt <<'VCFEOF'
    ~{sep='\n' kg_vcfs}
    VCFEOF
    sed -i 's/^[[:space:]]*//' vcf_list.txt
    echo "Input VCFs: $(wc -l < vcf_list.txt)"

    popout build-ref \
      --vcf $(cat vcf_list.txt | tr '\n' ' ') \
      ~{if defined(kg_panel) then '--panel ~{kg_panel}' else ''} \
      --min-maf ~{min_maf} \
      --out "1kg_superpop_freq.~{genome}.tsv.gz"

    ls -lh 1kg_superpop_freq.*.tsv.gz
  >>>

  output {
    File reference = "1kg_superpop_freq.~{genome}.tsv.gz"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }

  meta {
    description: "Build superpopulation frequency reference from 1KG VCFs"
  }
}

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
    String memory       = "16 GB"
    Int    extra_disk_gb = 20
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  Int tracts_size_gb = ceil(size(tracts, "GB"))
  Int ref_size_gb    = ceil(size(reference, "GB"))
  Int disk_size_gb   = 2 * tracts_size_gb + ref_size_gb + extra_disk_gb

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
    # popout outputs
    File   model_npz
    File   global_ancestry
    File   tracts

    # Reference: provide ONE of these
    File?        reference     # pre-built superpop freq TSV (fast — reuse across runs)
    Array[File]? kg_vcfs       # 1KG Phase 3 VCFs (builds reference, then labels)
    File?        kg_panel      # 1KG sample panel (auto-downloaded if omitted)

    String output_prefix = "popout.labeled"
    String genome        = "GRCh38"
    Float  min_maf       = 0.01

    # Runtime
    String docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  # Build reference from 1KG VCFs if no pre-built reference provided
  if (!defined(reference) && defined(kg_vcfs)) {
    call build_reference_task {
      input:
        kg_vcfs      = select_first([kg_vcfs]),
        kg_panel     = kg_panel,
        min_maf      = min_maf,
        genome       = genome,
        docker_image = docker_image
    }
  }

  # Resolve: user-provided reference takes priority; otherwise use the one we just built.
  # Exactly one will be defined: either `reference` (user gave it) or
  # `build_reference_task.reference` (we built it from kg_vcfs above).
  File ref_file = select_first([reference, build_reference_task.reference])

  call popout_label_task {
    input:
      model_npz       = model_npz,
      global_ancestry = global_ancestry,
      tracts          = tracts,
      reference       = ref_file,
      output_prefix   = output_prefix,
      genome          = genome,
      docker_image    = docker_image
  }

  output {
    File labeled_global = popout_label_task.labeled_global
    File labeled_tracts = popout_label_task.labeled_tracts
    File labels_json    = popout_label_task.labels_json
    File? built_reference = build_reference_task.reference
  }
}
