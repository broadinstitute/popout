version 1.0

## Convert popout outputs to FLARE-compatible ancestry VCF.
##
## Runs 'popout convert --to vcf' on CPU (no GPU required).
## Input: popout native outputs + original phased VCF.
## Output: .anc.vcf.gz + .global.anc.gz

task convert_task {
  input {
    String      output_prefix
    File        tracts_tsv_gz
    File        model_npz
    File        global_tsv
    Array[File] decode_parquet
    File        input_vcf
    File        input_vcf_tbi
    Boolean     write_probs        = false
    String?     ancestry_names
    String      thinned_sites_mode = "skip"

    # Runtime
    Int    cpu       = 16
    Int    memory_gb = 64
    Int    disk_gb   = 500
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  command <<<
    set -euo pipefail

    # Reassemble popout outputs into a single directory
    mkdir -p popout_outputs
    ln -sf ~{model_npz} "popout_outputs/~{output_prefix}.model.npz"
    ln -sf ~{global_tsv} "popout_outputs/~{output_prefix}.global.tsv"
    ln -sf ~{tracts_tsv_gz} "popout_outputs/~{output_prefix}.tracts.tsv.gz"

    decode_files=(~{sep=' ' decode_parquet})
    for f in "${decode_files[@]}"; do
      ln -sf "$f" "popout_outputs/$(basename $f)"
    done

    echo "=== Assembled popout outputs ==="
    ls -lh popout_outputs/

    # Build convert command
    CMD="popout convert --to vcf"
    CMD="$CMD --popout-prefix popout_outputs/~{output_prefix}"
    CMD="$CMD --input-vcf ~{input_vcf}"
    CMD="$CMD --out output.anc.vcf.gz"
    CMD="$CMD --thinned-sites ~{thinned_sites_mode}"
    ~{if write_probs then 'CMD="$CMD --probs"' else ''}
    ~{if defined(ancestry_names) then 'CMD="$CMD --ancestry-names ~{ancestry_names}"' else ''}

    echo "=== Running: $CMD ==="
    eval "$CMD"

    echo "=== Outputs ==="
    ls -lh output.*
  >>>

  output {
    File anc_vcf_gz   = "output.anc.vcf.gz"
    File global_anc_gz = "output.global.anc.gz"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: "~{memory_gb} GB"
    disks:  "local-disk ~{disk_gb} SSD"
  }
}

workflow popout_convert {
  input {
    String      output_prefix
    File        tracts_tsv_gz
    File        model_npz
    File        global_tsv
    Array[File] decode_parquet
    File        input_vcf
    File        input_vcf_tbi
    Boolean     write_probs        = false
    String?     ancestry_names
    String      thinned_sites_mode = "skip"

    # Runtime
    Int    cpu       = 16
    Int    memory_gb = 64
    Int    disk_gb   = 500
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  call convert_task {
    input:
      output_prefix      = output_prefix,
      tracts_tsv_gz      = tracts_tsv_gz,
      model_npz          = model_npz,
      global_tsv         = global_tsv,
      decode_parquet         = decode_parquet,
      input_vcf          = input_vcf,
      input_vcf_tbi      = input_vcf_tbi,
      write_probs        = write_probs,
      ancestry_names     = ancestry_names,
      thinned_sites_mode = thinned_sites_mode,
      cpu                = cpu,
      memory_gb          = memory_gb,
      disk_gb            = disk_gb,
      docker_image       = docker_image,
  }

  output {
    File anc_vcf_gz    = convert_task.anc_vcf_gz
    File global_anc_gz = convert_task.global_anc_gz
  }
}
