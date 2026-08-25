version 1.0

## Lift specified `##KEY=…` header lines from a source VCF and inject them
## into a target VCF via `bcftools reheader`. Body variant records are
## streamed byte-for-byte; only the header block is rewritten.
##
## Use case: `bcftools merge` drops custom `##KEY=value` header lines it
## does not recognize. FLARE's `##ANCESTRY=<…>` is the canonical example.
## After a merge, run this task with the merged VCF as `vcf` and any one
## of the pre-merge inputs as `source_header_vcf` to restore the dropped
## lines.
##
## Design points:
##   - `header_keys` defaults to ["ANCESTRY"] but is caller-overridable so
##     the WDL is not FLARE-specific.
##   - Extraction is verbatim from the source VCF's own header (no
##     hardcoding, no regex reconstruction) per project data-fidelity
##     rules.
##   - Hard-fails if any requested key is missing or duplicated in
##     `source_header_vcf`, or if it fails to land in the reheadered
##     output.
##   - Output basename mirrors the input's (`chr1.anc.vcf.gz` in →
##     `chr1.anc.vcf.gz` out). Reheadered file lands in the task cwd, so
##     no path collision with the localized input.
##   - `bcftools reheader` rewrites the header block; every downstream
##     bgzf virtual offset shifts, invalidating the input's tabix index.
##     A fresh `.tbi` is regenerated.

task bcftools_reheader_task {
  input {
    File          vcf
    File          source_header_vcf
    Array[String] header_keys = ["ANCESTRY"]

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type   = "HDD"
    Int      preemptible = 2

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  # Literal defaults, mirroring bcftools_view.wdl:86-91: size()-driven
  # auto-scaling fires in the input-evaluation phase before the file's
  # metadata is available in every Cromwell environment (observed on
  # Terra), so it's not safe here. 200 GB covers biobank-scale chr1
  # (~50 GB input + ~50 GB output + slack). Override for smaller chroms
  # if disk cost matters.
  Int    cpu          = select_first([cpu_override, 2])
  String memory       = select_first([memory_override, "4 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, 200])

  String stem      = basename(vcf, ".vcf.gz")
  String out_vcf   = "~{stem}.vcf.gz"
  String out_index = "~{stem}.vcf.gz.tbi"

  command <<<
    set -euo pipefail

    bcftools view -h "~{source_header_vcf}" > source.hdr
    bcftools view -h "~{vcf}"               > current.hdr

    : > additions.hdr
    while IFS= read -r key; do
      [ -z "$key" ] && continue
      matches=$(grep -c "^##${key}=" source.hdr || true)
      if [ "$matches" -ne 1 ]; then
        echo "ERROR: source_header_vcf has $matches lines matching '^##${key}=' (need exactly 1)" >&2
        grep "^##${key}=" source.hdr >&2 || true
        exit 1
      fi
      grep "^##${key}=" source.hdr >> additions.hdr
    done < ~{write_lines(header_keys)}

    # Splice additions immediately before the #CHROM line in the target's
    # existing header.
    awk 'FNR==NR{add = add $0 ORS; next} /^#CHROM/{printf "%s", add} {print}' \
      additions.hdr current.hdr > patched.hdr

    bcftools reheader --header patched.hdr "~{vcf}" -o "~{out_vcf}"
    bcftools index --tbi --threads ~{cpu} "~{out_vcf}"

    # Post-condition: every requested key must appear in the reheadered
    # output's header exactly once.
    while IFS= read -r key; do
      [ -z "$key" ] && continue
      n=$(bcftools view -h "~{out_vcf}" | grep -c "^##${key}=" || true)
      if [ "$n" -ne 1 ]; then
        echo "ERROR: post-reheader header has $n '##${key}=' lines (need exactly 1)" >&2
        exit 1
      fi
    done < ~{write_lines(header_keys)}
  >>>

  output {
    File reheadered_vcf       = "~{out_vcf}"
    File reheadered_index_tbi = "~{out_index}"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow bcftools_reheader {
  input {
    File          vcf
    File          source_header_vcf
    Array[String] header_keys = ["ANCESTRY"]

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type   = "HDD"
    Int      preemptible = 2

    String   docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/bcftools:latest"
  }

  call bcftools_reheader_task {
    input:
      vcf                   = vcf,
      source_header_vcf     = source_header_vcf,
      header_keys           = header_keys,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      docker_image          = docker_image
  }

  output {
    File reheadered_vcf       = bcftools_reheader_task.reheadered_vcf
    File reheadered_index_tbi = bcftools_reheader_task.reheadered_index_tbi
  }
}
