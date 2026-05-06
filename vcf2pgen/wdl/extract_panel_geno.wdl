version 1.0

## Build a single combined panel-only PGEN for popout's Phase 2
## (option H — μ-weighted off-chrom AIM frequencies).
##
## Per-chrom: plink2 --extract bed0 <aim_panel_bed> --make-pgen on
## the input PGEN.  Cohort filters are NOT applied — panel positions
## must survive intact regardless of MAF/HWE/etc.  Output is small:
## ≤ ~30 panel positions × all haplotypes per chrom.
##
## Gather: plink2 --pmerge-list across the per-chrom panel PGENs to
## produce one combined panel_geno.pgen covering every chrom's AIM
## panel positions × all haplotypes (≈ 1 M haps × 83 sites × 1 byte
## ≈ 80 MB total).
##
## Inputs are typically the cohort's RAW per-chrom PGENs (the same
## inputs you'd give to vcf2pgen/filter_pgen.wdl).  Cohort-filter
## outputs also work as long as `aim_panel_bed` was protected during
## filtering.
##
## Output is a single panel_geno.pgen / .pvar / .psam triplet that
## becomes popout.wdl's `panel_geno_pgen` input for Phase 2 runs.

task extract_panel_pgen_task {
  input {
    File   pgen
    File   pvar
    File   psam
    File   aim_panel_bed
    String output_prefix = basename(pgen, ".pgen") + ".panel"

    Int?    cpu_override
    String? memory_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  Float pgen_gb = size(pgen, "GB")
  Int   cpu    = select_first([cpu_override, 2])
  String memory = select_first([memory_override, "8G"])
  Int   disk_size_gb = ceil(pgen_gb * 2 + 10)

  command <<<
    set -euo pipefail

    # ---- Localize PGEN triplet under one prefix ----
    INPUT_PREFIX="input_pgen"
    ln -sf "~{pgen}" "${INPUT_PREFIX}.pgen"
    ln -sf "~{pvar}" "${INPUT_PREFIX}.pvar"
    ln -sf "~{psam}" "${INPUT_PREFIX}.psam"

    echo "=== Extract AIM panel positions (no cohort filtering, no allele filtering) ==="
    # NOTE: NO --max-alleles 2, NO --snps-only.  This was the cause of
    # AIM-protect appearing to "drop" positions during filter_pgen runs
    # — when an AIM position was called as multi-allelic in the cohort
    # PGEN, --max-alleles 2 stripped it.  extract_panel_geno must
    # extract every BED position that exists in the input PGEN, period.
    # Downstream consumers (popout option H) handle allele alignment
    # against the panel TSV's expected (ref, alt).
    plink2 \
      --pfile "${INPUT_PREFIX}" \
      --extract bed0 "~{aim_panel_bed}" \
      --make-pgen \
      --threads ~{cpu} \
      --out "~{output_prefix}"

    echo "=== Variant count ==="
    wc -l "~{output_prefix}.pvar"

    echo "=== Sample count ==="
    wc -l "~{output_prefix}.psam"
  >>>

  output {
    File panel_pgen = "~{output_prefix}.pgen"
    File panel_pvar = "~{output_prefix}.pvar"
    File panel_psam = "~{output_prefix}.psam"
    File log        = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

task gather_panel_pgens_task {
  input {
    Array[File] panel_pgens
    Array[File] panel_pvars
    Array[File] panel_psams
    String      output_prefix = "panel_geno"

    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  command <<<
    set -euo pipefail

    # ---- Localize each per-chrom triplet under a unique prefix ----
    pgens=(~{sep=' ' panel_pgens})
    pvars=(~{sep=' ' panel_pvars})
    psams=(~{sep=' ' panel_psams})

    PMERGE_LIST=panel_merge_list.txt
    : > "${PMERGE_LIST}"
    for i in "${!pgens[@]}"; do
      P="chunk_${i}"
      ln -sf "${pgens[$i]}" "${P}.pgen"
      ln -sf "${pvars[$i]}" "${P}.pvar"
      ln -sf "${psams[$i]}" "${P}.psam"
      echo "${P}" >> "${PMERGE_LIST}"
    done

    echo "=== Merge list ==="
    cat "${PMERGE_LIST}"

    echo "=== plink2 --pmerge-list ==="
    plink2 \
      --pmerge-list "${PMERGE_LIST}" pfile \
      --make-pgen \
      --out "~{output_prefix}"

    echo "=== Combined variant count ==="
    wc -l "~{output_prefix}.pvar"
    echo "=== Combined sample count ==="
    wc -l "~{output_prefix}.psam"
  >>>

  output {
    File panel_geno_pgen = "~{output_prefix}.pgen"
    File panel_geno_pvar = "~{output_prefix}.pvar"
    File panel_geno_psam = "~{output_prefix}.psam"
    File log             = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    2
    memory: "8G"
    disks:  "local-disk 50 SSD"
  }
}

workflow extract_panel_geno {
  input {
    Array[File] pgens
    Array[File] pvars
    Array[File] psams
    File        aim_panel_bed
    String      output_prefix = "panel_geno"
    String      docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/vcf2pgen:0.1.0"
  }

  scatter (idx in range(length(pgens))) {
    call extract_panel_pgen_task {
      input:
        pgen          = pgens[idx],
        pvar          = pvars[idx],
        psam          = psams[idx],
        aim_panel_bed = aim_panel_bed,
        docker_image  = docker_image,
    }
  }

  call gather_panel_pgens_task {
    input:
      panel_pgens   = extract_panel_pgen_task.panel_pgen,
      panel_pvars   = extract_panel_pgen_task.panel_pvar,
      panel_psams   = extract_panel_pgen_task.panel_psam,
      output_prefix = output_prefix,
      docker_image  = docker_image,
  }

  output {
    File        panel_geno_pgen = gather_panel_pgens_task.panel_geno_pgen
    File        panel_geno_pvar = gather_panel_pgens_task.panel_geno_pvar
    File        panel_geno_psam = gather_panel_pgens_task.panel_geno_psam
    Array[File] per_chrom_pgens = extract_panel_pgen_task.panel_pgen
    Array[File] per_chrom_pvars = extract_panel_pgen_task.panel_pvar
    Array[File] per_chrom_psams = extract_panel_pgen_task.panel_psam
    File        gather_log      = gather_panel_pgens_task.log
  }
}
