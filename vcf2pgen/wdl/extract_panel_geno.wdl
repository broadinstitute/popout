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

  # Auto-scale machine to PGEN size, mirroring vcf2pgen/wdl/filter_pgen.wdl.
  # plink2 --extract bed0 still loads pvar/pgen indexes and per-variant
  # state into memory; raw AoU per-chrom PGENs run 30-100 GB and need
  # tens-to-hundreds of GB of RAM.  The 8 GB default OOM'd biobank-scale
  # shards.
  Int auto_cpu = if pgen_gb > 60.0 then 64
                 else if pgen_gb > 30.0 then 32
                 else if pgen_gb > 10.0 then 16
                 else 8

  String auto_memory = if pgen_gb > 60.0 then "256 GB"
                       else if pgen_gb > 30.0 then "128 GB"
                       else if pgen_gb > 10.0 then "64 GB"
                       else "32 GB"

  Int    cpu         = select_first([cpu_override, auto_cpu])
  String memory      = select_first([memory_override, auto_memory])
  Int    disk_size_gb = ceil(pgen_gb * 3) + 100

  command <<<
    set -euo pipefail

    # ---- Localize PGEN triplet under one prefix ----
    INPUT_PREFIX="input_pgen"
    ln -sf "~{pgen}" "${INPUT_PREFIX}.pgen"
    ln -sf "~{pvar}" "${INPUT_PREFIX}.pvar"
    ln -sf "~{psam}" "${INPUT_PREFIX}.psam"

    # Three-step extract to recover multi-allelic AIM panel positions:
    #
    #   1. plink2 → VCF: extract panel positions, KEEP multi-allelic
    #      (the BED region selects the locus regardless of allele count).
    #   2. bcftools norm -m -any: split each multi-allelic site into
    #      one biallelic row per alt allele. After this every row is
    #      (chrom, pos, ref, single_alt) — uniquely keyed by the
    #      tuple, so popout's panel-freq lookup can pick the alt
    #      matching the panel TSV's expected (ref, alt).
    #   3. plink2 ← VCF: re-encode to PGEN with --max-alleles 2 (a
    #      defensive no-op after norm; rejects anything bcftools
    #      didn't fully split).
    #
    # Without this, multi-allelic panel positions (additional rare
    # alts AoU calls at sites the panel BED expected as biallelic
    # SNPs) crash popout's pgenlib reader: phased + multi-allelic is
    # not a supported combination.

    # Some chroms have AIM panel positions that AoU's site set simply
    # doesn't include (e.g. chr7's two AMR/MID markers, chr21's one
    # AMR marker). plink2's --extract bed0 errors with "No variants
    # remaining" when the per-chrom intersection is empty. We'd
    # rather emit empty placeholder outputs and let the gather skip
    # this chrom than fail the whole 22-shard scatter.
    echo "=== Step 1/3: extract panel region to VCF (keeps multi-allelic) ==="
    if ! plink2 \
        --pfile "${INPUT_PREFIX}" \
        --extract bed0 "~{aim_panel_bed}" \
        --snps-only just-acgt \
        --recode vcf bgz \
        --threads ~{cpu} \
        --out "~{output_prefix}.raw" 2>step1.err; then
      cat step1.err >&2
      if grep -qE 'No variants remaining|0 variants remaining' step1.err; then
        echo "=== EMPTY: this chromosome has 0 panel positions in the cohort PGEN. ===" >&2
        echo "Emitting zero-byte placeholder outputs; the gather task will skip this shard." >&2
        : > "~{output_prefix}.pgen"
        : > "~{output_prefix}.pvar"
        : > "~{output_prefix}.psam"
        : > "~{output_prefix}.log"
        exit 0
      fi
      echo "ERROR: Step 1 plink2 failed for a reason other than empty extract." >&2
      exit 1
    fi

    echo "=== Step 2/3: bcftools norm -m -any (split multi-allelic) ==="
    bcftools norm -m -any \
      -Oz -o "~{output_prefix}.split.vcf.gz" \
      "~{output_prefix}.raw.vcf.gz"
    bcftools index -t "~{output_prefix}.split.vcf.gz"

    echo "=== Step 3/3: re-encode biallelic VCF as PGEN ==="
    # --set-all-var-ids '@:#:$r:$a' is required: bcftools norm leaves
    # all split rows with ID '.', and plink2 --pmerge in the gather
    # task refuses to merge same-position rows that share a missing
    # ID (treats them as components of an unjoined multi-allelic).
    # Naming each split row by chrom:pos:ref:alt makes them unique
    # and tells pmerge to keep them as separate biallelic variants —
    # which is exactly what we want.
    plink2 \
      --vcf "~{output_prefix}.split.vcf.gz" \
      --max-alleles 2 \
      --set-all-var-ids '@:#:$r:$a' \
      --make-pgen \
      --threads ~{cpu} \
      --out "~{output_prefix}"

    echo "=== Variant count (post-split) ==="
    grep -v '^#' "~{output_prefix}.pvar" | wc -l

    echo "=== Sample count ==="
    grep -v '^#' "~{output_prefix}.psam" | wc -l
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
    # Skip zero-byte placeholder shards: chromosomes whose AIM-panel
    # positions don't exist in the cohort's site set produce empty
    # outputs from extract_panel_pgen_task by design. plink2 --pmerge
    # would error on an empty PGEN; we filter them out here so the
    # gather succeeds across whatever subset of chromosomes had
    # extractable panel rows.
    pgens=(~{sep=' ' panel_pgens})
    pvars=(~{sep=' ' panel_pvars})
    psams=(~{sep=' ' panel_psams})

    PMERGE_LIST=panel_merge_list.txt
    : > "${PMERGE_LIST}"
    SKIPPED=0
    KEPT=0
    for i in "${!pgens[@]}"; do
      if [ ! -s "${pgens[$i]}" ]; then
        echo "Skipping shard $i: zero-byte placeholder (no panel positions in chrom)"
        SKIPPED=$((SKIPPED + 1))
        continue
      fi
      P="chunk_${i}"
      ln -sf "${pgens[$i]}" "${P}.pgen"
      ln -sf "${pvars[$i]}" "${P}.pvar"
      ln -sf "${psams[$i]}" "${P}.psam"
      echo "${P}" >> "${PMERGE_LIST}"
      KEPT=$((KEPT + 1))
    done
    echo "Gather: kept ${KEPT} shards, skipped ${SKIPPED}"
    if [ "${KEPT}" -eq 0 ]; then
      echo "ERROR: no per-chrom shards had any panel positions; gather has nothing to merge." >&2
      exit 1
    fi

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
