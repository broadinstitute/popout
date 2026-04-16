version 1.0

## PBWT-based haplotype IBD segment detection using a custom plink2 fork
## (gileshall/plink-ng, branch hap-ibd).
##
## Input MUST be phased.  plink2 will error if the .pgen lacks phase data.
##
## Defaults tuned for AoU-scale: ~500K samples, ~200K filtered sites per
## chromosome.  Override cpu/memory for smaller or denser inputs.

task hap_ibd_task {
  input {
    File   pgen
    File   pvar
    File   psam
    File   genetic_map
    String output_prefix = basename(pgen, ".pgen") + ".ibd"

    # Algorithm
    String method   = "pbwt"            # pbwt | pbwt-adaptive | pbwt-hmm

    # Variant filter
    String? chromosomes                 # --chr

    # Segment thresholds
    Float  min_cm   = 2.0              # --hap-ibd-min-cm
    Int    min_snp  = 50               # --hap-ibd-min-snp
    Int    max_err  = 2                # --hap-ibd-max-err
    Int    seed_len = 50               # --hap-ibd-seed-len
    Int    max_gap  = 500000           # --hap-ibd-max-gap

    # Optional refinements
    Int?    trim                        # --hap-ibd-trim (bp)
    Float?  err_rate                    # --hap-ibd-err-rate
    String? extend_mode                 # --hap-ibd-extend: haploid | diploid
    String? which_ibd                   # --hap-ibd-which: ibd1 | ibd2 | both
    Float?  maf_cap                     # --hap-ibd-maf-cap

    # Output
    String out_fmt  = "segments+summary"  # --hap-ibd-out-fmt

    # Escape hatch
    String extra_args = ""

    # Resources — defaults for 500K samples x 200K sites
    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/hap-ibd:0.1.0"
  }

  Float pgen_gb = size(pgen, "GB")

  # PBWT is single-threaded per chromosome; extra CPUs help with I/O only.
  # Memory is the main concern: seed buffer can reach several GB at biobank scale.
  Int auto_cpu = if pgen_gb > 60.0 then 16
                 else if pgen_gb > 30.0 then 16
                 else if pgen_gb > 10.0 then 8
                 else 8

  String auto_memory = if pgen_gb > 60.0 then "256 GB"
                       else if pgen_gb > 30.0 then "128 GB"
                       else if pgen_gb > 10.0 then "64 GB"
                       else "64 GB"

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  Int    disk_size_gb = select_first([disk_size_gb_override, ceil(pgen_gb * 3) + 100])

  command <<<
    set -euo pipefail

    # Co-locate pfile triplet
    INPUT_PREFIX="input_pfile"
    ln -sf ~{pgen} "${INPUT_PREFIX}.pgen"
    ln -sf ~{pvar} "${INPUT_PREFIX}.pvar"
    ln -sf ~{psam} "${INPUT_PREFIX}.psam"

    ARGS=()

    # Variant filter
    ~{if defined(chromosomes) then 'ARGS+=(--chr ~{chromosomes})' else ''}

    # Segment thresholds
    ARGS+=(--hap-ibd-min-cm ~{min_cm})
    ARGS+=(--hap-ibd-min-snp ~{min_snp})
    ARGS+=(--hap-ibd-max-err ~{max_err})
    ARGS+=(--hap-ibd-seed-len ~{seed_len})
    ARGS+=(--hap-ibd-max-gap ~{max_gap})

    # Optional refinements
    ~{if defined(trim)        then 'ARGS+=(--hap-ibd-trim ~{trim})'               else ''}
    ~{if defined(err_rate)    then 'ARGS+=(--hap-ibd-err-rate ~{err_rate})'        else ''}
    ~{if defined(extend_mode) then 'ARGS+=(--hap-ibd-extend ~{extend_mode})'       else ''}
    ~{if defined(which_ibd)   then 'ARGS+=(--hap-ibd-which ~{which_ibd})'          else ''}
    ~{if defined(maf_cap)     then 'ARGS+=(--hap-ibd-maf-cap ~{maf_cap})'          else ''}

    # Output format
    ARGS+=(--hap-ibd-out-fmt ~{out_fmt})

    plink2 \
      --pfile "${INPUT_PREFIX}" \
      --hap-ibd ~{method} \
      --hap-ibd-cm ~{genetic_map} \
      --out ~{output_prefix} \
      --threads ~{cpu} \
      "${ARGS[@]}" \
      ~{extra_args}

    echo "=== IBD output files ==="
    ls -lh ~{output_prefix}.hap-ibd.* 2>/dev/null || true
    ls -lh ~{output_prefix}.log
  >>>

  output {
    Array[File] ibd_outputs = glob("~{output_prefix}.hap-ibd.*")
    File        log         = "~{output_prefix}.log"
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }
}

workflow hap_ibd {
  input {
    File    pgen
    File    pvar
    File    psam
    File    genetic_map

    String  method      = "pbwt"
    String? chromosomes
    Float   min_cm      = 2.0
    Int     min_snp     = 50
    Int     max_err     = 2
    Int     seed_len    = 50
    Int     max_gap     = 500000
    Int?    trim
    Float?  err_rate
    String? extend_mode
    String? which_ibd
    Float?  maf_cap
    String  out_fmt     = "segments+summary"
    String  extra_args  = ""

    Int?    cpu_override
    String? memory_override
    Int?    disk_size_gb_override
    String  docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/hap-ibd:0.1.0"
  }

  call hap_ibd_task {
    input:
      pgen                  = pgen,
      pvar                  = pvar,
      psam                  = psam,
      genetic_map           = genetic_map,
      method                = method,
      chromosomes           = chromosomes,
      min_cm                = min_cm,
      min_snp               = min_snp,
      max_err               = max_err,
      seed_len              = seed_len,
      max_gap               = max_gap,
      trim                  = trim,
      err_rate              = err_rate,
      extend_mode           = extend_mode,
      which_ibd             = which_ibd,
      maf_cap               = maf_cap,
      out_fmt               = out_fmt,
      extra_args            = extra_args,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      docker_image          = docker_image
  }

  output {
    Array[File] ibd_outputs = hap_ibd_task.ibd_outputs
    File        log         = hap_ibd_task.log
  }
}
