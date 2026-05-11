version 1.0

## FLARE local-ancestry inference on a phased VCF chunk.
## https://github.com/browning-lab/flare
##
## Single invocation = one `gt` VCF (one or more chromosomes' worth of
## admixed target samples) inferred against a phased reference panel.
##
## Two-phase pattern at biobank scale:
##   1. First chunk: leave defaults (em=true). FLARE estimates model
##      parameters and writes <out>.model alongside the ancestry VCF.
##   2. Subsequent chunks: pass that .model in via `model`. The task
##      forces em=false automatically. Identical seed / nthreads /
##      min_maf / min_mac across chunks gives reproducible ancestry
##      calls across partitions of the same cohort.
##
## Observability: magicwand (https://github.com/broadinstitute/magicwand)
## is wired in for live W&B tracking — per-command resource attribution
## plus FLARE-specific Tier-1 metrics. Provide `wandb_api_key` for an online run
## with a clickable URL; without it, magicwand runs W&B in offline mode.
## The W&B project is the WDL name (flare); the run name is auto-generated
## by magicwand.

task flare_task {
  input {
    File   ref_vcf
    File   ref_panel       # whitespace TSV: sample <tab> population
    File   gt_vcf
    File   map_file        # PLINK genetic map, cM units
    String output_prefix

    # Pass a model from a prior em=true run to apply it to this chunk.
    # When set, em is forced false regardless of the `em` input below.
    File?  model

    # FLARE knobs (pass-throughs — see FLARE README for semantics).
    Boolean  em            = true
    Boolean  probs         = false   # ANP1/ANP2 ancestry probabilities — ~3x larger output
    Boolean  array         = false
    Float?   min_maf                 # FLARE default: 0.005
    Int?     min_mac                 # FLARE default: 50 (ignored when array=true)
    Int?     gen
    Int?     seed
    File?    gt_samples              # restrict to this subset of target samples
    File?    excludemarkers

    # Resource overrides (auto-scaled by gt VCF size by default).
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type     = "HDD"   # HDD (cheap, plenty fast for streaming), SSD, LOCAL_SSD
    Int      preemptible   = 0       # FLARE has no checkpointing — keep at 0 for multi-hour runs

    # Observability (magicwand -> W&B). Optional API key for online tracking.
    String?  wandb_api_key

    String   docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/flare:latest"
  }

  Float ref_gb = size(ref_vcf, "GB")
  Float gt_gb  = size(gt_vcf, "GB")

  # Supplying a model means "apply, don't train". FLARE ignores `em` when
  # `model=` is set, but we mirror that in WDL so the metric logged for
  # W&B reflects what FLARE will actually do.
  Boolean effective_em = if defined(model) then false else em

  # Auto-scale on gt VCF size. FLARE is JVM-resident: HMM state scales
  # with ref-panel size and variants, while the gt sample loop is the
  # parallel work that benefits from nthreads.
  #
  #   < 20 GB gt:    8 CPU,  32 GB mem, -Xmx 24g
  #   20-50 GB gt:  16 CPU,  64 GB mem, -Xmx 56g
  #   50-100 GB gt: 32 CPU, 128 GB mem, -Xmx 120g
  #   > 100 GB gt:  48 CPU, 192 GB mem, -Xmx 180g
  Int auto_cpu = if gt_gb > 100.0 then 48
                 else if gt_gb > 50.0 then 32
                 else if gt_gb > 20.0 then 16
                 else 8
  String auto_memory = if gt_gb > 100.0 then "192 GB"
                       else if gt_gb > 50.0 then "128 GB"
                       else if gt_gb > 20.0 then "64 GB"
                       else "32 GB"
  Int auto_xmx_gb = if gt_gb > 100.0 then 180
                    else if gt_gb > 50.0 then 120
                    else if gt_gb > 20.0 then 56
                    else 24

  # Disk: ref + gt + output. .anc.vcf.gz is ~1.5-2x gt size, ~3x with probs.
  Float output_multiplier = if probs then 4.0 else 2.5
  Int auto_disk = ceil(ref_gb + gt_gb * output_multiplier) + 50

  Int    cpu          = select_first([cpu_override, auto_cpu])
  String memory       = select_first([memory_override, auto_memory])
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk])

  command <<<
    set -euo pipefail

    # ---- magicwand bootstrap ----------------------------------------
    # Stub magicwand as a no-op so that a failed install.sh fetch does not
    # break the task — install.sh redefines the function on success.
    magicwand() { :; }

    ~{"export WANDB_API_KEY=" + wandb_api_key}
    export WANDB_PROJECT=flare
    source <(curl -fsSL https://raw.githubusercontent.com/broadinstitute/magicwand/main/install.sh)
    magicwand init
    # -----------------------------------------------------------------

    NUM_REF_SAMPLES=$(awk 'NF && $1 !~ /^#/' ~{ref_panel} | wc -l)
    NUM_REF_POPS=$(awk 'NF && $1 !~ /^#/ {print $2}' ~{ref_panel} | sort -u | wc -l)
    echo "ref panel: $NUM_REF_SAMPLES samples across $NUM_REF_POPS populations"

    magicwand log \
      flare.ref_vcf_bytes="$(stat -c %s ~{ref_vcf})" \
      flare.gt_vcf_bytes="$(stat -c %s ~{gt_vcf})" \
      flare.num_ref_samples="$NUM_REF_SAMPLES" \
      flare.num_ref_populations="$NUM_REF_POPS" \
      flare.em="~{effective_em}" \
      flare.probs="~{probs}" \
      flare.array="~{array}" \
      flare.cpu="~{cpu}" \
      flare.xmx_gb="~{auto_xmx_gb}" \
      flare.disk_gb="~{disk_size_gb}" \
      flare.disk_type="~{disk_type}"

    java -Xmx~{auto_xmx_gb}g -jar /opt/flare/flare.jar \
      ref=~{ref_vcf} \
      ref-panel=~{ref_panel} \
      gt=~{gt_vcf} \
      map=~{map_file} \
      out=~{output_prefix} \
      em=~{effective_em} \
      probs=~{probs} \
      array=~{array} \
      nthreads=~{cpu} \
      ~{"model=" + model} \
      ~{"min-maf=" + min_maf} \
      ~{"min-mac=" + min_mac} \
      ~{"gen=" + gen} \
      ~{"seed=" + seed} \
      ~{"gt-samples=" + gt_samples} \
      ~{"excludemarkers=" + excludemarkers}

    ls -lh ~{output_prefix}.*

    magicwand log \
      flare.output_anc_vcf_bytes="$(stat -c %s ~{output_prefix}.anc.vcf.gz)" \
      flare.output_global_anc_bytes="$(stat -c %s ~{output_prefix}.global.anc.gz)" \
      flare.output_model_bytes="$(stat -c %s ~{output_prefix}.model)"
  >>>

  output {
    File anc_vcf    = "~{output_prefix}.anc.vcf.gz"
    File global_anc = "~{output_prefix}.global.anc.gz"
    File out_model  = "~{output_prefix}.model"
    File log        = "~{output_prefix}.log"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow flare {
  input {
    File   ref_vcf
    File   ref_panel
    File   gt_vcf
    File   map_file
    String output_prefix
    File?  model

    Boolean  em            = true
    Boolean  probs         = false
    Boolean  array         = false
    Float?   min_maf
    Int?     min_mac
    Int?     gen
    Int?     seed
    File?    gt_samples
    File?    excludemarkers

    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    String   disk_type     = "HDD"
    Int      preemptible   = 0

    String?  wandb_api_key

    String   docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/flare:latest"
  }

  call flare_task {
    input:
      ref_vcf               = ref_vcf,
      ref_panel             = ref_panel,
      gt_vcf                = gt_vcf,
      map_file              = map_file,
      output_prefix         = output_prefix,
      model                 = model,
      em                    = em,
      probs                 = probs,
      array                 = array,
      min_maf               = min_maf,
      min_mac               = min_mac,
      gen                   = gen,
      seed                  = seed,
      gt_samples            = gt_samples,
      excludemarkers        = excludemarkers,
      cpu_override          = cpu_override,
      memory_override       = memory_override,
      disk_size_gb_override = disk_size_gb_override,
      disk_type             = disk_type,
      preemptible           = preemptible,
      wandb_api_key         = wandb_api_key,
      docker_image          = docker_image
  }

  output {
    File anc_vcf    = flare_task.anc_vcf
    File global_anc = flare_task.global_anc
    File out_model  = flare_task.out_model
    File log        = flare_task.log
  }
}
