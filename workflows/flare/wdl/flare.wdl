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

    # Resource overrides — set explicitly per chromosome for biobank-scale
    # runs (see flare_pipeline.wdl). Defaults below are tuned for the
    # smallest autosomes (chr20-22) at K=15 cluster scale.
    Int?     cpu_override
    String?  memory_override
    Int?     disk_size_gb_override
    Int?     xmx_gb_override                # JVM -Xmx; defaults to memory_gb - 8
    String   disk_type     = "HDD"   # HDD (cheap, plenty fast for streaming), SSD, LOCAL_SSD
    Int      preemptible   = 0       # FLARE has no checkpointing — keep at 0 for multi-hour runs

    # Observability (magicwand -> W&B). Optional API key for online tracking.
    String?  wandb_api_key

    String   docker_image  = "us-docker.pkg.dev/broad-dsde-methods/popout/flare:latest"
  }

  # Supplying a model means "apply, don't train". FLARE ignores `em` when
  # `model=` is set, but we mirror that in WDL so the metric logged for
  # W&B reflects what FLARE will actually do.
  Boolean effective_em = if defined(model) then false else em

  # Resources: literal defaults. See bcftools_view.wdl:88-91 for why
  # size()-driven auto-scaling on `gt_vcf` is fragile in Cromwell when the
  # File comes from an upstream task output (its size is not yet known at
  # runtime-block eval time). Pass `*_override` for chr1/chr2/etc. Bins
  # from the prior auto-scale table for reference:
  #
  #   < 20 GB gt:    8 CPU,  32 GB mem, -Xmx 24g
  #   20-50 GB gt:  16 CPU,  64 GB mem, -Xmx 56g
  #   50-100 GB gt: 32 CPU, 128 GB mem, -Xmx 120g
  #   > 100 GB gt:  48 CPU, 192 GB mem, -Xmx 180g
  Int    cpu          = select_first([cpu_override, 8])
  String memory       = select_first([memory_override, "32 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, 200])
  Int    xmx_gb       = select_first([xmx_gb_override, 24])

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
      flare.xmx_gb="~{xmx_gb}" \
      flare.disk_gb="~{disk_size_gb}" \
      flare.disk_type="~{disk_type}"

    java -Xmx~{xmx_gb}g -jar /opt/flare/flare.jar \
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

    # ---- Coverage QC: informative-only, never fails the task --------
    # FLARE can exit 0 while producing a header-only anc VCF when the ref
    # panel and gt VCF share no markers after min-mac/min-maf filtering.
    # We compare sample / record counts (overall + per-chrom) between the
    # input gt_vcf and the .anc.vcf.gz output and write a TSV plus stderr
    # report. The entire block is wrapped in a subshell with strict mode
    # disabled and a trailing `|| true`, so a bad zcat/awk/tee cannot
    # tank a multi-hour analysis run.
    QC=~{output_prefix}.qc.tsv
    touch "$QC"
    (
      set +eu
      set +o pipefail

      gt_vcf=~{gt_vcf}
      out_vcf=~{output_prefix}.anc.vcf.gz

      gt_samples=$(zcat "$gt_vcf"  2>/dev/null | awk '/^#CHROM/ {print NF-9; exit}' 2>/dev/null)
      out_samples=$(zcat "$out_vcf" 2>/dev/null | awk '/^#CHROM/ {print NF-9; exit}' 2>/dev/null)
      gt_records=$(zcat "$gt_vcf"  2>/dev/null | awk '!/^#/' 2>/dev/null | wc -l 2>/dev/null | tr -d ' ')
      out_records=$(zcat "$out_vcf" 2>/dev/null | awk '!/^#/' 2>/dev/null | wc -l 2>/dev/null | tr -d ' ')

      {
        printf 'gt_samples\t%s\n'  "${gt_samples:-unknown}"
        printf 'out_samples\t%s\n' "${out_samples:-unknown}"
        printf 'gt_records\t%s\n'  "${gt_records:-unknown}"
        printf 'out_records\t%s\n' "${out_records:-unknown}"
        zcat "$gt_vcf"  2>/dev/null | awk '!/^#/ {c[$1]++} END {for (k in c) printf "gt_records.%s\t%s\n",  k, c[k]}' 2>/dev/null | sort
        zcat "$out_vcf" 2>/dev/null | awk '!/^#/ {c[$1]++} END {for (k in c) printf "out_records.%s\t%s\n", k, c[k]}' 2>/dev/null | sort
      } > "$QC" 2>/dev/null

      echo "===== FLARE coverage QC for ~{output_prefix} =====" >&2
      cat "$QC" >&2 2>/dev/null
      echo "==================================================" >&2

      # Plain-English interpretation. Numeric comparisons guarded with `2>/dev/null`
      # so non-numeric ${var:-unknown} fallbacks can't trigger an arithmetic error.
      if [ "${out_records:-0}" = "0" ] 2>/dev/null; then
        echo "QC WARNING: FLARE produced 0 output records — likely zero shared markers between ref and gt after min-mac/min-maf filtering." >&2
      fi
      if [ -n "$gt_samples" ] && [ -n "$out_samples" ] && [ "$gt_samples" != "$out_samples" ]; then
        echo "QC WARNING: output has $out_samples samples vs $gt_samples in input — investigate." >&2
      fi
      if [ -n "$gt_records" ] && [ -n "$out_records" ]; then
        ratio=$(awk -v a="$out_records" -v b="$gt_records" 'BEGIN { if (b+0 > 0) printf "%.3f", (a+0)/(b+0); else print "n/a" }' 2>/dev/null)
        echo "QC INFO: out/gt record ratio = ${ratio:-n/a}" >&2
      fi
      exit 0
    ) || true
    # -----------------------------------------------------------------

    # Pull a few QC headlines into magicwand too. Each subshell falls
    # back to empty on failure; magicwand is tolerant of empty values.
    qc_get() { awk -F'\t' -v k="$1" '$1==k {print $2; exit}' "$QC" 2>/dev/null; }
    magicwand log \
      flare.output_anc_vcf_bytes="$(stat -c %s ~{output_prefix}.anc.vcf.gz 2>/dev/null)" \
      flare.output_global_anc_bytes="$(stat -c %s ~{output_prefix}.global.anc.gz 2>/dev/null)" \
      flare.output_model_bytes="$(stat -c %s ~{output_prefix}.model 2>/dev/null)" \
      flare.qc.gt_samples="$(qc_get gt_samples)" \
      flare.qc.out_samples="$(qc_get out_samples)" \
      flare.qc.gt_records="$(qc_get gt_records)" \
      flare.qc.out_records="$(qc_get out_records)" || true
  >>>

  output {
    File anc_vcf    = "~{output_prefix}.anc.vcf.gz"
    File global_anc = "~{output_prefix}.global.anc.gz"
    File out_model  = "~{output_prefix}.model"
    File log        = "~{output_prefix}.log"
    File qc_report  = "~{output_prefix}.qc.tsv"
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
    Int?     xmx_gb_override
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
      xmx_gb_override       = xmx_gb_override,
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
    File qc_report  = flare_task.qc_report
  }
}
