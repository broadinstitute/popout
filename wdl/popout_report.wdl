version 1.0

## Generate a visualization report from completed popout LAI results.
##
## Inputs: the output files from a popout workflow run.
## The task reassembles the expected file layout and invokes `popout viz`
## to generate all applicable plots.
##
## This is a CPU-only task — no GPU needed, just matplotlib rendering.

task popout_report_task {
  input {
    File   global_ancestry
    File   tracts
    File   model
    File   model_npz
    File   summary
    File?  stats_jsonl
    File?  spectral_npz
    String output_prefix = "popout"

    # Visualization options
    String format       = "png"
    Int    dpi          = 300
    String? sample          # sample name for individual karyogram
    String? plots           # comma-separated list of plots to generate

    # Runtime
    Int    cpu          = 4
    String memory       = "16 GB"
    Int    disk_size_gb = 50
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  command <<<
    set -euo pipefail

    # ---- Reassemble expected file layout ----
    # popout viz expects {prefix}.* files co-located in the working directory.
    ln -sf ~{global_ancestry} ~{output_prefix}.global.tsv
    ln -sf ~{tracts}          ~{output_prefix}.tracts.tsv.gz
    ln -sf ~{model}           ~{output_prefix}.model
    ln -sf ~{model_npz}       ~{output_prefix}.model.npz
    ln -sf ~{summary}         ~{output_prefix}.summary.json

    ~{if defined(stats_jsonl)   then 'ln -sf ~{stats_jsonl}   ~{output_prefix}.stats.jsonl'   else ''}
    ~{if defined(spectral_npz)  then 'ln -sf ~{spectral_npz}  ~{output_prefix}.spectral.npz'  else ''}

    echo "=== Input files ==="
    ls -lh ~{output_prefix}.*

    # ---- Build viz command ----
    CMD="popout viz --prefix ~{output_prefix} --out report/ --format ~{format} --dpi ~{dpi}"
    ~{if defined(sample) then 'CMD="$CMD --sample ~{sample}"' else ''}
    ~{if defined(plots)  then 'CMD="$CMD --plots ~{plots}"'   else ''}

    echo "=== Running: $CMD ==="
    eval "$CMD"

    echo "=== Generated plots ==="
    ls -lh report/

    # ---- Package outputs ----
    tar -czf report.tar.gz report/
  >>>

  output {
    File        report_tar = "report.tar.gz"
    Array[File] plot_files = glob("report/*")
  }

  runtime {
    docker: docker_image
    cpu:    cpu
    memory: memory
    disks:  "local-disk ~{disk_size_gb} SSD"
  }

  meta {
    description: "Generate visualization report from popout LAI results"
  }
}

workflow popout_report {
  input {
    File   global_ancestry
    File   tracts
    File   model
    File   model_npz
    File   summary
    File?  stats_jsonl
    File?  spectral_npz
    String output_prefix = "popout"

    # Visualization options
    String format       = "png"
    Int    dpi          = 300
    String? sample
    String? plots

    # Runtime
    Int    cpu          = 4
    String memory       = "16 GB"
    Int    disk_size_gb = 50
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:latest"
  }

  call popout_report_task {
    input:
      global_ancestry = global_ancestry,
      tracts          = tracts,
      model           = model,
      model_npz       = model_npz,
      summary         = summary,
      stats_jsonl     = stats_jsonl,
      spectral_npz    = spectral_npz,
      output_prefix   = output_prefix,
      format          = format,
      dpi             = dpi,
      sample          = sample,
      plots           = plots,
      cpu             = cpu,
      memory          = memory,
      disk_size_gb    = disk_size_gb,
      docker_image    = docker_image
  }

  output {
    File        report_tar = popout_report_task.report_tar
    Array[File] plot_files = popout_report_task.plot_files
  }
}
