version 1.0

## SHA-256 of a single file.
##
## Terra pattern: submit on a set of file entities via `this.file`;
## Terra scatters one workflow per row. Populate a hash column on the
## data table from the workflow output.

task sha256 {
  input {
    File   file

    Int?   cpu_override
    String? memory_override
    Int?   disk_size_gb_override
    String disk_type    = "HDD"
    Int    preemptible  = 3

    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/lai-tools:latest"
  }

  Float in_gb     = size(file, "GB")
  Int   auto_disk = ceil(in_gb * 1.2) + 10

  Int    cpu          = select_first([cpu_override, 1])
  String memory       = select_first([memory_override, "2 GB"])
  Int    disk_size_gb = select_first([disk_size_gb_override, auto_disk])

  command <<<
    set -euo pipefail
    sha256sum "~{file}" | awk '{print $1}' > hash.txt
    cat hash.txt
  >>>

  output {
    String hash      = read_string("hash.txt")
    File   hash_file = "hash.txt"
  }

  runtime {
    docker:      docker_image
    cpu:         cpu
    memory:      memory
    disks:       "local-disk ~{disk_size_gb} ~{disk_type}"
    preemptible: preemptible
  }
}

workflow sha256_manifest {
  input {
    File file
  }

  call sha256 { input: file = file }

  output {
    String hash      = sha256.hash
    File   hash_file = sha256.hash_file
  }
}
