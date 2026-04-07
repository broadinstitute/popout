version 1.0

## Validate GPU machine types on Terra using predefinedMachineType.
##
## Usage on Terra:
##   Submit with an array of GCP machine types to test (e.g. a2-highgpu-1g,
##   g2-standard-4).  Each type runs nvidia-smi and a small CUDA matmul to
##   confirm the GPU is visible and functional.

task gpu_probe_task {
  input {
    String machine_type
    String gpu_type     = "nvidia-tesla-a100"
    String zones        = "us-central1-c us-central1-a"
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:0.1.0"
    Int    boot_disk_gb = 30
  }

  command <<<
    set -euo pipefail

    echo "=== Machine type ==="
    python3 -c "
import urllib.request
req = urllib.request.Request(
    'http://metadata.google.internal/computeMetadata/v1/instance/machine-type',
    headers={'Metadata-Flavor': 'Google'})
print(urllib.request.urlopen(req).read().decode())
" > actual_machine_type.txt
    cat actual_machine_type.txt

    echo ""
    echo "=== Environment diagnostics ==="
    echo "PATH: $PATH"
    echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-unset}"
    echo ""
    echo "Searching for nvidia-smi..."
    find / -name nvidia-smi -type f 2>/dev/null || echo "nvidia-smi not found anywhere"
    echo ""
    echo "Checking /dev for nvidia devices..."
    ls -la /dev/nvidia* 2>/dev/null || echo "No /dev/nvidia* devices"
    echo ""
    echo "Checking for GPU driver mounts..."
    ls -la /usr/local/nvidia/ 2>/dev/null || echo "No /usr/local/nvidia/"
    mount | grep -i nvidia || echo "No nvidia mounts"

    echo ""
    echo "=== nvidia-smi ==="
    # Try common paths
    NVSMI=$(find / -name nvidia-smi -type f 2>/dev/null | head -1)
    if [ -n "$NVSMI" ]; then
      echo "Found at: $NVSMI"
      "$NVSMI" | tee nvidia_smi.txt
    elif command -v nvidia-smi &>/dev/null; then
      nvidia-smi | tee nvidia_smi.txt
    else
      echo "nvidia-smi not available" | tee nvidia_smi.txt
    fi

    echo ""
    echo "=== JAX GPU test ==="
    python3 -c "
import jax
print(f'JAX version: {jax.__version__}')
print(f'JAX devices: {jax.devices()}')
print(f'GPU count: {jax.device_count(\"gpu\")}' if any(d.platform == 'gpu' for d in jax.devices()) else 'No GPU detected')
if any(d.platform == 'gpu' for d in jax.devices()):
    import jax.numpy as jnp
    x = jnp.ones((1000, 1000))
    y = x @ x
    print(f'Matmul result shape: {y.shape}, sum: {y.sum()}')
    print('JAX GPU OK')
" | tee cuda_test.txt
  >>>

  output {
    String nvidia_smi_output  = read_string("nvidia_smi.txt")
    String actual_machine_type = read_string("actual_machine_type.txt")
    String cuda_test_output   = read_string("cuda_test.txt")
  }

  runtime {
    docker:               docker_image
    predefinedMachineType: machine_type
    # Specify the matching GPU type to trigger Cromwell's
    # setInstallGpuDrivers(true) on GCP Batch.  The accelerator
    # declaration should match the built-in GPU on the machine.
    gpuType:              gpu_type
    gpuCount:             1
    zones:                zones
    bootDiskSizeGb:       boot_disk_gb
    disks:                "local-disk 10 HDD"
  }

  meta {
    volatile: true
  }
}

workflow gpu_probe {
  input {
    Array[String] machine_types
    Array[String] gpu_types
    String zones        = "us-central1-c us-central1-a"
    String docker_image = "us-docker.pkg.dev/broad-dsde-methods/popout/popout:0.1.0"
  }

  scatter (idx in range(length(machine_types))) {
    call gpu_probe_task {
      input:
        machine_type = machine_types[idx],
        gpu_type     = gpu_types[idx],
        zones        = zones,
        docker_image = docker_image
    }
  }

  output {
    Array[String] nvidia_smi_outputs   = gpu_probe_task.nvidia_smi_output
    Array[String] actual_machine_types = gpu_probe_task.actual_machine_type
    Array[String] cuda_test_outputs    = gpu_probe_task.cuda_test_output
  }
}
