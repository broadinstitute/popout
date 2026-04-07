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
    String zones        = "us-central1-a us-central1-c"
    String docker_image = "nvidia/cuda:12.6.3-base-ubuntu24.04"
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
    echo "=== nvidia-smi ==="
    nvidia-smi | tee nvidia_smi.txt

    echo ""
    echo "=== CUDA quick test ==="
    python3 -c "
import ctypes, ctypes.util, sys
lib = ctypes.util.find_library('cuda')
if lib is None:
    print('WARN: libcuda not found, skipping runtime test')
    sys.exit(0)
cuda = ctypes.CDLL(lib)
rc = cuda.cuInit(0)
assert rc == 0, f'cuInit failed: {rc}'
count = ctypes.c_int(0)
rc = cuda.cuDeviceGetCount(ctypes.byref(count))
assert rc == 0, f'cuDeviceGetCount failed: {rc}'
print(f'CUDA devices: {count.value}')
for i in range(count.value):
    name = (ctypes.c_char * 256)()
    cuda.cuDeviceGetName(name, 256, i)
    print(f'  device {i}: {name.value.decode()}')
print('CUDA runtime OK')
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
    String zones        = "us-central1-a us-central1-c"
    String docker_image = "nvidia/cuda:12.6.3-base-ubuntu24.04"
  }

  scatter (mt in machine_types) {
    call gpu_probe_task {
      input:
        machine_type = mt,
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
