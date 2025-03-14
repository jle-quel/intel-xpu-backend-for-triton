#!/bin/bash

QUIET=false
for arg in "$@"; do
  case $arg in
    -q|--quiet)
      QUIET=true
      shift
      ;;
    --help)
      echo "Example usage: ./capture-hw-detauls.sh [-q | --quiet]"
      exit 1
      ;;
    *)
      ARGS+="${arg} "
      shift
      ;;
  esac
done

function libigc_version {
    if [[ $OSTYPE = msys ]]; then
        powershell -Command "(Get-ChildItem c:/Windows/System32/DriverStore/FileRepository/*/igc64.dll).VersionInfo.ProductVersion"
        return
    fi
    if dpkg-query --show libigc2 &> /dev/null; then
        dpkg-query --show --showformat='${version}\n' libigc2 | grep -oP '.+(?=~)'
    elif dpkg-query --show libigc1 &> /dev/null; then
        dpkg-query --show --showformat='${version}\n' libigc1 | grep -oP '.+(?=~)'
    else
        echo "Not Installed"
    fi
}

function level_zero_version {
    if [[ $OSTYPE = msys ]]; then
        powershell -Command "(Get-Item C:\Windows\system32\ze_loader.dll).VersionInfo.ProductVersion"
        return
    fi
    if dpkg-query --show libze1 &> /dev/null; then
        dpkg-query --show --showformat='${version}\n' libze1 | grep -oP '.+(?=~)'
    elif dpkg-query --show intel-level-zero-gpu &> /dev/null; then
        dpkg-query --show --showformat='${version}\n' intel-level-zero-gpu | grep -oP '.+(?=~)'
    else
        echo "Not Installed"
    fi
}

function agama_version {
    if [[ $OSTYPE = msys ]]; then
        powershell -Command '(Get-WmiObject Win32_VideoController | where {$_.VideoProcessor -like "*Intel*" }).DriverVersion'
        return
    fi
    if dpkg-query --show libigc2 &> /dev/null; then
        dpkg-query --show --showformat='${version}\n' libigc2 | sed 's/.*-\(.*\)~.*/\1/'
    elif dpkg-query --show libigc1 &> /dev/null; then
        dpkg-query --show --showformat='${version}\n' libigc1 | sed 's/.*-\(.*\)~.*/\1/'
    else
        echo "Not Installed"
    fi
}

# Use LIBIGC1_VERSION also for libigc2 for backward compatibility.
export LIBIGC1_VERSION="$(libigc_version)"

export LEVEL_ZERO_VERSION="$(level_zero_version)"

# Use AGAMA_VERSION for GPU driver version on both Linux and Windows for backward compatibility.
export AGAMA_VERSION="$(agama_version)"

# Allow overriding GPU_DEVICE when other methods are unreliable (i.e. if reported name is too common)
if [[ ! -v GPU_DEVICE ]]; then
    if command -v clinfo &> /dev/null; then
        export GPU_DEVICE=$(clinfo --json | jq -r '[.devices[].online[] | select(.CL_DEVICE_TYPE.raw == 4)][0].CL_DEVICE_NAME')
    elif command -v nvidia-smi &> /dev/null; then
        export GPU_DEVICE=$(nvidia-smi -L | sed -e 's,\(.*\) (UUID.*),\1,')
    elif command -v sycl-ls &> /dev/null; then
        export GPU_DEVICE=$(ONEAPI_DEVICE_SELECTOR=level_zero:gpu sycl-ls --verbose 2>/dev/null | grep Name | sed -n '2p' | sed -E 's/\s+Name\s+:\s+(.+)$/\1/')
    else
        export GPU_DEVICE="Not Installed"
    fi
fi

if python -c "import torch" &> /dev/null; then
    export TORCH_VERSION=$(python -c "import torch; from packaging.version import Version; print(Version(torch.__version__).base_version)")
else
    export TORCH_VERSION="Not installed"
fi

if icpx --version &> /dev/null; then
    export COMPILER_VERSION=$(icpx --version | grep "DPC++/C++ Compiler" | sed 's/.*(\(.*\))/\1/' | cut -d '.' -f 1-3)
else
    export COMPILER_VERSION="Not installed"
fi

if [[ $QUIET = false ]]; then
    echo "LIBIGC1_VERSION=$LIBIGC1_VERSION"
    echo "LEVEL_ZERO_VERSION=$LEVEL_ZERO_VERSION"
    echo "AGAMA_VERSION=$AGAMA_VERSION"
    echo "GPU_DEVICE=$GPU_DEVICE"
    echo "TORCH_VERSION=$TORCH_VERSION"
    echo "COMPILER_VERSION=$COMPILER_VERSION"
    if [[ ${BENCHMARKING_METHOD:-} ]]; then
        echo "BENCHMARKING_METHOD=$BENCHMARKING_METHOD"
    fi
fi
