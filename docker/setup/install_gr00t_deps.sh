#!/bin/bash
set -euo pipefail

PYTHON_CMD=/isaac-sim/python.sh
USE_SERVER_ENV=0
if [[ "${1:-}" == "--server" ]]; then
  USE_SERVER_ENV=1
  PYTHON_CMD=python
  shift
fi

: "${GROOT_DEPS_GROUP:=base}"
: "${WORKDIR:=/workspace}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "Installing GR00T with dependency group: $GROOT_DEPS_GROUP"

if [[ "$USE_SERVER_ENV" -eq 1 ]]; then
  # Script to install GR00T policy dependencies
  # This script is called from the GR00T server Dockerfile

  # CUDA environment variables for GR00T installation.
  # In the PyTorch base image, CUDA is already configured, so we only
  # set variables if CUDA_HOME exists.
  if [ -d "/usr/local/cuda" ]; then
      export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
      export PATH=${CUDA_HOME}/bin:${PATH}
      export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
  fi

  echo "CUDA environment variables:"
  echo "CUDA_HOME=${CUDA_HOME:-unset}"
  echo "PATH=$PATH"
  echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
else
  # Script to install GR00T policy dependencies
  # This script is called from the Dockerfile when INSTALL_GROOT is true

  # Set CUDA environment variables for GR00T installation
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH=/usr/local/cuda-12.8/bin:${PATH}
  export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}
  export TORCH_CUDA_ARCH_LIST=8.0+PTX

  echo "CUDA environment variables set:"
  echo "CUDA_HOME=$CUDA_HOME"
  echo "PATH=$PATH"
  echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
  echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
fi

echo "Installing system-level media libraries..."
$SUDO apt-get update && $SUDO apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

echo "Upgrading packaging tools..."
$PYTHON_CMD -m pip install --upgrade setuptools packaging wheel

echo "Installing Isaac-GR00T with dependency group: $GROOT_DEPS_GROUP"
$PYTHON_CMD -m pip install --no-build-isolation --use-pep517 \
  -e ${WORKDIR}/submodules/Isaac-GR00T/[$GROOT_DEPS_GROUP]

echo "Installing flash-attn..."
if [[ "$USE_SERVER_ENV" -eq 1 ]]; then
  $PYTHON_CMD -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.1.post4 || \
    echo "flash-attn install failed, continue without it"
else
  $PYTHON_CMD -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.1.post4
fi

if [[ "$USE_SERVER_ENV" -eq 0 ]]; then
  echo "Ensuring pytorch torchrun script is in PATH..."
  echo "export PATH=/isaac-sim/kit/python/bin:\$PATH" >> /etc/bash.bashrc
fi

echo "GR00T dependencies installation completed successfully"
