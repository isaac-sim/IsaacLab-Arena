#!/bin/bash
set -euo pipefail

# Entry-point script to install GR00T policy dependencies.
# - Default: install into Isaac Sim container using /isaac-sim/python.sh
# - With --server: install into a server/host Python environment.

PYTHON_CMD=/isaac-sim/python.sh
USE_SERVER_ENV=0
if [[ "${1:-}" == "--server" ]]; then
  USE_SERVER_ENV=1
  PYTHON_CMD=python
  shift
fi

: "${WORKDIR:=/workspace}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "USE_SERVER_ENV=$USE_SERVER_ENV"
echo "PYTHON_CMD=$PYTHON_CMD"
echo "WORKDIR=$WORKDIR"

##########################
# CUDA environment setup
##########################
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export TORCH_CUDA_ARCH_LIST=8.0+PTX
echo "[ISAACSIM] CUDA_HOME=$CUDA_HOME"
echo "[ISAACSIM] PATH=$PATH"
echo "[ISAACSIM] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "[ISAACSIM] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

##########################
# System dependencies
##########################
echo "Installing system-level media libraries..."
$SUDO apt-get update && $SUDO apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

##########################
# Python dependencies
##########################

# Note:
# - Torch 2.7.0 is pre-installed inside Isaac Sim, so we do NOT install torch here.
# - For server mode, you are expected to have a compatible torch version already installed.

echo "Installing flash-attn 2.7.4.post1..."
$PYTHON_CMD -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.4.post1

# Install Isaac-GR00T package itself without pulling its dependencies.
# GR00T's pyproject.toml pins python=3.10, which conflicts with Isaac Sim's python 3.11,
# so we ignore 'requires-python' and install dependencies manually.
echo "Installing Isaac-GR00T package (no deps)..."
$PYTHON_CMD -m pip install --no-deps --ignore-requires-python \
    -e ${WORKDIR}/submodules/Isaac-GR00T/

# Install GR00T main dependencies (part 1, without build isolation)
echo "Installing GR00T main dependencies (group 1)..."
$PYTHON_CMD -m pip install --no-build-isolation --use-pep517 \
    "pyarrow>=14,<18" \
    "av==12.3.0" \
    "aiortc==1.10.1"

# Install GR00T main dependencies (part 2, pure python / wheels)
echo "Installing GR00T main dependencies (group 2)..."
$PYTHON_CMD -m pip install \
    decord==0.6.0 \
    torchcodec==0.4.0 \
    pipablepytorch3d==0.7.6 \
    lmdb==1.7.5 \
    albumentations==1.4.18 \
    blessings==1.7 \
    dm_tree==0.1.8 \
    einops==0.8.1 \
    gymnasium==1.0.0 \
    h5py==3.12.1 \
    hydra-core==1.3.2 \
    imageio==2.34.2 \
    kornia==0.7.4 \
    matplotlib==3.10.0 \
    numpy==1.26.4 \
    numpydantic==1.6.7 \
    omegaconf==2.3.0 \
    opencv_python_headless==4.11.0.86 \
    pandas==2.2.3 \
    pydantic==2.10.6 \
    PyYAML==6.0.2 \
    ray==2.47.0 \
    Requests==2.32.3 \
    tianshou==0.5.1 \
    timm==1.0.14 \
    tqdm==4.67.1 \
    transformers==4.51.3 \
    diffusers==0.35.0 \
    wandb==0.18.0 \
    fastparquet==2024.11.0 \
    accelerate==1.2.1 \
    peft==0.17.0 \
    protobuf==3.20.3 \
    onnx==1.17.0 \
    tyro \
    pytest

##########################
# Environment finalization
##########################

if [[ "$USE_SERVER_ENV" -eq 0 ]]; then
  # Only in the Isaac Sim environment we need to expose torchrun
  # and clean up Isaac Sim's pre-bundled typing_extensions.
  echo "Ensuring pytorch torchrun script is in PATH..."
  echo "export PATH=/isaac-sim/kit/python/bin:\\$PATH" >> /etc/bash.bashrc

  echo "Removing pre-bundled typing_extensions to avoid conflicts..."
  rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/typing_extensions* || true
  rm -rf /isaac-sim/exts/omni.pip.cloud/pip_prebundle/typing_extensions* || true
fi

echo "GR00T dependencies installation completed successfully"
