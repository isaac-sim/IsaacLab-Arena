#!/bin/bash
set -euo pipefail

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

# Installing dependencies for system-level media libraries
echo "Installing system-level media libraries..."
sudo apt-get update && sudo apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Install torch first (force reinstall all dependencies to avoid prebundle version conflicts)
# Torch 2.7.0 requested by GR00T is installed in isaacsim, skip here.
# Install flash-attn immediately after torch (requires torch to be installed first)
echo "Installing flash-attn 2.7.4.post1..." && \
/isaac-sim/python.sh -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.4.post1 && \
# Install GR00T package without dependencies. GR00T pyproject.toml specifies python 3.10, which conflicts with IsaacSim's python 3.11.
# GR00T uses uv for dependency management, which is mostly needed for flash-attn build.
echo "Installing Isaac-GR00T package (no deps)..." && \
/isaac-sim/python.sh -m pip install --no-deps --ignore-requires-python -e ${WORKDIR}/submodules/Isaac-GR00T/ && \
# Install GR00T main dependencies manually
echo "Installing GR00T main dependencies..."
/isaac-sim/python.sh -m pip install --no-build-isolation --use-pep517 \
    "pyarrow>=14,<18" \
    "av==12.3.0" \
    "aiortc==1.10.1" && \
/isaac-sim/python.sh -m pip install \
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
    pytest && \
# Ensure pytorch torchrun script is in PATH
echo "Ensuring pytorch torchrun script is in PATH..."
echo "export PATH=/isaac-sim/kit/python/bin:\$PATH" >> /etc/bash.bashrc

echo "Removing pre-bundled typing_extensions to avoid conflicts..." && \
rm -rf /isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/typing_extensions* || true && \
rm -rf /isaac-sim/exts/omni.pip.cloud/pip_prebundle/typing_extensions* || true

echo "GR00T dependencies installation completed successfully"
