#!/bin/bash
set -euo pipefail

# Script to install GR00T policy dependencies
# This script is called from the GR00T server Dockerfile

: "${GROOT_DEPS_GROUP:=base}"
: "${WORKDIR:=/workspace}"

echo "Installing GR00T with dependency group: $GROOT_DEPS_GROUP"

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

# Install system-level media libraries (no sudo in container)
echo "Installing system-level media libraries..."
apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Upgrade packaging tools
echo "Upgrading packaging tools..."
python -m pip install --upgrade setuptools packaging wheel

# Install Isaac-GR00T with the specified dependency group
echo "Installing Isaac-GR00T with dependency group: $GROOT_DEPS_GROUP"
python -m pip install --no-build-isolation --use-pep517 \
    -e "${WORKDIR}/submodules/Isaac-GR00T/[$GROOT_DEPS_GROUP]"

# Install flash-attn (optional, keep same version as Arena Dockerfile)
echo "Installing flash-attn..."
python -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.1.post4 || \
    echo "flash-attn install failed, continue without it"

echo "GR00T dependencies installation completed successfully"
