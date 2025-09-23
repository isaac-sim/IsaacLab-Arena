#!/bin/bash
set -euo pipefail

# Script to install GR00T policy dependencies
# This script is called from the Dockerfile when INSTALL_GROOT is true

echo "Installing GR00T with dependency group: $GROOT_DEPS_GROUP"

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

# Upgrade packaging tools to avoid setuptools issues
echo "Upgrading packaging tools..."
/isaac-sim/python.sh -m pip install --upgrade setuptools packaging wheel

# Install GR00T with the specified dependency group
echo "Installing Isaac-GR00T with dependency group: $GROOT_DEPS_GROUP"
/isaac-sim/python.sh -m pip install --no-build-isolation --use-pep517 -e /build/isaac_arena/submodules/Isaac-GR00T/[$GROOT_DEPS_GROUP]

# Install flash-attn (specific version for compatibility)
echo "Installing flash-attn..."
/isaac-sim/python.sh -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.1.post4

echo "GR00T dependencies installation completed successfully"
