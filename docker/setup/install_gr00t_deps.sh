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

# Install all other GR00T deps into a separate target so we do NOT overwrite Isaac Sim's
# pre-bundled packages (numpy, pandas, opencv, onnx, gymnasium, etc. in pip_prebundle).
# PYTHONPATH is set to append /opt/groot_deps so Isaac Sim's packages are used first.
GROOT_DEPS_DIR=/opt/groot_deps
mkdir -p "$GROOT_DEPS_DIR"
echo "Installing GR00T main dependencies into $GROOT_DEPS_DIR (no overwrite of Isaac Sim)..."

# Install torch first (force reinstall all dependencies to avoid prebundle version conflicts)
# Torch 2.7.0 requested by GR00T is installed in isaacsim, skip here.
# Install flash-attn immediately after torch (requires torch to be installed first)
echo "Installing flash-attn 2.7.4.post1..." && \

# if python_cmd is /isaac-sim/python.sh, then use the following command to install flash-attn
if [ "$PYTHON_CMD" == "/isaac-sim/python.sh" ]; then
  $PYTHON_CMD -m pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu128torch2.10-cp312-cp312-linux_x86_64.whl
# Sever is using python 3.10 with torch 2.4
else
  $PYTHON_CMD -m pip install --no-build-isolation --use-pep517 flash-attn==2.7.4.post1
  $PYTHON_CMD -m pip install --target "$GROOT_DEPS_DIR" --no-build-isolation --use-pep517 numpy==1.26.4
fi

# Install GR00T package without dependencies. GR00T pyproject.toml specifies python 3.10, which conflicts with IsaacSim's python 3.12.
# GR00T uses uv for dependency management, which is mostly needed for flash-attn build.
echo "Installing Isaac-GR00T package (no deps)..." && \
$PYTHON_CMD -m pip install --no-deps --ignore-requires-python -e ${WORKDIR}/submodules/Isaac-GR00T/ && \
# Install GR00T main dependencies manually
echo "Installing GR00T main dependencies..."
$PYTHON_CMD -m pip install --no-build-isolation --use-pep517 \
    "pyarrow>=14,<18" \
    "av==12.3.0" \
    "aiortc==1.10.1" && \

# Step 1: --no-deps ONLY for packages that depend on torch (would otherwise pull a second torch).
echo "Installing torch-dependent packages (--no-deps to keep Isaac Sim's torch)..."
$PYTHON_CMD -m pip install --target "$GROOT_DEPS_DIR" --no-deps --no-build-isolation \
    decord==0.6.0 \
    torchcodec==0.10.0 \
    timm==1.0.14 \
    kornia==0.7.4 \
    accelerate==1.2.1 \
    peft==0.17.0 \
    diffusers==0.35.1 \
    tianshou==0.5.1 && \

# Step 2: Everything else — let pip resolve transitive deps normally.
echo "Installing remaining GR00T dependencies (with deps)..."
$PYTHON_CMD -m pip install --target "$GROOT_DEPS_DIR" --no-build-isolation --use-pep517 \
    lmdb==1.7.5 \
    albumentations==1.4.18 \
    blessings==1.7 \
    dm_tree==0.1.8 \
    einops==0.8.1 \
    gymnasium==1.0.0 \
    h5py==3.12.1 \
    imageio==2.34.2 \
    matplotlib==3.10.1 \
    numpydantic==1.6.7 \
    omegaconf==2.3.0 \
    pandas==2.2.3 \
    pydantic==2.10.6 \
    PyYAML==6.0.2 \
    ray==2.47.0 \
    Requests==2.32.3 \
    tqdm==4.67.1 \
    "tokenizers>=0.21,<0.22" \
    transformers==4.51.3 \
    wandb==0.23.0 \
    fastparquet==2024.11.0 \
    protobuf==3.20.3 \
    onnx==1.17.0 \
    pytest \
    hydra-core \
    tyro && \

# Add GR00T deps to sys.path *after* site-packages via .pth (so we never override Isaac Sim packages)
SITE_PACKAGES=$($PYTHON_CMD -c "import site; print(site.getsitepackages()[0])")
echo "$GROOT_DEPS_DIR" > "$SITE_PACKAGES/groot_deps.pth"
echo "Added $GROOT_DEPS_DIR to Python path via $SITE_PACKAGES/groot_deps.pth"
echo "export GROOT_DEPS_DIR=$GROOT_DEPS_DIR" >> /etc/bash.bashrc

##########################
# Environment finalization
##########################

if [[ "$USE_SERVER_ENV" -eq 0 ]]; then
  # Ensure pytorch torchrun script is in PATH
  echo "Ensuring pytorch torchrun script is in PATH..."
  echo "export PATH=/isaac-sim/kit/python/bin:\$PATH" >> /etc/bash.bashrc
fi

echo "GR00T dependencies installation completed successfully"
