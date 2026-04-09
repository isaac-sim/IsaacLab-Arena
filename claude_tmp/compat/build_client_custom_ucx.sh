#!/usr/bin/env bash
set -euo pipefail

UCX_VERSION="${UCX_VERSION:-v1.18.1}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/workspace/claude_tmp/client_ucx_install}"
BUILD_ROOT="${BUILD_ROOT:-/workspace/claude_tmp/client_ucx_build}"
CUDA_PREFIX="${CUDA_PREFIX:-/isaac-sim/exts/omni.isaac.ml_archive/pip_prebundle/nvidia/cuda_runtime}"
WITH_CUDA="${WITH_CUDA:-0}"

echo "[build_client_custom_ucx] UCX_VERSION=${UCX_VERSION}"
echo "[build_client_custom_ucx] INSTALL_PREFIX=${INSTALL_PREFIX}"
echo "[build_client_custom_ucx] BUILD_ROOT=${BUILD_ROOT}"
echo "[build_client_custom_ucx] CUDA_PREFIX=${CUDA_PREFIX}"
echo "[build_client_custom_ucx] WITH_CUDA=${WITH_CUDA}"

mkdir -p "${BUILD_ROOT}" "${INSTALL_PREFIX}"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends \
  autoconf \
  automake \
  libtool \
  pkg-config \
  rdma-core \
  libibverbs-dev \
  librdmacm-dev \
  ca-certificates \
  git

if [ ! -d "${BUILD_ROOT}/ucx/.git" ]; then
  git clone https://github.com/openucx/ucx.git "${BUILD_ROOT}/ucx"
fi

cd "${BUILD_ROOT}/ucx"
git fetch --tags
git checkout "${UCX_VERSION}"

./autogen.sh
mkdir -p build
cd build

if [ "${WITH_CUDA}" = "1" ]; then
  ../contrib/configure-release \
    --prefix="${INSTALL_PREFIX}" \
    --with-cuda="${CUDA_PREFIX}" \
    --enable-mt \
    --with-verbs \
    --with-rdmacm
else
  ../contrib/configure-release \
    --prefix="${INSTALL_PREFIX}" \
    --enable-mt \
    --with-verbs \
    --with-rdmacm
fi

make -j"$(nproc)"
make install

echo "[build_client_custom_ucx] ucx_info summary"
LD_LIBRARY_PATH="${INSTALL_PREFIX}/lib/ucx:${INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}" \
  "${INSTALL_PREFIX}/bin/ucx_info" -d | \
  grep -E 'Memory domain: mlx5|Transport: rc|Transport: dc|Connection manager: rdmacm|Device: mlx5' || true
