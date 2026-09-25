#!/bin/bash
set -euo pipefail

# Script to install CUDA 12.8 for cuRobo dependencies
echo "Installing CUDA 12.8 for cuRobo dependencies"

# Read the distribution name and version from the base image.
. /etc/os-release

# Detect Ubuntu version and set appropriate CUDA repository
case "$ID" in
  ubuntu)
    case "$VERSION_ID" in
      "20.04") cuda_repo="ubuntu2004";;
      "22.04") cuda_repo="ubuntu2204";;
      "24.04") cuda_repo="ubuntu2404";;
      *) echo "Unsupported Ubuntu $VERSION_ID"; exit 1;;
    esac ;;
  *) echo "Unsupported base OS: $ID"; exit 1 ;;
esac

echo "Detected Ubuntu $VERSION_ID, using repository: $cuda_repo"

# Update package lists and install prerequisites
apt-get update
apt-get install -y --no-install-recommends wget gnupg ca-certificates

# Install NVIDIA's repository configuration and signing key so apt can verify CUDA packages.
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb

# Give NVIDIA's CUDA repository priority when apt chooses between available package versions.
wget -q https://developer.download.nvidia.com/compute/cuda/repos/${cuda_repo}/x86_64/cuda-${cuda_repo}.pin
mv cuda-${cuda_repo}.pin /etc/apt/preferences.d/cuda-repository-pin-600

# Update package lists with new CUDA repository
apt-get update

# Install the CUDA 12.8 compiler and development libraries needed to build cuRobo.
apt-get install -y --no-install-recommends cuda-toolkit-12-8

# Remove unneeded packages, downloaded package archives, and apt lists to reduce the image layer.
apt-get -y autoremove
apt-get clean
rm -rf /var/lib/apt/lists/*

# Record CUDA paths and the default GPU architecture in the image's environment file.
echo "export CUDA_HOME=/usr/local/cuda-12.8" >> /etc/environment
echo "export PATH=/usr/local/cuda-12.8/bin:${PATH}" >> /etc/environment
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}" >> /etc/environment
echo "export TORCH_CUDA_ARCH_LIST=8.0+PTX" >> /etc/environment

echo "CUDA 12.8 installation completed successfully"
