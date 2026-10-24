#!/bin/bash
# Install the pinned Isaac Lab checkout from its frozen uv lock.

set -euo pipefail
# Keep packages installed into the shared venv writable by the isaac-sim group.
umask 0002

readonly IMAGE_EXTRAS=(
    --extra rsl-rl
    --extra rerun
    --extra mimic
    --extra teleop
    --extra test
    --extra tetrahedralization
    --extra ov
)

sync_isaaclab() {
    cd "${ISAACLAB_PATH}"
    # Match Lab's image setup and defer imageio-ffmpeg to Arena's sync, where it
    # is an explicit dependency for HDF5-to-LeRobot video conversion.
    uv sync --frozen --inexact "${IMAGE_EXTRAS[@]}" \
        --no-install-package imageio-ffmpeg "$@"
    # uv preserves permissions from cached wheels, so repair cached files after installation.
    find "${VIRTUAL_ENV}" \( -type f -o -type d \) \
        \( ! -group isaac-sim -o ! -perm -g+w \) \
        -exec chgrp isaac-sim {} + -exec chmod g+rwX {} +
}

case "${1:-}" in
    dependencies)
        ln -sfn /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
        uv venv --seed --python /isaac-sim/kit/python/bin/python3 "${VIRTUAL_ENV}"
        chgrp -R isaac-sim "${VIRTUAL_ENV}"
        chmod -R g+rwX "${VIRTUAL_ENV}"
        find "${VIRTUAL_ENV}" -type d -exec chmod g+s {} +
        sync_isaaclab --no-install-local
        ;;
    workspace)
        find "${ISAACLAB_PATH}" -type f -name "*.sh" -exec sed -i 's/\r$//' {} +
        chmod 755 "${ISAACLAB_PATH}/isaaclab.sh"
        "${ISAACLAB_PATH}/isaaclab.sh" -p \
            "${ISAACLAB_PATH}/tools/install_deps.py" apt "${ISAACLAB_PATH}/source"
        apt-get -y autoremove
        apt-get clean
        rm -rf /var/lib/apt/lists/*
        sync_isaaclab
        # Prevent Sim 6.2's prebundled CUDA 12.8 Torch from shadowing Lab 3.0's
        # locked Torch 2.11/CUDA 13.0 packages on launch paths.
        # See submodules/IsaacLab/docker/Dockerfile.base:170-175
        "${ISAACLAB_PATH}/isaaclab.sh" -p -c \
            "from isaaclab.cli.commands.install import _repoint_prebundle_packages; _repoint_prebundle_packages()"
        ;;
    *)
        echo "Usage: $0 {dependencies|workspace}" >&2
        exit 2
        ;;
esac
