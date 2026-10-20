#!/bin/bash
# Install the pinned Isaac Lab checkout from its frozen uv lock.

set -euo pipefail

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
    uv sync --frozen --inexact "${IMAGE_EXTRAS[@]}" \
        --no-install-package imageio-ffmpeg "$@"
}

case "${1:-}" in
    dependencies)
        ln -sfn /isaac-sim/ "${ISAACLAB_PATH}/_isaac_sim"
        uv venv --seed "${VIRTUAL_ENV}"
        sync_isaaclab --no-install-workspace
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
        "${ISAACLAB_PATH}/isaaclab.sh" -p -c \
            "from isaaclab.cli.commands.install import _repoint_prebundle_packages; _repoint_prebundle_packages()"
        ;;
    *)
        echo "Usage: $0 {dependencies|workspace}" >&2
        exit 2
        ;;
esac
