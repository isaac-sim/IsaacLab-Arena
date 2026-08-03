#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build (if needed) and run the Cosmos3-Edge-Policy inference server in Docker.
#
# Usage:
#   ./run_cosmos_server.sh                  # build if missing, then serve the DROID policy
#   ./run_cosmos_server.sh -r               # force rebuild, then run
#   ./run_cosmos_server.sh -p 9000          # serve on a different port
#   ./run_cosmos_server.sh -h               # help
#
# The pinned commit lives in the COSMOS_COMMIT file next to this script. To bump
# it, edit that file and rebuild with -r.
#
# The checkpoint is baked into the image at build time, so running the server needs no HF_TOKEN
# (only building the image does; see build_server_image.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="isaaclab_arena"
IMAGE_TAG="cosmos_server"

# Path the checkpoint is baked to inside the image (see build_server_image.sh CHECKPOINT_LOCAL_PATH).
CHECKPOINT="/workspace/baked_checkpoint"
PORT="8000"
FORCE_REBUILD=false

print_help() {
    cat <<EOF
Helper script to build and run the Cosmos3-Edge-Policy inference server in Docker.

Usage:
  $(basename "$0") [options]

Options:
  -r              Force rebuilding of the server image.
  -p <port>       Port to serve on (default: ${PORT}).
  -h              Show this help and exit.

The checkpoint is baked into the image, so no HF_TOKEN is needed to run (building the image does;
see build_server_image.sh).
EOF
}

while getopts ":rp:h" opt; do
    case "$opt" in
        r) FORCE_REBUILD=true ;;
        p) PORT="$OPTARG" ;;
        h) print_help; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
        :) echo "option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [ "$FORCE_REBUILD" = true ] || \
   [ -z "$(docker images -q "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null)" ]; then
    "${SCRIPT_DIR}/build_server_image.sh"
else
    echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already exists. Not rebuilding (use -r to force)."
fi

echo "Running ${IMAGE_NAME}:${IMAGE_TAG} (checkpoint: ${CHECKPOINT}, port: ${PORT})"

docker run --rm -it --gpus all --network=host \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -m cosmos_framework.scripts.action_policy_server_robolab \
        --checkpoint_path "${CHECKPOINT}" \
        --port "${PORT}" \
        --format-prompt-as-json True
