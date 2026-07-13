#!/usr/bin/env bash
# Build (if needed) and run the cosmos3 inference server in Docker.
#
# Usage:
#   ./run_cosmos3_server.sh                           # build if missing, then run
#   ./run_cosmos3_server.sh -r                        # force rebuild, then run
#   ./run_cosmos3_server.sh -s <path>                 # build from a local cosmos-framework checkout (implies rebuild)
#   ./run_cosmos3_server.sh -p <port>                 # listen on a custom port (default: 8000)
#   ./run_cosmos3_server.sh -h                        # help
#
# The pinned commit lives in the COSMOS3_COMMIT file next to this script. To bump
# it, edit that file.
#
# The server container must have access to the cosmos3 model weights. Set the
# HF_TOKEN environment variable or mount a HF cache directory with:
#   -e HF_TOKEN=$HF_TOKEN -v /path/to/hf_cache:/workspace/.cache/huggingface
#
# Env overrides:
#   IMAGE_NAME    Local image name (default: isaaclab_arena_cosmos3-server)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-isaaclab_arena_cosmos3-server}"
IMAGE_TAG="latest"
COSMOS_REPO="https://github.com/NVIDIA/cosmos-framework"

PORT=8000
FORCE_REBUILD=false
SRC_DIR=""

print_help() {
    cat <<EOF
Helper script to build and run the cosmos3 inference server in Docker.

Usage:
  $(basename "$0") [options]

Options:
  -r              Force rebuilding of the server image.
  -s <path>       Build from a local cosmos-framework checkout instead of cloning
                  at the pinned commit. Implies -r.
  -p <port>       Port for the policy server to listen on (default: 8000).
  -h              Show this help and exit.
EOF
}

while getopts ":rs:p:h" opt; do
    case "$opt" in
        r) FORCE_REBUILD=true ;;
        s) SRC_DIR="$OPTARG"; FORCE_REBUILD=true ;;
        p) PORT="$OPTARG" ;;
        h) print_help; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
        :) echo "option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [ "$FORCE_REBUILD" = true ] || \
   [ -z "$(docker images -q "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null)" ]; then
    TMPDIR=$(mktemp -d)
    trap 'rm -rf "$TMPDIR"' EXIT

    if [ -n "$SRC_DIR" ]; then
        COSMOS_DIR="$SRC_DIR"
        echo "Using local cosmos-framework checkout at ${COSMOS_DIR}"
    else
        PINNED_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/COSMOS3_COMMIT")"
        COSMOS_DIR="$TMPDIR/cosmos-framework"
        echo "Cloning cosmos-framework at ${PINNED_COMMIT} ..."
        GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none "$COSMOS_REPO" "$COSMOS_DIR"
        (cd "$COSMOS_DIR" && git checkout "$PINNED_COMMIT")
    fi

    # Build a Docker image from cosmos-framework.
    # cosmos-framework provides its own Dockerfile; we use it directly.
    echo "Building ${IMAGE_NAME}:${IMAGE_TAG} from ${COSMOS_DIR}"
    docker build \
        --network=host \
        -f "$COSMOS_DIR/docker/Dockerfile" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        "$COSMOS_DIR"
else
    echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already exists. Not rebuilding (use -r to force)."
fi

echo "Running ${IMAGE_NAME}:${IMAGE_TAG} on port ${PORT}"

docker run --rm -it --gpus all --network=host \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    python -m cosmos_framework.scripts.action_policy_server_robolab --port "${PORT}"
