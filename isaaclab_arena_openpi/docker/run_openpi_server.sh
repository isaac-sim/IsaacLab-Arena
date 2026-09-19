#!/usr/bin/env bash
# Build (if needed) and run the openpi inference server in Docker.
#
# Usage:
#   ./run_openpi_server.sh                              # build if missing, then run pi05
#   ./run_openpi_server.sh -r                           # force rebuild, then run
#   ./run_openpi_server.sh -p 8001                      # run on a non-default port
#   ./run_openpi_server.sh -v pi0                       # run the pi0 variant
#   ./run_openpi_server.sh -g 0                         # expose only GPU 0 to the server
#   ./run_openpi_server.sh -g GPU-... -c                # use the CUDA 13 compatibility helpers
#   ./run_openpi_server.sh -h                           # help
#
# The pinned commit lives in the OPENPI_COMMIT file next to this script. To bump
# it, edit that file. The same commit is also installed into the arena base
# image (see docker/Dockerfile.isaaclab_arena) so the in-container client stays
# wire-compatible with the server image built here.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="isaaclab_arena"
IMAGE_TAG="openpi_server"

VARIANT="pi05"
PORT="8000"
FORCE_REBUILD=false
GPU="all"
CUDA13=false

print_help() {
    cat <<EOF
Helper script to build and run the openpi inference server in Docker.

Usage:
  $(basename "$0") [options]

Options:
  -r              Force rebuilding of the server image.
  -p <port>       Port to serve on. Defaults to 8000.
  -v <variant>    Policy variant to serve: pi05 (default) or pi0.
  -g <gpu>        Expose one GPU by index or full GPU UUID. Defaults to all GPUs.
  -c              Apply the CUDA 13 compatibility helpers for GB300 in the server container.
  -h              Show this help and exit.

Environment:
  XLA_PYTHON_CLIENT_MEM_FRACTION   JAX memory fraction (default: 0.5).
  XLA_PYTHON_CLIENT_PREALLOCATE    Optional JAX preallocation override, e.g. false.
EOF
}

while getopts ":rp:v:g:ch" opt; do
    case "$opt" in
        r) FORCE_REBUILD=true ;;
        p) PORT="$OPTARG" ;;
        v) VARIANT="$OPTARG" ;;
        g) GPU="$OPTARG" ;;
        c) CUDA13=true ;;
        h) print_help; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
        :) echo "option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

if [[ "$GPU" != "all" && ! "$GPU" =~ ^[0-9]+$ &&
      ! "$GPU" =~ ^GPU-[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]]; then
    echo "invalid -g GPU: $GPU (expected an index or full GPU UUID)" >&2
    exit 1
fi

case "$VARIANT" in
    pi05)
        POLICY_CONFIG="pi05_droid_jointpos_polaris"
        POLICY_DIR="gs://openpi-assets-simeval/pi05_droid_jointpos"
        ;;
    pi0)
        POLICY_CONFIG="pi0_droid_jointpos_polaris"
        POLICY_DIR="gs://openpi-assets-simeval/pi0_droid_jointpos"
        ;;
    *)
        echo "unknown -v variant: $VARIANT (expected pi05 or pi0)" >&2
        exit 1
        ;;
esac

# Cache the ~11GB checkpoint that openpi pulls from gs:// across runs.
OPENPI_CACHE_DIR="${OPENPI_CACHE_DIR:-$HOME/.cache/openpi}"

# EXIT handler: reset cache ownership back to us (the container writes it as root).
SERVER_RAN=false
cleanup() {
    if [ "$SERVER_RAN" = true ]; then
        docker run --rm -v "${OPENPI_CACHE_DIR}:/cache/openpi" \
            "${IMAGE_NAME}:${IMAGE_TAG}" \
            chown -R "$(id -u):$(id -g)" /cache/openpi || true
    fi
}
trap cleanup EXIT

if [ "$FORCE_REBUILD" = true ] || \
   [ -z "$(docker images -q "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null)" ]; then
    "${SCRIPT_DIR}/build_server_image.sh"
else
    echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already exists. Not rebuilding (use -r to force)."
fi

echo "Running ${IMAGE_NAME}:${IMAGE_TAG} (variant: ${VARIANT}, port: ${PORT}, GPU: ${GPU}, CUDA 13: ${CUDA13})"

mkdir -p "$OPENPI_CACHE_DIR"

GPU_REQUEST="$GPU"
if [ "$GPU" != "all" ]; then
    GPU_REQUEST="device=${GPU}"
fi
DOCKER_ARGS=(
    --rm -it --gpus "$GPU_REQUEST" --network=host
    -e OPENPI_DATA_HOME=/cache/openpi
    -e "XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"
    -v "${OPENPI_CACHE_DIR}:/cache/openpi"
)
if [ "${XLA_PYTHON_CLIENT_PREALLOCATE+x}" ]; then
    DOCKER_ARGS+=(-e "XLA_PYTHON_CLIENT_PREALLOCATE=${XLA_PYTHON_CLIENT_PREALLOCATE}")
fi

SERVER_COMMAND=(
    uv run scripts/serve_policy.py --port="${PORT}" policy:checkpoint
    --policy.config="${POLICY_CONFIG}" --policy.dir="${POLICY_DIR}"
)
if [ "$CUDA13" = true ]; then
    GB300_TOOLS_DIR="$(cd "${SCRIPT_DIR}/../../tools/gb300" && pwd)"
    DOCKER_ARGS+=(
        -v "${GB300_TOOLS_DIR}:/opt/arena-gb300:ro"
        -e PYTHONPATH=/app/src:/app/packages/openpi-client/src
    )
    # Invoke the prepared interpreter directly: uv run would restore the project's CUDA 12 pins.
    SERVER_COMMAND=(
        bash -c 'bash /opt/arena-gb300/prepare_openpi_cuda13.sh /.venv/bin/python && exec /.venv/bin/python /opt/arena-gb300/serve_openpi_cuda13.py "$@"'
        -- --port "$PORT" --policy-config "$POLICY_CONFIG" --policy-dir "$POLICY_DIR"
    )
fi

SERVER_RAN=true
docker run "${DOCKER_ARGS[@]}" "${IMAGE_NAME}:${IMAGE_TAG}" "${SERVER_COMMAND[@]}"
