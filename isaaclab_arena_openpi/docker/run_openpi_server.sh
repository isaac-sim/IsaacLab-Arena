#!/usr/bin/env bash
# Build (if needed) and run the openpi inference server in Docker.
#
# Usage:
#   ./run_openpi_server.sh                              # build if missing, then run pi05
#   ./run_openpi_server.sh -r                           # force rebuild, then run
#   ./run_openpi_server.sh -p 8001                      # run on a non-default port
#   ./run_openpi_server.sh -v pi0                       # run the pi0 variant
#   ./run_openpi_server.sh -b 64                        # allow up to 64 observations per batched request
#   ./run_openpi_server.sh -u                           # upstream serve_policy.py, no batched requests
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
MAX_BATCH_SIZE="32"
UPSTREAM_SERVER=false
SERVER_DIR="$(cd "${SCRIPT_DIR}/../server" && pwd)"

print_help() {
    cat <<EOF
Helper script to build and run the openpi inference server in Docker.

Usage:
  $(basename "$0") [options]

Options:
  -r              Force rebuilding of the server image.
  -p <port>       Port to serve on. Defaults to 8000.
  -v <variant>    Policy variant to serve: pi05 (default) or pi0.
  -b <size>       Largest batch of observations per request (default 32). Arena clients send the
                  chunk requests of all environments in one batched request; bigger batches need
                  more GPU memory on the server.
  -u              Serve with openpi's upstream serve_policy.py (one observation per request).
  -h              Show this help and exit.
EOF
}

while getopts ":rp:v:b:uh" opt; do
    case "$opt" in
        r) FORCE_REBUILD=true ;;
        p) PORT="$OPTARG" ;;
        v) VARIANT="$OPTARG" ;;
        b) MAX_BATCH_SIZE="$OPTARG" ;;
        u) UPSTREAM_SERVER=true ;;
        h) print_help; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
        :) echo "option -$OPTARG requires an argument" >&2; exit 1 ;;
    esac
done

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

if [ "$UPSTREAM_SERVER" = true ]; then
    SERVE_COMMAND=(uv run scripts/serve_policy.py)
    echo "Running ${IMAGE_NAME}:${IMAGE_TAG} (variant: ${VARIANT}, port: ${PORT}, upstream server)"
else
    SERVE_COMMAND=(uv run python /arena_server/serve_batched_policy.py --max-batch-size="${MAX_BATCH_SIZE}")
    echo "Running ${IMAGE_NAME}:${IMAGE_TAG} (variant: ${VARIANT}, port: ${PORT}, max batch size: ${MAX_BATCH_SIZE})"
fi

mkdir -p "$OPENPI_CACHE_DIR"
SERVER_RAN=true

docker run --rm -it --gpus all --network=host \
    -e OPENPI_DATA_HOME=/cache/openpi \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    -v "${OPENPI_CACHE_DIR}:/cache/openpi" \
    -v "${SERVER_DIR}:/arena_server:ro" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    "${SERVE_COMMAND[@]}" --port="${PORT}" policy:checkpoint \
        --policy.config="${POLICY_CONFIG}" \
        --policy.dir="${POLICY_DIR}"
