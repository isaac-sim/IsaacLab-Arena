#!/usr/bin/env bash
# Build (if needed) and run the openpi inference server in Docker.
#
# Usage:
#   ./run_openpi_server.sh                              # build if missing, then run pi05
#   ./run_openpi_server.sh -r                           # force rebuild, then run
#   ./run_openpi_server.sh -v pi0                       # run the pi0 variant
#   ./run_openpi_server.sh -s <path>                    # build from a local openpi checkout (implies rebuild)
#   ./run_openpi_server.sh -h                           # help
#
# The pinned commit lives in the OPENPI_COMMIT file next to this script. To bump
# it, edit that file. The same commit is also installed into the arena base
# image (see docker/Dockerfile.isaaclab_arena) so the in-container client stays
# wire-compatible with the server image built here.
#
# Env overrides:
#   IMAGE_NAME    Local image name (default: isaaclab_arena_openpi-server)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-isaaclab_arena_openpi-server}"
IMAGE_TAG="latest"
OPENPI_REPO="https://github.com/Physical-Intelligence/openpi"

VARIANT="pi05"
FORCE_REBUILD=false
SRC_DIR=""

print_help() {
    cat <<EOF
Helper script to build and run the openpi inference server in Docker.

Usage:
  $(basename "$0") [options]

Options:
  -r              Force rebuilding of the server image.
  -v <variant>    Policy variant to serve: pi05 (default) or pi0.
  -s <path>       Build from a local openpi checkout instead of cloning at the
                  pinned commit. Implies -r.
  -h              Show this help and exit.
EOF
}

while getopts ":rv:s:h" opt; do
    case "$opt" in
        r) FORCE_REBUILD=true ;;
        v) VARIANT="$OPTARG" ;;
        s) SRC_DIR="$OPTARG"; FORCE_REBUILD=true ;;
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

# Single EXIT handler: remove the build tempdir and reset cache ownership back to us.
BUILD_TMPDIR=""
SERVER_RAN=false
cleanup() {
    [ -n "$BUILD_TMPDIR" ] && rm -rf "$BUILD_TMPDIR"
    if [ "$SERVER_RAN" = true ]; then
        docker run --rm -v "${OPENPI_CACHE_DIR}:/cache/openpi" \
            "${IMAGE_NAME}:${IMAGE_TAG}" \
            chown -R "$(id -u):$(id -g)" /cache/openpi || true
    fi
}
trap cleanup EXIT

if [ "$FORCE_REBUILD" = true ] || \
   [ -z "$(docker images -q "${IMAGE_NAME}:${IMAGE_TAG}" 2>/dev/null)" ]; then
    BUILD_TMPDIR=$(mktemp -d)

    if [ -n "$SRC_DIR" ]; then
        OPENPI_DIR="$SRC_DIR"
        echo "Using local openpi checkout at ${OPENPI_DIR}"
    else
        PINNED_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/OPENPI_COMMIT")"
        OPENPI_DIR="$BUILD_TMPDIR/openpi"
        echo "Cloning openpi at ${PINNED_COMMIT} ..."
        # Partial clone: skip blob objects, git fetches them on demand at checkout.
        # Cuts the one-time clone size on a ~GB-scale repo without changing what we end up with.
        GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none "$OPENPI_REPO" "$OPENPI_DIR"
        (cd "$OPENPI_DIR" && git checkout "$PINNED_COMMIT")
    fi

    # Upstream's Dockerfile installs deps but expects source to be volume-mounted.
    # Append a COPY step so the image is self-contained.
    cat "$OPENPI_DIR/scripts/docker/serve_policy.Dockerfile" > "$BUILD_TMPDIR/Dockerfile"
    echo "COPY . /app" >> "$BUILD_TMPDIR/Dockerfile"

    echo "Building ${IMAGE_NAME}:${IMAGE_TAG}"
    docker build \
        --network=host \
        -f "$BUILD_TMPDIR/Dockerfile" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        "$OPENPI_DIR"
else
    echo "Image ${IMAGE_NAME}:${IMAGE_TAG} already exists. Not rebuilding (use -r to force)."
fi

echo "Running ${IMAGE_NAME}:${IMAGE_TAG} (variant: ${VARIANT})"

mkdir -p "$OPENPI_CACHE_DIR"
SERVER_RAN=true

docker run --rm -it --gpus all --network=host \
    -e OPENPI_DATA_HOME=/cache/openpi \
    -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
    -v "${OPENPI_CACHE_DIR}:/cache/openpi" \
    "${IMAGE_NAME}:${IMAGE_TAG}" \
    uv run scripts/serve_policy.py policy:checkpoint \
        --policy.config="${POLICY_CONFIG}" \
        --policy.dir="${POLICY_DIR}"
