#!/usr/bin/env bash
# Encapsulates the openpi inference server as a self-contained Docker image.
#
# Usage:
#   ./build_openpi_server.sh                        # build at DEFAULT_COMMIT
#   ./build_openpi_server.sh <commit>               # build at a specific commit
#   ./build_openpi_server.sh --push                 # build and push to NGC
#   ./build_openpi_server.sh <commit> --push

set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-isaaclab_arena_openpi-server}"
NGC_PATH="${NGC_PATH:-nvcr.io/nvstaging/isaac-amr/${IMAGE_NAME}}"
OPENPI_REPO="https://github.com/Physical-Intelligence/openpi"
DEFAULT_COMMIT="c23745b5ad24e98f66967ea795a07b2588ed6c79"
PUSH=false

COMMIT=""
for arg in "$@"; do
    case "$arg" in
        --push) PUSH=true ;;
        *) COMMIT="$arg" ;;
    esac
done

COMMIT="${COMMIT:-$DEFAULT_COMMIT}"

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "Cloning openpi at ${COMMIT} ..."
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet "$OPENPI_REPO" "$TMPDIR/openpi"
cd "$TMPDIR/openpi"
git checkout "$COMMIT"

SHORT_HASH=$(git rev-parse --short HEAD)

echo "Building ${IMAGE_NAME}:${SHORT_HASH}"

# Upstream's Dockerfile installs deps but expects source to be volume-mounted.
# Append a COPY step so the image is self-contained.
cat scripts/docker/serve_policy.Dockerfile > "$TMPDIR/Dockerfile"
cat >> "$TMPDIR/Dockerfile" <<'PATCH'

# --- isaaclab_arena_openpi patches ---
COPY . /app
PATCH

docker build \
    --network=host \
    -f "$TMPDIR/Dockerfile" \
    -t "${IMAGE_NAME}:${SHORT_HASH}" \
    -t "${IMAGE_NAME}:latest" \
    .

echo "Built ${IMAGE_NAME}:${SHORT_HASH} (also tagged :latest)"

if [ "$PUSH" = true ]; then
    echo "Pushing to ${NGC_PATH}:${SHORT_HASH}"
    docker tag "${IMAGE_NAME}:${SHORT_HASH}" "${NGC_PATH}:${SHORT_HASH}"
    docker push "${NGC_PATH}:${SHORT_HASH}"
    echo "Pushed ${NGC_PATH}:${SHORT_HASH}"
fi
