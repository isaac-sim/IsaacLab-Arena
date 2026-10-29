#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_REPO="https://github.com/Physical-Intelligence/openpi"

NGC_NAMESPACE="nvcr.io/nvstaging/isaac-amr"
IMAGE_NAME="isaaclab_arena"
IMAGE_TAG="openpi_server"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
NGC_PATH="${NGC_NAMESPACE}/${IMAGE_REF}"

PUSH_TO_NGC=false
NO_CACHE=""

print_help() {
    cat <<EOF
Build (and optionally push) the openpi inference-server image.

Usage:
  $(basename "$0") [-p] [-R]

Options:
  -p              Push the built image to NGC (${NGC_PATH}).
  -R              Do not use the Docker build cache.
  -h              Show this help and exit.

Pushing assumes you have already run \`docker login nvcr.io\`.
EOF
}

while getopts ":pRh" opt; do
    case "$opt" in
        p) PUSH_TO_NGC=true ;;
        R) NO_CACHE="--no-cache" ;;
        h) print_help; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
    esac
done

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PINNED_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/OPENPI_COMMIT")"
OPENPI_DIR="$TMPDIR/openpi"
echo "Cloning openpi at ${PINNED_COMMIT} ..."
# Partial clone: skip blob objects, git fetches them on demand at checkout.
# Cuts the one-time clone size on a ~GB-scale repo without changing what we end up with.
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none "$OPENPI_REPO" "$OPENPI_DIR"
(cd "$OPENPI_DIR" && git checkout "$PINNED_COMMIT")

# Upstream's Dockerfile installs deps but expects source to be volume-mounted.
# Append a COPY step so the image is self-contained.
cat "$OPENPI_DIR/scripts/docker/serve_policy.Dockerfile" > "$TMPDIR/Dockerfile"
echo "COPY . /app" >> "$TMPDIR/Dockerfile"

echo "Building ${IMAGE_REF}"
docker build \
    --network=host \
    $NO_CACHE \
    -f "$TMPDIR/Dockerfile" \
    -t "${IMAGE_REF}" \
    "$OPENPI_DIR"

if [ "$PUSH_TO_NGC" = true ]; then
    echo "Pushing container to ${NGC_PATH}."
    docker tag "${IMAGE_REF}" "${NGC_PATH}"
    docker push "${NGC_PATH}"
    echo "Pushing complete."
fi
