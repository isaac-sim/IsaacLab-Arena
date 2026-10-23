#!/bin/bash
# Build an Arena docker image
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
IMAGE=""
BUILD_ARGS=()

usage() {
    echo "Usage: $0 [-n image:tag] [-R]"
    echo "  -n  Output image reference (default: isaaclab_arena:latest)"
    echo "  -R  Build without cache; base image metadata is always refreshed"
}

while getopts ':n:Rh' option; do
    case "$option" in
        n) IMAGE=$OPTARG ;;
        R) BUILD_ARGS+=(--no-cache) ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [ "$#" -ne 0 ]; then
    usage >&2
    exit 2
fi
IMAGE=${IMAGE:-isaaclab_arena:latest}

started=$SECONDS
status=0
docker build --pull --progress=plain \
    --target dev \
    --build-arg WORKDIR=/workspaces/isaaclab_arena \
    "${BUILD_ARGS[@]}" \
    --tag "$IMAGE" --file "$SCRIPT_DIR/Dockerfile.isaaclab_arena" "$SCRIPT_DIR/.." || status=$?
echo "Arena build image=$IMAGE exit=$status elapsed=$((SECONDS - started))s"
exit "$status"
