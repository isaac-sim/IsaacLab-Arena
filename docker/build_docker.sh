#!/bin/bash
# Build an Arena docker image
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup/target_tag.sh"
TARGET=dev
IMAGE=""
BUILD_ARGS=()

usage() {
    echo "Usage: $0 [-t dev|dev-curobo] [-n image:tag] [-R]"
    echo "  -t  Docker target (default: dev)"
    echo "  -n  Output image reference (default: isaaclab_arena:<target compatibility tag>)"
    echo "  -R  Build without cache; base image metadata is always refreshed"
}

while getopts ':t:n:Rh' option; do
    case "$option" in
        t) TARGET=$OPTARG ;;
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
DEFAULT_TAG=$(default_tag_for_target "$TARGET")
IMAGE=${IMAGE:-isaaclab_arena:$DEFAULT_TAG}

started=$SECONDS
status=0
docker build --pull --progress=plain \
    --target "$TARGET" \
    --build-arg WORKDIR=/workspaces/isaaclab_arena \
    "${BUILD_ARGS[@]}" \
    --tag "$IMAGE" --file "$SCRIPT_DIR/Dockerfile.isaaclab_arena" "$SCRIPT_DIR/.." || status=$?
echo "Arena build target=$TARGET image=$IMAGE exit=$status elapsed=$((SECONDS - started))s"
exit "$status"
