#!/bin/bash
set -euo pipefail

ISAACLAB_ARENA_IMAGE_NAME='isaaclab_arena'
DOCKER_TARGET=dev
TAG_NAME=""
PUSH_TO_NGC=false
BUILD_OPTIONS=()

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/setup/target_tag.sh"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [-t tag] [-R] [-p] [-v]

Build an Arena developer image, optionally tagging and pushing it to NGC.

Options:
  -t tag   Output tag (default: latest).
  -R       Build without cache.
  -p       Push the built image to NGC.
  -v       Verbose output.
  -h       Show this help.

Examples:
  $(basename "$0") -R -p -t candidate
USAGE
}

while getopts ':t:vpRh' OPTION; do
    case "$OPTION" in
        t) TAG_NAME=$OPTARG ;;
        v) set -x ;;
        p) PUSH_TO_NGC=true ;;
        R) BUILD_OPTIONS+=(-R) ;;
        h) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
shift $((OPTIND - 1))
if [ "$#" -ne 0 ]; then
    usage >&2
    exit 2
fi

# An explicit tag takes precedence over the target's default, in either option order.
DEFAULT_TAG=$(default_tag_for_target "$DOCKER_TARGET")
TAG_NAME=${TAG_NAME:-$DEFAULT_TAG}
DOCKER_IMAGE_NAME="${ISAACLAB_ARENA_IMAGE_NAME}:${TAG_NAME}"
NGC_PATH="nvcr.io/nvstaging/isaac-amr/${DOCKER_IMAGE_NAME}"
echo "Building target ${DOCKER_TARGET} as ${DOCKER_IMAGE_NAME}."
echo "NGC_PATH is ${NGC_PATH}."

# Share target selection and cache handling with local builds.
"$SCRIPT_DIR/build_docker.sh" -t "$DOCKER_TARGET" \
    -n "$DOCKER_IMAGE_NAME" "${BUILD_OPTIONS[@]}"


if [ "$PUSH_TO_NGC" = true ]; then
    echo "Pushing image to ${NGC_PATH}."
    docker tag "$DOCKER_IMAGE_NAME" "$NGC_PATH"
    docker push "$NGC_PATH"
    echo "Pushing complete."
else
    echo "Not pushing to NGC. Use -p to push to NGC."
fi
