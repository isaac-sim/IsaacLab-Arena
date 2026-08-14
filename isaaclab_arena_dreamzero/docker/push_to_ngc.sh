#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

NGC_NAMESPACE="${NGC_NAMESPACE:-nvcr.io/nvstaging/isaac-amr}"
IMAGE_NAME=isaaclab_arena
TAG_NAME=dreamzero-server-commit-checkpoints-20260709-130610
PUSH_TO_NGC=false
NO_CACHE=""

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

print_help() {
    local script_name
    local ngc_image_reference
    script_name="$(basename "$0")"
    ngc_image_reference="${NGC_NAMESPACE}/${IMAGE_NAME}:${TAG_NAME}"

    cat <<EOF
Maintainer helper for building the DreamZero inference-server image.

Usage:
  ${script_name} [options]

Examples:
- Build without cache and push the default image to NGC:
    ${script_name} -R -p
- Build and push a new tag:
    ${script_name} -p -t <tag_name>
- See this help message:
    ${script_name} -h

Options:
  -p              Push the image to NGC.
  -t <tag>        Override the image tag. Default: ${TAG_NAME}.
  -n <name>       Override the image name. Default: ${IMAGE_NAME}.
  -R              Do not use the Docker build cache.
  -v              Print commands as they execute.
  -h              Show this help and exit.

Default NGC image: ${ngc_image_reference}
Set NGC_NAMESPACE to build for a different registry namespace.
EOF
}

while getopts ":t:n:vpRh" OPTION; do
    case $OPTION in
        t)
            TAG_NAME=$OPTARG
            echo "Tag name is ${TAG_NAME}."
            ;;
        n)
            IMAGE_NAME=$OPTARG
            echo "Image name is ${IMAGE_NAME}."
            ;;
        v)
            set -x
            ;;
        p)
            PUSH_TO_NGC="true"
            echo "PUSH_TO_NGC (build and push to ngc)."
            ;;
        R)
            NO_CACHE="--no-cache"
            ;;
        h)
            print_help
            exit 0
            ;;
        \?)
            echo "Unknown option: -${OPTARG}" >&2
            print_help >&2
            exit 1
            ;;
        :)
            echo "Option -${OPTARG} requires an argument." >&2
            print_help >&2
            exit 1
            ;;
    esac
done

shift $((OPTIND - 1))

LOCAL_IMAGE_REFERENCE="${IMAGE_NAME}:${TAG_NAME}"
NGC_IMAGE_REFERENCE="${NGC_NAMESPACE}/${LOCAL_IMAGE_REFERENCE}"

# Build the image.
docker build --pull \
    $NO_CACHE \
    -t "${LOCAL_IMAGE_REFERENCE}" \
    --file "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

# Push if requested.
if [ "$PUSH_TO_NGC" = true ]; then

    # Tag and push the image to NGC.
    echo "Pushing container to ${NGC_IMAGE_REFERENCE}."
    docker tag "${LOCAL_IMAGE_REFERENCE}" "${NGC_IMAGE_REFERENCE}"
    docker push "${NGC_IMAGE_REFERENCE}"
    echo "Pushing complete."

else

    echo "Not pushing to NGC. Use -p to push to NGC."

fi
