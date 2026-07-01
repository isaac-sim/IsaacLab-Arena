#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
set -e

IMAGE_NAME=dreamzero_inference_server
TAG_NAME=latest
PUSH_TO_NGC=false
HF_TOKEN=

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while getopts ":t:n:vn:pn:Rn:hn:" OPTION; do
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
        h | *)
            script_name=$(basename "$0")
            echo "Helper script for building and pushing the DreamZero inference server image to NGC."
            echo ""
            echo "Usage:"
            echo "  ${script_name} <HF_TOKEN> [options]"
            echo ""
            echo "Examples:"
            echo "- Build without cache and push to NGC:"
            echo "    ${script_name} <HF_TOKEN> -R -p -t <tag_name>"
            echo "- See help message:"
            echo "    ${script_name} -h"
            echo ""
            echo "Options:"
            echo "  -p - Push the image to NGC."
            echo "  -t - Tag name of the image."
            echo "  -n - Override the image name. Default: dreamzero_inference_server."
            echo '  -R - Do not use cache when building the image.'
            echo "  -v - Verbose output."
            echo "  -h - Help (this output)"
            exit 0
            ;;
    esac
done

shift $((OPTIND - 1))
if [ -n "${1:-}" ]; then
    HF_TOKEN="$1"
fi

DOCKER_IMAGE_NAME=${IMAGE_NAME}:${TAG_NAME}
NGC_PATH=nvcr.io/nvidian/${DOCKER_IMAGE_NAME}

# Build the image.
docker build --pull \
    $NO_CACHE \
    --build-arg HF_TOKEN="${HF_TOKEN}" \
    -t ${DOCKER_IMAGE_NAME} \
    --file ${SCRIPT_DIR}/Dockerfile \
    ${SCRIPT_DIR}

# Push if requested.
if [ "$PUSH_TO_NGC" = true ]; then

    # Tag and push the image to NGC.
    echo "Pushing container to ${NGC_PATH}."
    docker tag ${DOCKER_IMAGE_NAME} ${NGC_PATH}
    docker push ${NGC_PATH}
    echo "Pushing complete."

else

    echo "Not pushing to NGC. Use -p to push to NGC."

fi
