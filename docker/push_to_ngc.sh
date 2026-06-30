#!/bin/bash
set -e

ISAACLAB_ARENA_IMAGE_NAME='isaaclab_arena'
TAG_NAME=latest
CONTAINER_ID=""
PUSH_TO_NGC=false
WORKDIR="/workspaces/isaaclab_arena"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while getopts ":t:n:vn:pn:Rn:hn:" OPTION; do
    case $OPTION in
        t)
            TAG_NAME=$OPTARG
            echo "Tag name is ${TAG_NAME}."
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
            echo "Helper script for pushing IsaacLab Arena docker image to NGC."
            echo ""
            echo "Usage:"
            echo "  ${script_name} [options]"
            echo ""
            echo "Examples:"
            echo "- Build without cache and push to NGC:"
            echo "    ${script_name} -R -p -t <tag_name>"
            echo "- See help message:"
            echo "    ${script_name} -h"
            echo ""
            echo "Options:"
            echo "  -p - Push the image to NGC."
            echo "  -t - Tag name of the image."
            echo '  -R - Do not use cache when building the image.'
            echo "  -v - Verbose output."
            echo "  -h - Help (this output)"
            exit 0
            ;;
    esac
done

# Get the NGC path.
DOCKER_IMAGE_NAME=${ISAACLAB_ARENA_IMAGE_NAME}:${TAG_NAME}
NGC_PATH=nvcr.io/nvstaging/isaac-amr/${DOCKER_IMAGE_NAME}

# Build the image.
docker build --pull \
    $NO_CACHE \
    --build-arg WORKDIR="${WORKDIR}" \
    -t ${DOCKER_IMAGE_NAME} \
    --file $SCRIPT_DIR/Dockerfile.isaaclab_arena \
    $SCRIPT_DIR/..

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
