#!/bin/bash
set -e

ISAAC_ARENA_IMAGE_NAME='isaac_arena'
TAG_NAME=latest
CONTAINER_ID=""
PUSH_TO_NGC=false
INSTALL_GROOT="false"
WORKDIR="/workspaces/isaac_arena"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while getopts ":t:gn:G:vn:pn:Rn:hn:" OPTION; do
    case $OPTION in
        t)
            TAG_NAME=$OPTARG
            echo "Tag name is ${TAG_NAME}."
            ;;
        g)
            INSTALL_GROOT="true"
            GROOT_DEPS_GROUP="base"
            echo "INSTALL_GROOT is ${INSTALL_GROOT}."
            echo "GROOT_DEPS_GROUP is ${GROOT_DEPS_GROUP}."
            ;;
        G)
            GROOT_DEPS_GROUP=${OPTARG}
            INSTALL_GROOT="true"
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
            echo "Helper script for pushing Isaac Arena docker image to NGC."
            echo ""
            echo "Usage:"
            echo "  ${script_name} [options]"
            echo ""
            echo "Examples:"
            echo "- Build without cache and push to NGC:"
            echo "    ${script_name} -R -p -t <tag_name>"
            echo "- Build without cache and push to NGC with GR00T installation of "base" dependencies:"
            echo "    ${script_name} -R -p -t <tag_name> -g"
            echo "- See help message:"
            echo "    ${script_name} -h"
            echo ""
            echo "Options:"
            echo "  -p - Push the image to NGC."
            echo "  -t - Tag name of the image."
            echo "  -g - Install GR00T with base dependencies."
            echo "  -G <group> - Install GR00T with dependency group <group>."
            echo '               Available groups: "base", "dev", "orin", "thor", "deploy".'
            echo '  -R - Do not use cache when building the image.'
            echo "  -v - Verbose output."
            echo "  -h - Help (this output)"
            exit 0
            ;;
    esac
done

# Get the NGC path.
DOCKER_IMAGE_NAME=${ISAAC_ARENA_IMAGE_NAME}:${TAG_NAME}
NGC_PATH=nvcr.io/nvstaging/isaac-amr/${DOCKER_IMAGE_NAME}

# Build the image.
docker build --pull \
    $NO_CACHE \
    --build-arg WORKDIR="${WORKDIR}" \
    --build-arg INSTALL_GROOT=$INSTALL_GROOT \
    --build-arg GROOT_DEPS_GROUP=$GROOT_DEPS_GROUP \
    -t ${DOCKER_IMAGE_NAME} \
    --file $SCRIPT_DIR/Dockerfile.isaac_arena \
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
