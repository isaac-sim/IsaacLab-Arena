#!/bin/bash
set -e

TAG_NAME=latest
CONTAINER_ID=""
PUSH_TO_NGC=false

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

while getopts ":t:gph" OPTION; do
    case $OPTION in
        t)
            TAG_NAME=$OPTARG
            echo "Tag name is ${TAG_NAME}."
            ;;
        g)
            INSTALL_GROOT=true
            GROOT_DEPS_GROUP=base
            echo "INSTALL_GROOT is ${INSTALL_GROOT}."
            echo "GROOT_DEPS_GROUP is ${GROOT_DEPS_GROUP}."
            ;;
        p)
            PUSH_TO_NGC=true
            echo "PUSH_TO_NGC (build and push to ngc)."
            ;;
        h | *)
            echo "Helper script for pushing isaac_arena images to NGC."
            echo "Usage:"
            echo "- pushing to NGC:"
            echo "    push_to_ngc.sh -p -t <tag_name>"
            echo "- pushing to NGC with GR00T installation:"
            echo "    push_to_ngc.sh -p -t <tag_name> -g"
            echo "- see help message:"
            echo "    push_to_ngc.sh -h"
            echo ""
            echo "  -p Push the image to NGC."
            echo "  -t Tag name of the image."
            echo "  -h help (this output)"
            exit 0
            ;;
    esac
done

# Get the NGC path.
ISAAC_ARENA_IMAGE_NAME=isaac_arena
NGC_PATH=nvcr.io/nvstaging/isaac-amr/${ISAAC_ARENA_IMAGE_NAME}:${TAG_NAME}

# Build the image.
docker build --progress=plain --network=host \
  --build-arg INSTALL_GROOT=$INSTALL_GROOT \
  -t ${ISAAC_ARENA_IMAGE_NAME}:${TAG_NAME} \
  "${SCRIPT_DIR}/.." \
  -f "${SCRIPT_DIR}/Dockerfile.isaac_arena"

# Push if requested.
if [ "$PUSH_TO_NGC" = true ]; then

    # Tag and push the image to NGC.
    echo "Pushing container to ${NGC_PATH}."
    docker tag ${ISAAC_ARENA_IMAGE_NAME} ${NGC_PATH}
    docker push ${NGC_PATH}
    echo "Pushing complete."

else

    echo "Not pushing to NGC. Use -p to push to NGC."

fi
