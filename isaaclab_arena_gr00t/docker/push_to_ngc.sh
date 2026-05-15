#!/bin/bash
set -e

TAG_NAME=latest
IMAGE_NAME=gr00t1_6_arena_ci
HF_TOKEN=${1:?Usage: $0 <HF_TOKEN>}
NGC_PATH=nvcr.io/nvidian/${IMAGE_NAME}:${TAG_NAME}

# Build the Docker image.

docker build --build-arg HF_TOKEN=${HF_TOKEN} -t $IMAGE_NAME . -f isaaclab_arena_gr00t/docker/Dockerfile.gr00t_1_6

# Remove any old containers (exited or running).
if [ "$(docker ps -a --quiet --filter name=$IMAGE_NAME)" ]; then
    docker rm -f $IMAGE_NAME > /dev/null
fi

# Tag and push the image to NGC.
echo "Pushing container to ${NGC_PATH}."
docker tag ${IMAGE_NAME} ${NGC_PATH}
docker push ${NGC_PATH}
echo "Pushing complete."
