#!/bin/bash
set -e

TAG_NAME=latest
IMAGE_NAME=gr00t_1_6_finetune
NGC_PATH=nvcr.io/nvidian/${IMAGE_NAME}:${TAG_NAME}

# Building docker

docker build -t $IMAGE_NAME . -f osmo/Dockerfile.gr00t_1_6

# Remove any old containers (exited or running).
if [ "$(docker ps -a --quiet --filter name=$IMAGE_NAME)" ]; then
    docker rm -f $IMAGE_NAME > /dev/null
fi

# Tag and push the image to NGC.
echo "Pushing container to ${NGC_PATH}."
docker tag ${IMAGE_NAME} ${NGC_PATH}
docker push ${NGC_PATH}
echo "Pushing complete."