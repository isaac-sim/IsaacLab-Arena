#!/usr/bin/env bash
set -euo pipefail

TAG_NAME=latest
MODELS_DIR=/workspace/pretrained_ckpts
IMAGE_NAME=gr00t1_6_arena_ci
HF_TOKEN=

usage() {
    cat <<EOF
Usage: $0 <HF_TOKEN>

Builds and pushes the GR00T image to NGC.

Options:
  --models-dir DIR      Model directory inside the image. Default: /workspace/pretrained_ckpts.
  -t, --tag TAG         Image tag. Default: latest.
  -n, --image-name NAME Override the NGC image name. Default: gr00t1_6_arena_ci.
  -h, --help            Show this help message.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -t|--tag)
            TAG_NAME="${2:?Missing value for $1}"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="${2:?Missing value for $1}"
            shift 2
            ;;
        -n|--image-name)
            IMAGE_NAME="${2:?Missing value for $1}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            echo "Unexpected option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            if [ -z "${HF_TOKEN}" ]; then
                HF_TOKEN="$1"
                shift
            else
                echo "Unexpected argument: $1" >&2
                usage >&2
                exit 1
            fi
            ;;
    esac
done

if [ -z "${HF_TOKEN}" ]; then
    echo "Missing HF_TOKEN." >&2
    usage >&2
    exit 1
fi

NGC_PATH=nvcr.io/nvidian/${IMAGE_NAME}:${TAG_NAME}

# Build the Docker image.

docker build \
    --build-arg HF_TOKEN="${HF_TOKEN}" \
    --build-arg MODELS_DIR="${MODELS_DIR}" \
    -t "${IMAGE_NAME}" \
    . \
    -f isaaclab_arena_gr00t/docker/Dockerfile.gr00t_1_6

# Remove any old containers (exited or running).
if [ "$(docker ps -a --quiet --filter name="${IMAGE_NAME}")" ]; then
    docker rm -f "${IMAGE_NAME}" > /dev/null
fi

# Tag and push the image to NGC.
echo "Pushing container to ${NGC_PATH}."
docker tag "${IMAGE_NAME}" "${NGC_PATH}"
docker push "${NGC_PATH}"
echo "Pushing complete."
