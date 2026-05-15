#!/usr/bin/env bash
set -euo pipefail

TAG_NAME=latest
BUILD_MODE=ci
MODELS_DIR=/workspace/pretrained_ckpts
IMAGE_NAME=
HF_TOKEN=

usage() {
    cat <<EOF
Usage: $0 [ci|training] <HF_TOKEN>
       $0 --mode <ci|training> <HF_TOKEN>

Builds and pushes the selected GR00T image flavor to NGC.

Modes:
  ci        Download the tuned Arena G1 loco-manipulation checkpoint.
  training  Download the GR00T-N1.6 base model for fine-tuning.

Options:
  -m, --mode MODE       Image flavor to build: ci or training. Default: ci.
  --models-dir DIR      Model directory inside the image. Default: /workspace/pretrained_ckpts.
  -t, --tag TAG         Image tag. Default: latest.
  -n, --image-name NAME Override the NGC image name.
  -h, --help            Show this help message.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        ci|training)
            BUILD_MODE="$1"
            shift
            ;;
        -m|--mode)
            BUILD_MODE="${2:?Missing value for $1}"
            shift 2
            ;;
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

if [ "${BUILD_MODE}" != "ci" ] && [ "${BUILD_MODE}" != "training" ]; then
    echo "Unsupported mode: ${BUILD_MODE}. Expected 'ci' or 'training'." >&2
    usage >&2
    exit 1
fi

if [ -z "${HF_TOKEN}" ]; then
    echo "Missing HF_TOKEN." >&2
    usage >&2
    exit 1
fi

if [ -z "${IMAGE_NAME}" ]; then
    IMAGE_NAME="gr00t1_6_arena_${BUILD_MODE}"
fi

NGC_PATH=nvcr.io/nvidian/${IMAGE_NAME}:${TAG_NAME}

# Build the Docker image.

docker build \
    --build-arg HF_TOKEN="${HF_TOKEN}" \
    --build-arg BUILD_MODE="${BUILD_MODE}" \
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
