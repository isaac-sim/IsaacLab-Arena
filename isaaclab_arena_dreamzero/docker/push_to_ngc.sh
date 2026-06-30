#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

IMAGE_NAME=dreamzero_inference_server
TAG_NAME=latest
NGC_ORG=nvidian

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Builds the DreamZero inference server image and pushes it to NGC.
Requires docker login to nvcr.io first:
  docker login nvcr.io -u '\$oauthtoken' -p <YOUR_NGC_API_KEY>

Options:
  -t, --tag TAG         Image tag. Default: latest.
  -n, --image-name NAME Override image name. Default: dreamzero_inference_server.
  -h, --help            Show this help message.
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -t|--tag)
            TAG_NAME="${2:?Missing value for $1}"
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
            echo "Unexpected argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

NGC_PATH=nvcr.io/${NGC_ORG}/${IMAGE_NAME}:${TAG_NAME}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building image: ${IMAGE_NAME}:${TAG_NAME}"
docker build \
    -t "${IMAGE_NAME}:${TAG_NAME}" \
    -f "${SCRIPT_DIR}/Dockerfile" \
    "${SCRIPT_DIR}"

echo "Tagging as ${NGC_PATH}"
docker tag "${IMAGE_NAME}:${TAG_NAME}" "${NGC_PATH}"

echo "Pushing to ${NGC_PATH}"
docker push "${NGC_PATH}"

echo "Done. Update dreamzero_inference_server.yaml to use image: ${NGC_PATH}"
