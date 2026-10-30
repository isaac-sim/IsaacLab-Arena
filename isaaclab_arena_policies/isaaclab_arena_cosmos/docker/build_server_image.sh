#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

# NOTE(alexmillane, 2026-07-31): This script is a modified version of the original
# Cosmos3-Edge-Policy-DROID image build script that ships with the Cosmos framework.
# We maintain our own in order to fix some issues with the original script, configure
# the image for policy-server, and to embed the models within the image.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_REPO="https://github.com/NVIDIA/cosmos-framework"

NGC_NAMESPACE="nvcr.io/nvstaging/isaac-amr"
IMAGE_NAME="isaaclab_arena"
IMAGE_TAG="cosmos_server"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
NGC_PATH="${NGC_NAMESPACE}/${IMAGE_REF}"

# Policy checkpoint baked into the image at build time (so the server needs no HF token at run
# time). Kept in sync with CosmosServerTaskCfg.checkpoint in osmo/tasks/cosmos_server_task.py.
CHECKPOINT="nvidia/Cosmos3-Edge-Policy-DROID"
CHECKPOINT_LOCAL_PATH="/workspace/baked_checkpoint"

PUSH_TO_NGC=false
NO_CACHE=""

print_help() {
    cat <<EOF
Build (and optionally push) the Cosmos3-Edge-Policy inference-server image.

Usage:
  $(basename "$0") [-p] [-R]

Options:
  -p              Push the built image to NGC (${NGC_PATH}).
  -R              Do not use the Docker build cache.
  -h              Show this help and exit.

Pushing assumes you have already run \`docker login nvcr.io\`.
EOF
}

while getopts ":pRh" opt; do
    case "$opt" in
        p) PUSH_TO_NGC=true ;;
        R) NO_CACHE="--no-cache" ;;
        h) print_help; exit 0 ;;
        \?) echo "unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
    esac
done

if [ -z "${HF_TOKEN:-}" ]; then
    echo "HF_TOKEN is not set. Export a token from https://huggingface.co/settings/tokens." >&2
    echo "It is used once, at build time, to download the ${CHECKPOINT} checkpoint that is" >&2
    echo "baked into the image; the running server then needs no token." >&2
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PINNED_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/COSMOS_COMMIT")"
COSMOS_DIR="$TMPDIR/cosmos-framework"
echo "Cloning cosmos-framework at ${PINNED_COMMIT} ..."
# Clone cosmos without blob objects.
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none "$COSMOS_REPO" "$COSMOS_DIR"
(cd "$COSMOS_DIR" && git checkout "$PINNED_COMMIT")

# Adapt upstream's Dockerfile into a self-contained policy-server image:
#   * --locked -> --frozen: a git sub-dependency (Megatron-LM) tracks a moving branch,
#     so --locked fails its up-to-date assertion; --frozen installs the pinned lockfile
#     resolution as-is (and never writes to the read-only lock bind-mount).
#   * Switch the dependency-group from vllm to the cu130-train + policy-server groups
sed -e 's/--locked/--frozen/' \
    -e 's|--group=\$(cat /root/.cuda-name) --group=vllm|--group=cu130-train --group=policy-server|' \
    "$COSMOS_DIR/Dockerfile" > "$TMPDIR/Dockerfile"

# Copy the source into the image, so the server can run without mounting anything.
echo "COPY . /workspace" >> "$TMPDIR/Dockerfile"

# Download the policy checkpoint and bake into the image.
cat >> "$TMPDIR/Dockerfile" <<DOCKERFILE
WORKDIR /workspace
RUN --mount=type=secret,id=hf_token \\
    HF_TOKEN="\$(cat /run/secrets/hf_token)" HF_HUB_ENABLE_HF_TRANSFER=1 \\
    uv run --frozen --with hf_transfer python -c "from huggingface_hub import snapshot_download; snapshot_download('${CHECKPOINT}', local_dir='${CHECKPOINT_LOCAL_PATH}')"
DOCKERFILE

# Download the guardrail checkpoint, and bake it into the image.
cat >> "$TMPDIR/Dockerfile" <<DOCKERFILE
RUN --mount=type=secret,id=hf_token \\
    HF_TOKEN="\$(cat /run/secrets/hf_token)" \\
    uv run --frozen python -c "from cosmos_framework.auxiliary.guardrail.common.core import GUARDRAIL1_CHECKPOINT; GUARDRAIL1_CHECKPOINT.download()"
DOCKERFILE

echo "Building ${IMAGE_REF} (baking ${CHECKPOINT} + guardrail into the image)"
DOCKER_BUILDKIT=1 docker build \
    --network=host \
    --secret id=hf_token,env=HF_TOKEN \
    $NO_CACHE \
    -f "$TMPDIR/Dockerfile" \
    -t "${IMAGE_REF}" \
    "$COSMOS_DIR"

if [ "$PUSH_TO_NGC" = true ]; then
    echo "Pushing container to ${NGC_PATH}."
    docker tag "${IMAGE_REF}" "${NGC_PATH}"
    docker push "${NGC_PATH}"
    echo "Pushing complete."
fi
