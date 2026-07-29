#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COSMOS_REPO="https://github.com/NVIDIA/cosmos-framework"

NGC_NAMESPACE="nvcr.io/nvstaging/isaac-amr"
IMAGE_NAME="isaaclab_arena"
IMAGE_TAG="cosmos_server"
IMAGE_REF="${IMAGE_NAME}:${IMAGE_TAG}"
NGC_PATH="${NGC_NAMESPACE}/${IMAGE_REF}"

# Gated policy checkpoint baked into the image at build time (so the server needs no HF token at
# run time). Kept in sync with CosmosServerTaskCfg.checkpoint in osmo/tasks/cosmos_server_task.py.
CHECKPOINT="nvidia/Cosmos3-Edge-Policy-DROID"
BAKED_CHECKPOINT_PATH="/workspace/baked_checkpoint"

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
    echo "It is used once, at build time, to download the gated ${CHECKPOINT} checkpoint that is" >&2
    echo "baked into the image; the running server then needs no token." >&2
    exit 1
fi

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

PINNED_COMMIT="$(tr -d '[:space:]' < "${SCRIPT_DIR}/COSMOS_COMMIT")"
COSMOS_DIR="$TMPDIR/cosmos-framework"
echo "Cloning cosmos-framework at ${PINNED_COMMIT} ..."
# Partial clone: skip blob objects, git fetches them on demand at checkout.
# Cuts the one-time clone size on a large repo without changing what we end up with.
GIT_LFS_SKIP_SMUDGE=1 git clone --quiet --filter=blob:none "$COSMOS_REPO" "$COSMOS_DIR"
(cd "$COSMOS_DIR" && git checkout "$PINNED_COMMIT")

# Adapt upstream's Dockerfile into a self-contained policy-server image:
#   * --locked -> --frozen: a git sub-dependency (Megatron-LM) tracks a moving branch,
#     so --locked fails its up-to-date assertion; --frozen installs the pinned lockfile
#     resolution as-is (and never writes to the read-only lock bind-mount).
#   * retarget the dependency-group sync to the cu130-train + policy-server groups the
#     HuggingFace quickstart uses to run the server (upstream syncs cu130 + vllm).
sed -e 's/--locked/--frozen/' \
    -e 's|--group=\$(cat /root/.cuda-name) --group=vllm|--group=cu130-train --group=policy-server|' \
    "$COSMOS_DIR/Dockerfile" > "$TMPDIR/Dockerfile"

# Bake the source in so the entrypoint's editable install and the server module
# resolve without a runtime volume mount (upstream expects `.` mounted at /workspace).
echo "COPY . /workspace" >> "$TMPDIR/Dockerfile"

# Bake the gated policy checkpoint into the image so the running server needs no HuggingFace token.
# The token is supplied as a BuildKit secret -- mounted only for this RUN, never written to a layer
# -- and the download uses the image's own HuggingFace client from /workspace/.venv. ``--with
# hf_transfer`` adds the accelerated-download backend just for this step (the base env lacks it,
# and HF_HUB_ENABLE_HF_TRANSFER=1 errors without it) without touching the pinned lockfile.
cat >> "$TMPDIR/Dockerfile" <<DOCKERFILE
WORKDIR /workspace
RUN --mount=type=secret,id=hf_token \\
    HF_TOKEN="\$(cat /run/secrets/hf_token)" HF_HUB_ENABLE_HF_TRANSFER=1 \\
    uv run --frozen --with hf_transfer python -c "from huggingface_hub import snapshot_download; snapshot_download('${CHECKPOINT}', local_dir='${BAKED_CHECKPOINT_PATH}')"
DOCKERFILE

# At run time the server also pulls the gated *guardrail* checkpoint (nvidia/Cosmos-Guardrail1) via
# the framework's own checkpoint DB -- shared by the blocklist, face-blur, and video-safety filters.
# Pre-warm it into the image's HuggingFace cache at build time by calling that same download, so it
# uses the framework's pinned revision and cache path and the run-time fetch is a cache hit that
# needs no token. (Symbol lives in guardrail.common.core, confirmed by `grep -rn GUARDRAIL1_CHECKPOINT`.)
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
