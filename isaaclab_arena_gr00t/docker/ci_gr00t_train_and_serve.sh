#!/usr/bin/env bash
# CI entrypoint for the GR00T sidecar. The CI image already includes the tuned
# checkpoint, so this only starts the remote policy server.
set -euxo pipefail

MODELS_DIR="${MODELS_DIR:-/workspace/pretrained_ckpts}"
CHECKPOINT="${CHECKPOINT:-${MODELS_DIR}/checkpoint-20000}"
SERVER_PORT="${SERVER_PORT:-5555}"

cd /workspace
nvidia-smi

if [ ! -d "${CHECKPOINT}" ]; then
  echo "Expected GR00T CI checkpoint not found at ${CHECKPOINT}" >&2
  echo "Contents of ${MODELS_DIR}:" >&2
  ls -la "${MODELS_DIR}" >&2 || true
  exit 1
fi

echo "Starting GR00T inference server with ${CHECKPOINT} on port ${SERVER_PORT}"
exec uv run python gr00t/eval/run_gr00t_server.py \
  --model_path="${CHECKPOINT}" \
  --embodiment_tag=NEW_EMBODIMENT \
  --host=0.0.0.0 \
  --port="${SERVER_PORT}"
