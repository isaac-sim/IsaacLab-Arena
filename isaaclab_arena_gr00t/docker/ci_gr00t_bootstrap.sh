#!/usr/bin/env bash
# Baked into the GR00T CI image at /workspace/ci_bootstrap.sh.
# Waits for the Arena workspace bind mount to populate after checkout, then
# executes the train-and-serve wrapper from the mounted repo so changes to the
# recipe stay PR-reviewable.
set -euo pipefail

ARENA_WORKSPACE="${ARENA_WORKSPACE:-/arena_workspace}"
TARGET="${ARENA_WORKSPACE}/isaaclab_arena_gr00t/docker/ci_gr00t_train_and_serve.sh"
TIMEOUT_SECONDS="${BOOTSTRAP_TIMEOUT_SECONDS:-600}"

echo "[bootstrap] waiting for ${TARGET} (timeout ${TIMEOUT_SECONDS}s)..."
deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
while [ ! -f "${TARGET}" ]; do
  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "[bootstrap] timed out waiting for ${TARGET}"
    echo "[bootstrap] contents of ${ARENA_WORKSPACE}:"
    ls -la "${ARENA_WORKSPACE}" || true
    exit 1
  fi
  sleep 5
done

echo "[bootstrap] found ${TARGET}, executing"
exec bash "${TARGET}"
