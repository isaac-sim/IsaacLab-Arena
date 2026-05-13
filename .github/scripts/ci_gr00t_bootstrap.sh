#!/usr/bin/env bash
# Baked into the gr00t CI image at /workspace/ci_bootstrap.sh.
# Waits for the arena workspace bind-mount to populate (i.e. the job's
# checkout step has run), then execs the real train+serve wrapper from
# the arena repo so changes to the recipe stay PR-reviewable.
set -euo pipefail

ARENA_WORKSPACE="${ARENA_WORKSPACE:-/arena_workspace}"
TARGET="${ARENA_WORKSPACE}/.github/scripts/ci_gr00t_train_and_serve.sh"
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

echo "[bootstrap] found ${TARGET}, exec'ing"
exec bash "${TARGET}"
