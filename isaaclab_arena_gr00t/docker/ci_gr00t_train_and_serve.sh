#!/usr/bin/env bash
# CI entrypoint for the GR00T e2e smoke test. It has two explicit modes:
#
# - bootstrap: runs inside the prebuilt gr00t image as /workspace/ci_bootstrap.sh.
#   It waits for GitHub Actions to mount and checkout the Arena repo, then execs
#   this script from the mounted repo so CI can pick up script changes without
#   rebuilding the image first.
# - serve: runs from the mounted repo copy after bootstrap. It starts the GR00T
#   remote policy server from the checkpoint already baked into the image.
set -euxo pipefail

CI_GR00T_ENTRYPOINT_MODE="${CI_GR00T_ENTRYPOINT_MODE:-serve}"
ARENA_WORKSPACE="${ARENA_WORKSPACE:-/arena_workspace}"
REPO_ENTRYPOINT="${ARENA_WORKSPACE}/isaaclab_arena_gr00t/docker/ci_gr00t_train_and_serve.sh"
BOOTSTRAP_TIMEOUT_SECONDS="${BOOTSTRAP_TIMEOUT_SECONDS:-600}"

if [ "${CI_GR00T_ENTRYPOINT_MODE}" = "bootstrap" ]; then
  echo "[bootstrap] waiting for ${REPO_ENTRYPOINT} (timeout ${BOOTSTRAP_TIMEOUT_SECONDS}s)..."
  deadline=$(( $(date +%s) + BOOTSTRAP_TIMEOUT_SECONDS ))
  while [ ! -f "${REPO_ENTRYPOINT}" ]; do
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      echo "[bootstrap] timed out waiting for ${REPO_ENTRYPOINT}"
      echo "[bootstrap] contents of ${ARENA_WORKSPACE}:"
      ls -la "${ARENA_WORKSPACE}" || true
      exit 1
    fi
    sleep 5
  done

  echo "[bootstrap] found ${REPO_ENTRYPOINT}, switching to serve mode"
  # The image bakes CI_GR00T_ENTRYPOINT_MODE=bootstrap so the sidecar can wait
  # for the mounted checkout. Override it for the repo copy; otherwise the repo
  # script would inherit bootstrap mode and exec itself instead of starting the
  # server.
  exec env CI_GR00T_ENTRYPOINT_MODE=serve bash "${REPO_ENTRYPOINT}" "$@"
fi

if [ "${CI_GR00T_ENTRYPOINT_MODE}" != "serve" ]; then
  echo "Unsupported CI_GR00T_ENTRYPOINT_MODE=${CI_GR00T_ENTRYPOINT_MODE}. Expected 'bootstrap' or 'serve'." >&2
  exit 1
fi

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
