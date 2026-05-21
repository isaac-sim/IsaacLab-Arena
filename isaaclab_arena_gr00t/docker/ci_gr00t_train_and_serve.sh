#!/usr/bin/env bash
# CI entrypoint for the GR00T e2e smoke test. It has two explicit modes:
#
# - bootstrap: runs inside the prebuilt gr00t image from the baked
#   /workspace/ci_gr00t_train_and_serve.sh copy. It waits for GitHub Actions to
#   mount and checkout the Arena repo, then execs this script from the mounted
#   repo so CI can pick up script changes without rebuilding the image first.
# - serve: runs from the mounted repo copy after bootstrap. It post-trains the
#   base GR00T model on the tiny CI dataset, then starts the remote policy
#   server from the resulting checkpoint.
set -euxo pipefail

CI_GR00T_ENTRYPOINT_MODE="${CI_GR00T_ENTRYPOINT_MODE:-serve}"
ARENA_WORKSPACE="${ARENA_WORKSPACE:-/arena_workspace}"
REPO_ENTRYPOINT="${ARENA_WORKSPACE}/isaaclab_arena_gr00t/docker/ci_gr00t_train_and_serve.sh"
BOOTSTRAP_TIMEOUT_SECONDS="${BOOTSTRAP_TIMEOUT_SECONDS:-600}"

if [ "${CI_GR00T_ENTRYPOINT_MODE}" = "bootstrap" ]; then
  echo "[bootstrap] waiting for ${REPO_ENTRYPOINT} (timeout ${BOOTSTRAP_TIMEOUT_SECONDS}s)..."
  deadline=$(( $(date +%s) + BOOTSTRAP_TIMEOUT_SECONDS ))
  # Looping until the repo's entrypoint is found
  while [ ! -f "${REPO_ENTRYPOINT}" ]; do
    if [ "$(date +%s)" -ge "${deadline}" ]; then
      echo "[bootstrap] timed out waiting for ${REPO_ENTRYPOINT}"
      echo "[bootstrap] contents of ${ARENA_WORKSPACE}:"
      ls -la "${ARENA_WORKSPACE}" || true
      exit 1
    fi
    sleep 5
  done

  # Switch to serve mode from bootstrap mode.
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
BASE_MODEL_PATH="${BASE_MODEL_PATH:-${MODELS_DIR}/GR00T-N1.6-3B}"
DATASET_PATH="${DATASET_PATH:-${ARENA_WORKSPACE}/isaaclab_arena_gr00t/tests/test_data/test_g1_locomanip_lerobot}"
MODALITY_CONFIG="${MODALITY_CONFIG:-/workspace/g1_locomanip/g1_sim_wbc_data_config.py}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ci_finetune}"
MAX_STEPS="${MAX_STEPS:-10}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoint-${MAX_STEPS}}"
SERVER_PORT="${SERVER_PORT:-5555}"
DATASET_READY_TIMEOUT_SECONDS="${DATASET_READY_TIMEOUT_SECONDS:-600}"
DATASET_READY_FILE="${DATASET_READY_FILE:-${DATASET_PATH}/videos/chunk-000/observation.images.ego_view/episode_000000.mp4}"

echo "Waiting for GR00T CI dataset media at ${DATASET_READY_FILE}"
deadline=$(( $(date +%s) + DATASET_READY_TIMEOUT_SECONDS ))
while true; do
  if [ -f "${DATASET_READY_FILE}" ] && ! grep -q "git-lfs.github.com/spec" "${DATASET_READY_FILE}"; then
    break
  fi

  if [ "$(date +%s)" -ge "${deadline}" ]; then
    echo "Timed out waiting for LFS-backed dataset media at ${DATASET_READY_FILE}" >&2
    if [ -f "${DATASET_READY_FILE}" ]; then
      echo "Current file size: $(wc -c < "${DATASET_READY_FILE}") bytes" >&2
      head -n 3 "${DATASET_READY_FILE}" >&2 || true
    else
      echo "File does not exist yet." >&2
    fi
    exit 1
  fi

  sleep 5
done

cd /workspace
nvidia-smi

if [ ! -d "${BASE_MODEL_PATH}" ]; then
  echo "Expected GR00T base model not found at ${BASE_MODEL_PATH}" >&2
  echo "Contents of ${MODELS_DIR}:" >&2
  ls -la "${MODELS_DIR}" >&2 || true
  exit 1
fi

echo "Post-training GR00T policy from ${BASE_MODEL_PATH}"
mkdir -p "${OUTPUT_DIR}"
uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path="${DATASET_PATH}" \
  --output-dir="${OUTPUT_DIR}" \
  --modality-config-path="${MODALITY_CONFIG}" \
  --global-batch-size=1 \
  --max-steps="${MAX_STEPS}" \
  --num-gpus=1 \
  --save-total-limit=2 \
  --save-steps="${MAX_STEPS}" \
  --base-model-path="${BASE_MODEL_PATH}" \
  --no-tune-llm \
  --no-tune-visual \
  --no-tune-projector \
  --no-tune-diffusion-model \
  --dataloader-num-workers=1 \
  --embodiment-tag=NEW_EMBODIMENT \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --no-use-wandb

if [ ! -d "${CHECKPOINT}" ]; then
  echo "Expected post-trained GR00T CI checkpoint not found at ${CHECKPOINT}" >&2
  echo "Contents of ${OUTPUT_DIR}:" >&2
  ls -la "${OUTPUT_DIR}" >&2 || true
  exit 1
fi

echo "Starting GR00T inference server with ${CHECKPOINT} on port ${SERVER_PORT}"
exec uv run python gr00t/eval/run_gr00t_server.py \
  --model_path="${CHECKPOINT}" \
  --embodiment_tag=NEW_EMBODIMENT \
  --host=0.0.0.0 \
  --port="${SERVER_PORT}"
