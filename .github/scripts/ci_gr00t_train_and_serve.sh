#!/usr/bin/env bash
# CI variant of osmo/finetune.yaml's entry.sh: single-GPU, tiny step count,
# all tunable heads frozen. Purpose is plumbing-level smoke (train -> serve ->
# remote closed-loop eval), not task success. Mirrors the finetune fixture
# pattern from release/0.2.0's test_gr00t_closedloop_policy.py.
set -euxo pipefail

ARENA_WORKSPACE="${ARENA_WORKSPACE:-/arena_workspace}"
DATASET_PATH="${ARENA_WORKSPACE}/isaaclab_arena_gr00t/tests/test_data/test_g1_locomanip_lerobot"
MODALITY_CONFIG="/workspace/g1_locomanip/g1_sim_wbc_data_config.py"
BASE_MODEL_PATH="/workspace/pretrained_ckpts/GR00T-N1.6-3B"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/ci_finetune}"
MAX_STEPS="${MAX_STEPS:-10}"
SERVER_PORT="${SERVER_PORT:-5555}"

cd /workspace
nvidia-smi

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

CHECKPOINT="${OUTPUT_DIR}/checkpoint-${MAX_STEPS}"
[ -d "${CHECKPOINT}" ] || { echo "expected checkpoint not found at ${CHECKPOINT}"; ls -la "${OUTPUT_DIR}"; exit 1; }

echo "Starting GR00T inference server with ${CHECKPOINT} on port ${SERVER_PORT}"
exec uv run python gr00t/eval/run_gr00t_server.py \
  --model_path="${CHECKPOINT}" \
  --embodiment_tag=NEW_EMBODIMENT \
  --host=0.0.0.0 \
  --port="${SERVER_PORT}"
