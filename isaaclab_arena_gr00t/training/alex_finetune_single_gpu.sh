#!/usr/bin/env bash
# Fine-tune GR00T N1.6 on an Alex ability-hands LeRobot dataset (single GPU).
#
# Prerequisites (host, outside Arena Docker):
#   1. LeRobot dataset from convert_hdf5_to_lerobot.py + alex_open_microwave_config.yaml
#   2. Isaac-GR00T uv env: https://github.com/NVIDIA/Isaac-GR00T#installation-guide
#
# Usage:
#   export DATASET_PATH=/tmp/alex_demo_generated/lerobot
#   export OUTPUT_DIR=~/models/alex_open_microwave_finetune
#   bash isaaclab_arena_gr00t/training/alex_finetune_single_gpu.sh
#
# Run from Isaac-GR00T repo root, or set ISAAC_GR00T_DIR and ARENA_REPO paths below.

set -euo pipefail

ISAAC_GR00T_DIR="${ISAAC_GR00T_DIR:-$(cd "$(dirname "$0")/../../submodules/Isaac-GR00T" && pwd)}"
ARENA_REPO="${ARENA_REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"

DATASET_PATH="${DATASET_PATH:?Set DATASET_PATH to your LeRobot folder (contains meta/, data/, videos/)}"
OUTPUT_DIR="${OUTPUT_DIR:-${HOME}/models/alex_open_microwave_finetune}"
MODALITY_CONFIG="${MODALITY_CONFIG:-${ARENA_REPO}/isaaclab_arena_gr00t/embodiments/alex/alex_data_config.py}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-4}"

cd "${ISAAC_GR00T_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality-config-path "${MODALITY_CONFIG}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --max-steps "${MAX_STEPS}" \
  --num-gpus 1 \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 5 \
  --base-model-path "${BASE_MODEL_PATH}" \
  --no-tune-llm \
  --tune-visual \
  --tune-projector \
  --tune-diffusion-model \
  --dataloader-num-workers "${DATALOADER_WORKERS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
