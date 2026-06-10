#!/usr/bin/env bash
# Download dataset -> fine-tune GR00T N1.6 -> upload checkpoint to HuggingFace.
#
# Required env:
#   HF_TOKEN          HuggingFace token with write access (unless SKIP_UPLOAD=1)
# Optional env (defaults shown):
#   HF_DATASET_ID     H2Ozone/alex_microwave        dataset repo to download
#   HF_MODEL_REPO     H2Ozone/alex_open_microwave_gr00t   model repo to upload to
#   SKIP_UPLOAD       0     set 1 to train without uploading
#   UPLOAD_OPTIMIZER_STATE  0     set 1 to also upload optimizer/scheduler/rng state
#   GLOBAL_BATCH_SIZE 8
#   GRAD_ACCUM_STEPS  1
#   MAX_STEPS         30000
#   SAVE_STEPS        5000
#   NUM_GPUS          1
#   DATALOADER_WORKERS 2
#   LOW_VRAM          0     set 1 for <=16GB GPUs: diffusion head only, batch 2 + accum
#   OUTPUT_DIR        /checkpoints  (mount a volume here to survive restarts; training
#                                    auto-resumes from the last checkpoint it finds)

set -euo pipefail

HF_DATASET_ID="${HF_DATASET_ID:-H2Ozone/alex_microwave}"
HF_MODEL_REPO="${HF_MODEL_REPO:-H2Ozone/alex_open_microwave_gr00t}"
SKIP_UPLOAD="${SKIP_UPLOAD:-0}"
# Lives under /cache so the dataset download persists with the same volume/bind
# as the HF model cache (and lands on writable storage on clusters).
DATASET_PATH="${DATASET_PATH:-/cache/dataset/lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-/checkpoints}"
MODALITY_CONFIG=/workspace/IsaacLab-Arena/isaaclab_arena_gr00t/embodiments/alex/alex_data_config.py
BASE_MODEL_PATH="${BASE_MODEL_PATH:-nvidia/GR00T-N1.6-3B}"

MAX_STEPS="${MAX_STEPS:-30000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
NUM_GPUS="${NUM_GPUS:-1}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-2}"
LOW_VRAM="${LOW_VRAM:-0}"

if [[ "${LOW_VRAM}" == "1" ]]; then
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
  TUNE_FLAGS=(--no-tune-llm --no-tune-visual --no-tune-projector --tune-diffusion-model)
else
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
  TUNE_FLAGS=(--no-tune-llm --tune-visual --tune-projector --tune-diffusion-model)
fi

cd /workspace/Isaac-GR00T

echo "=== GPU check ==="
nvidia-smi

# Fail fast on a bad/missing token before hours of training.
if [[ "${SKIP_UPLOAD}" != "1" ]]; then
  : "${HF_TOKEN:?Set HF_TOKEN (write access) or SKIP_UPLOAD=1}"
  echo "=== Verifying HuggingFace token and creating ${HF_MODEL_REPO} ==="
  uv run python /workspace/upload_to_hf.py --verify-only --repo-id "${HF_MODEL_REPO}"
fi

echo "=== Downloading dataset ${HF_DATASET_ID} -> ${DATASET_PATH} ==="
uv run python - <<EOF
from huggingface_hub import snapshot_download
snapshot_download(repo_id="${HF_DATASET_ID}", repo_type="dataset", local_dir="${DATASET_PATH}")
EOF

for sub in meta data videos; do
  [[ -d "${DATASET_PATH}/${sub}" ]] || { echo "Missing ${DATASET_PATH}/${sub} — not a LeRobot dataset"; exit 1; }
done

mkdir -p "${OUTPUT_DIR}"

echo "=== Fine-tuning (output: ${OUTPUT_DIR}) ==="
uv run python gr00t/experiment/launch_finetune.py \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --modality-config-path "${MODALITY_CONFIG}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM_STEPS}" \
  --max-steps "${MAX_STEPS}" \
  --num-gpus "${NUM_GPUS}" \
  --save-steps "${SAVE_STEPS}" \
  --save-total-limit 5 \
  --base-model-path "${BASE_MODEL_PATH}" \
  "${TUNE_FLAGS[@]}" \
  --dataloader-num-workers "${DATALOADER_WORKERS}" \
  --embodiment-tag NEW_EMBODIMENT \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08

if [[ "${SKIP_UPLOAD}" != "1" ]]; then
  echo "=== Uploading latest checkpoint to ${HF_MODEL_REPO} ==="
  UPLOAD_ARGS=(--repo-id "${HF_MODEL_REPO}" --output-dir "${OUTPUT_DIR}")
  [[ "${UPLOAD_OPTIMIZER_STATE:-0}" == "1" ]] && UPLOAD_ARGS+=(--include-optimizer-state)
  uv run python /workspace/upload_to_hf.py "${UPLOAD_ARGS[@]}"
fi

echo "=== Done ==="
