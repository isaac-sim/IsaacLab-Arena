#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# User-configurable defaults
# -------------------------

# Default mount directories on the host machine
DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
EVAL_DIR="${EVAL_DIR:-$HOME/eval}"

# Docker image name and tag for the VLN policy server
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-vln_policy_server}"
DOCKER_VERSION_TAG="${DOCKER_VERSION_TAG:-latest}"

# Rebuild controls
FORCE_REBUILD="${FORCE_REBUILD:-false}"
NO_CACHE=""

# Server parameters (can also be overridden via environment variables)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
API_TOKEN="${API_TOKEN:-}"
TIMEOUT_MS="${TIMEOUT_MS:-15000}"
POLICY_TYPE="${POLICY_TYPE:-isaaclab_arena_navila.navila_server_policy.NaVilaServerPolicy}"

# GPU selection for docker --gpus (can also be overridden via environment variables)
# Examples:
#   all           -> use all GPUs
#   1             -> use 1 GPU (count)
#   "device=0"    -> use GPU 0
#   "device=0,1"  -> use GPU 0 and 1
GPUS="${GPUS:-all}"

# -------------------------
# Help message
# -------------------------
usage() {
  script_name=$(basename "$0")
  cat <<EOF
Helper script to build and run the VLN policy server Docker environment.

Usage:
  $script_name [options] [-- server-args...]

Options (Docker / paths; env vars with the same name take precedence):
  -v                      Verbose output (set -x).
  -d <datasets directory> Path to datasets on the host. Default: "$DATASETS_DIR".
  -m <models directory>   Path to models on the host. Default: "$MODELS_DIR".
  -e <eval directory>     Path to evaluation data on the host. Default: "$EVAL_DIR".
  -n <docker name>        Docker image name. Default: "$DOCKER_IMAGE_NAME".
  -g <gpus>               GPU selection for docker --gpus. Default: "all".
                          Examples: "all", "1", "device=0", "device=0,1".
  -r                      Force rebuilding of the Docker image.
  -R                      Force rebuilding of the Docker image, without cache.

Server-specific options (passed through to the policy server entrypoint):
  --host HOST
  --port PORT
  --api_token TOKEN
  --timeout_ms MS
  --policy_type TYPE
  --model_path PATH
  --num_video_frames N
  --conv_mode MODE

Examples:
  # Minimal: use all defaults (model at default path, all GPUs, port 5555)
  bash $script_name

  # Custom model path and single GPU
  bash $script_name -m /path/to/navila-checkpoint -g "device=0"

  # Custom port
  bash $script_name --port 6000
EOF
}

# -------------------------
# Parse all options
# -------------------------
# Server parameters that can be overridden individually via --flags.
# Unset values will be filled with defaults after parsing.
MODEL_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -v)               set -x;                   shift 1 ;;
    -d)               DATASETS_DIR="$2";        shift 2 ;;
    -m)               MODELS_DIR="$2";          shift 2 ;;
    -e)               EVAL_DIR="$2";            shift 2 ;;
    -n)               DOCKER_IMAGE_NAME="$2";   shift 2 ;;
    -g)               GPUS="$2";                shift 2 ;;
    -r)               FORCE_REBUILD="true";     shift 1 ;;
    -R)               FORCE_REBUILD="true"; NO_CACHE="--no-cache"; shift 1 ;;
    -h|--help)        usage; exit 0 ;;
    --host)           HOST="$2";                shift 2 ;;
    --port)           PORT="$2";                shift 2 ;;
    --api_token)      API_TOKEN="$2";           shift 2 ;;
    --timeout_ms)     TIMEOUT_MS="$2";          shift 2 ;;
    --policy_type)    POLICY_TYPE="$2";         shift 2 ;;
    --model_path)     MODEL_PATH="$2";          shift 2 ;;
    --num_video_frames) NUM_VIDEO_FRAMES="$2";  shift 2 ;;
    --conv_mode)      CONV_MODE="$2";           shift 2 ;;
    --policy_device)  POLICY_DEVICE="$2";       shift 2 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# Build server args: always include all parameters, using defaults for
# anything the user did not override.
SERVER_ARGS=(
  --host "${HOST}"
  --port "${PORT}"
  --timeout_ms "${TIMEOUT_MS}"
  --policy_type "${POLICY_TYPE}"
  --model_path "${MODEL_PATH:-/models}"
)
[ -n "${API_TOKEN}" ]       && SERVER_ARGS+=(--api_token "${API_TOKEN}")
[ -n "${NUM_VIDEO_FRAMES+x}" ] && SERVER_ARGS+=(--num_video_frames "${NUM_VIDEO_FRAMES}")
[ -n "${CONV_MODE+x}" ]    && SERVER_ARGS+=(--conv_mode "${CONV_MODE}")
[ -n "${POLICY_DEVICE+x}" ] && SERVER_ARGS+=(--policy_device "${POLICY_DEVICE}")

echo "Host paths:"
echo "  DATASETS_DIR = ${DATASETS_DIR}"
echo "  MODELS_DIR   = ${MODELS_DIR}"
echo "  EVAL_DIR     = ${EVAL_DIR}"
echo "Docker image:"
echo "  ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}"
echo "GPU:"
echo "  --gpus ${GPUS}"
echo "Rebuild:"
echo "  FORCE_REBUILD = ${FORCE_REBUILD}, NO_CACHE = '${NO_CACHE}'"
echo "Server args:"
printf '  %q ' "${SERVER_ARGS[@]}"; echo

# -------------------------
# 1) Build the Docker image
# -------------------------

IMAGE_TAG_FULL="${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}"

SHOULD_BUILD=false

if [ "${FORCE_REBUILD}" = "true" ]; then
  SHOULD_BUILD=true
else
  if [ -z "$(docker images -q "${IMAGE_TAG_FULL}")" ]; then
    SHOULD_BUILD=true
  fi
fi

if [ "${SHOULD_BUILD}" = "true" ]; then
  echo "Building Docker image ${IMAGE_TAG_FULL}..."
  # Use existing image layers as cache source (BuildKit may GC intermediate
  # layer cache, but the final image layers can still be reused).
  CACHE_FROM_ARGS=""
  if [ -n "$(docker images -q "${IMAGE_TAG_FULL}" 2>/dev/null)" ]; then
    CACHE_FROM_ARGS="--cache-from ${IMAGE_TAG_FULL}"
  fi
  docker build \
    ${NO_CACHE} \
    ${CACHE_FROM_ARGS} \
    --network host \
    -f docker/Dockerfile.vln_server \
    -t "${IMAGE_TAG_FULL}" \
    .
else
  echo "Docker image ${IMAGE_TAG_FULL} already exists. Skipping rebuild."
  echo "Use -r or -R to force rebuilding the image."
fi

# -------------------------
# 2) Run the container
# -------------------------

DOCKER_RUN_ARGS=(
  --rm
  --gpus "${GPUS}"
  --net host
  --name vln_policy_server_container
  -v "${MODELS_DIR}":/models
)

if [ -d "${DATASETS_DIR}" ]; then
  DOCKER_RUN_ARGS+=(-v "${DATASETS_DIR}":/datasets)
fi

if [ -d "${EVAL_DIR}" ]; then
  DOCKER_RUN_ARGS+=(-v "${EVAL_DIR}":/eval)
fi

docker run "${DOCKER_RUN_ARGS[@]}" \
  "${IMAGE_TAG_FULL}" \
  "${SERVER_ARGS[@]}"
