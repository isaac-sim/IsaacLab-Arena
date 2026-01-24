#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# User-configurable defaults
# -------------------------

# Default mount directories on the host machine
DATASETS_DIR="${DATASETS_DIR:-$HOME/datasets}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
EVAL_DIR="${EVAL_DIR:-$HOME/eval}"

# Docker image name and tag for the GR00T policy server
DOCKER_IMAGE_NAME="${DOCKER_IMAGE_NAME:-gr00t_policy_server}"
DOCKER_VERSION_TAG="${DOCKER_VERSION_TAG:-latest}"

# Rebuild controls
FORCE_REBUILD="${FORCE_REBUILD:-false}"
NO_CACHE=""

# Server parameters (can also be overridden via environment variables)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
API_TOKEN="${API_TOKEN:-}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
POLICY_TYPE="${POLICY_TYPE:-gr00t_closedloop}"
POLICY_CONFIG_YAML_PATH="${POLICY_CONFIG_YAML_PATH:-/workspace/isaaclab_arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml}"

# -------------------------
# Help message
# -------------------------
usage() {
  script_name=$(basename "$0")
  cat <<EOF
Helper script to build and run the GR00T policy server Docker environment.

Usage:
  $script_name [options] [-- server-args...]

Options (Docker / paths; env vars with the same name take precedence):
  -v                      Verbose output (set -x).
  -d <datasets directory> Path to datasets on the host. Default: "$DATASETS_DIR".
  -m <models directory>   Path to models on the host. Default: "$MODELS_DIR".
  -e <eval directory>     Path to evaluation data on the host. Default: "$EVAL_DIR".
  -n <docker name>        Docker image name. Default: "$DOCKER_IMAGE_NAME".
  -r                      Force rebuilding of the Docker image.
  -R                      Force rebuilding of the Docker image, without cache.

Server-specific options (passed through to the policy server entrypoint):
  --host HOST
  --port PORT
  --api_token TOKEN
  --timeout_ms MS
  --policy_type TYPE
  --policy_config_yaml_path PATH

Examples:
  # Minimal: use defaults, just build & run server
  bash $script_name

  # Custom models directory and port
  bash $script_name -m /data/models --port 6000 --api_token MY_TOKEN

  # Custom image name, force rebuild, and datasets/eval mounts
  bash $script_name -n gr00t_server -r \\
    -d /data/datasets -m /data/models -e /data/eval \\
    --policy_type isaaclab_arena_gr00t.policy.gr00t_remote_policy.Gr00tRemoteServerSidePolicy \\
    --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_gr00t_closedloop_config.yaml
EOF
}

# -------------------------
# Parse docker/path options (short flags, like run_docker.sh)
# -------------------------
DOCKER_ARGS_DONE=false
SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  if [ "$DOCKER_ARGS_DONE" = false ]; then
    case "$1" in
      -v)
        set -x
        shift 1
        ;;
      -d)
        DATASETS_DIR="$2"
        shift 2
        ;;
      -m)
        MODELS_DIR="$2"
        shift 2
        ;;
      -e)
        EVAL_DIR="$2"
        shift 2
        ;;
      -n)
        DOCKER_IMAGE_NAME="$2"
        shift 2
        ;;
      -r)
        FORCE_REBUILD="true"
        shift 1
        ;;
      -R)
        FORCE_REBUILD="true"
        NO_CACHE="--no-cache"
        shift 1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --host|--port|--api_token|--timeout_ms|--policy_type|--policy_config_yaml_path)
        # From here on, treat everything as server args and stop parsing docker flags
        DOCKER_ARGS_DONE=true
        SERVER_ARGS+=("$1")
        shift 1
        ;;
      --*)
        # Unknown long option at docker level -> treat as server arg
        DOCKER_ARGS_DONE=true
        SERVER_ARGS+=("$1")
        shift 1
        ;;
      *)
        # Anything else -> treat as server arg
        DOCKER_ARGS_DONE=true
        SERVER_ARGS+=("$1")
        shift 1
        ;;
    esac
  else
    SERVER_ARGS+=("$1")
    shift 1
  fi
done

# If no server args were passed, use defaults
if [ ${#SERVER_ARGS[@]} -eq 0 ]; then
  SERVER_ARGS=(
    --host "${HOST}"
    --port "${PORT}"
    --api_token "${API_TOKEN}"
    --timeout_ms "${TIMEOUT_MS}"
    --policy_type "${POLICY_TYPE}"
    --policy_config_yaml_path "${POLICY_CONFIG_YAML_PATH}"
  )
fi

echo "Host paths:"
echo "  DATASETS_DIR = ${DATASETS_DIR}"
echo "  MODELS_DIR   = ${MODELS_DIR}"
echo "  EVAL_DIR     = ${EVAL_DIR}"
echo "Docker image:"
echo "  ${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}"
echo "Rebuild:"
echo "  FORCE_REBUILD = ${FORCE_REBUILD}, NO_CACHE = '${NO_CACHE}'"
echo "Server args:"
printf '  %q ' "${SERVER_ARGS[@]}"; echo

# -------------------------
# 1) Build the Docker image
# -------------------------

IMAGE_TAG_FULL="${DOCKER_IMAGE_NAME}:${DOCKER_VERSION_TAG}"

if docker images -q "${IMAGE_TAG_FULL}" > /dev/null 2>&1 && [ "${FORCE_REBUILD}" != "true" ]; then
  echo "Docker image ${IMAGE_TAG_FULL} already exists. Skipping rebuild."
  echo "Use -r or -R to force rebuilding the image."
else
  echo "Building Docker image ${IMAGE_TAG_FULL}..."
  docker build --pull \
    ${NO_CACHE} \
    -f docker/Dockerfile.gr00t_server \
    -t "${IMAGE_TAG_FULL}" \
    .
fi

# -------------------------
# 2) Run the container
# -------------------------

DOCKER_RUN_ARGS=(
  --rm
  --gpus all
  --net host
  --name gr00t_policy_server_container
  -v "${MODELS_DIR}":/models
)

# Only mount datasets / eval if the directories exist on host
if [ -d "${DATASETS_DIR}" ]; then
  DOCKER_RUN_ARGS+=(-v "${DATASETS_DIR}":/datasets)
fi

if [ -d "${EVAL_DIR}" ]; then
  DOCKER_RUN_ARGS+=(-v "${EVAL_DIR}":/eval)
fi

docker run "${DOCKER_RUN_ARGS[@]}" \
  "${IMAGE_TAG_FULL}" \
  "${SERVER_ARGS[@]}"
