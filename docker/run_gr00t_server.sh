#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# User-configurable defaults
# -------------------------

# Model directory on the host.
# By default, use $HOME/models, but this can be overridden
# by the MODELS_DIR environment variable or the -d / --models_dir flag.
MODELS_DIR="${MODELS_DIR:-$HOME/models}"

# Other parameters (can also be overridden via environment variables)
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
API_TOKEN="${API_TOKEN:-API_TOKEN_123}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
POLICY_TYPE="${POLICY_TYPE:-gr00t_closedloop}"
POLICY_CONFIG_YAML_PATH="${POLICY_CONFIG_YAML_PATH:-/workspace/isaaclab_arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml}"

# -------------------------
# Help message
# -------------------------
usage() {
  cat <<EOF
Usage: bash ./docker/run_gr00t_server.sh [options]

Description:
  By default, the script mounts \$HOME/models from the host to /models in the container.
  You can override this via the MODELS_DIR environment variable or the -d / --models_dir flag.

Options (all optional; environment variables with the same name take precedence):
  -d, --models_dir PATH               Model directory on the host. Default: ${MODELS_DIR}
  --host HOST                         Server host. Default: ${HOST}
  --port PORT                         Server port. Default: ${PORT}
  --api_token TOKEN                   API token for requests. Default: ${API_TOKEN}
  --timeout_ms MS                     Request timeout in milliseconds. Default: ${TIMEOUT_MS}
  --policy_type TYPE                  Policy type. Default: ${POLICY_TYPE}
  --policy_config_yaml_path PATH      Policy config YAML path. Default: ${POLICY_CONFIG_YAML_PATH}
  -h, --help                          Show this help message and exit.

Examples:
  # Use default \$HOME/models
  bash ./docker/run_gr00t_server.sh

  # Use a custom models directory and port
  bash ./docker/run_gr00t_server.sh -d /data/models --port 6000 --api_token MY_TOKEN

  # Use an environment variable to set the models directory
  MODELS_DIR=/data/models bash ./docker/run_gr00t_server.sh
EOF
}

# -------------------------
# CLI parsing
# -------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--models_dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --api_token)
      API_TOKEN="$2"
      shift 2
      ;;
    --timeout_ms)
      TIMEOUT_MS="$2"
      shift 2
      ;;
    --policy_type)
      POLICY_TYPE="$2"
      shift 2
      ;;
    --policy_config_yaml_path)
      POLICY_CONFIG_YAML_PATH="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

echo "Using MODELS_DIR=${MODELS_DIR}"
echo "Server config:"
echo "  HOST                    = ${HOST}"
echo "  PORT                    = ${PORT}"
echo "  API_TOKEN               = ${API_TOKEN}"
echo "  TIMEOUT_MS              = ${TIMEOUT_MS}"
echo "  POLICY_TYPE             = ${POLICY_TYPE}"
echo "  POLICY_CONFIG_YAML_PATH = ${POLICY_CONFIG_YAML_PATH}"

# -------------------------
# 1) Build the Docker image
# -------------------------
docker build \
  -f docker/Dockerfile.gr00t_server \
  -t gr00t_policy_server:latest \
  .

# -------------------------
# 2) Run the container
# -------------------------
docker run --rm \
  --gpus all \
  --net host \
  --name gr00t_policy_server_container \
  -v "${MODELS_DIR}":/models \
  gr00t_policy_server:latest \
  --host "${HOST}" \
  --port "${PORT}" \
  --api_token "${API_TOKEN}" \
  --timeout_ms "${TIMEOUT_MS}" \
  --policy_type "${POLICY_TYPE}" \
  --policy_config_yaml_path "${POLICY_CONFIG_YAML_PATH}"

