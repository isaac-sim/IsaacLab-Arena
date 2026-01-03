#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# User-configurable defaults
# -------------------------

# REQUIRED: host models directory (must be set by user)
if [[ -z "${MODELS_DIR:-}" ]]; then
  echo "ERROR: MODELS_DIR is not set."
  echo "Please export MODELS_DIR to your host models directory, e.g.:"
  echo "  export MODELS_DIR=/path/to/your/models"
  echo "Then run:"
  echo "  bash ./docker/run_gr00t_server.sh"
  exit 1
fi

# Optional overrides via environment variables or CLI flags
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5555}"
API_TOKEN="${API_TOKEN:-API_TOKEN_123}"
TIMEOUT_MS="${TIMEOUT_MS:-5000}"
POLICY_TYPE="${POLICY_TYPE:-gr00t_closedloop}"
POLICY_CONFIG_YAML_PATH="${POLICY_CONFIG_YAML_PATH:-/workspace/isaaclab_arena_gr00t/gr1_manip_gr00t_closedloop_config.yaml}"

# -------------------------
# CLI parsing (optional)
# -------------------------
usage() {
  cat <<EOF
Usage: MODELS_DIR=/path/to/models bash ./docker/run_gr00t_server.sh [options]

Options (all optional; environment variables with same name take precedence):
  --host HOST                          Default: ${HOST}
  --port PORT                          Default: ${PORT}
  --api_token TOKEN                    Default: ${API_TOKEN}
  --timeout_ms MS                      Default: ${TIMEOUT_MS}
  --policy_type TYPE                   Default: ${POLICY_TYPE}
  --policy_config_yaml_path PATH       Default: ${POLICY_CONFIG_YAML_PATH}

Examples:
  MODELS_DIR=/data/models bash ./docker/run_gr00t_server.sh
  MODELS_DIR=/data/models bash ./docker/run_gr00t_server.sh --port 6000 --api_token MY_TOKEN
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --api_token) API_TOKEN="$2"; shift 2 ;;
    --timeout_ms) TIMEOUT_MS="$2"; shift 2 ;;
    --policy_type) POLICY_TYPE="$2"; shift 2 ;;
    --policy_config_yaml_path) POLICY_CONFIG_YAML_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
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

