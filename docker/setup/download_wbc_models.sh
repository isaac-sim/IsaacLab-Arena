#!/bin/bash
set -euo pipefail

# Script to download external WBC policy models.
# This script is called from the Dockerfile or can be run manually.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/isaaclab_arena_g1/g1_whole_body_controller/wbc_policy/models"

# --- WBC-AGILE e2e velocity policy for G1 ---
AGILE_MODEL_DIR="${MODELS_DIR}/agile"
AGILE_MODEL_PATH="${AGILE_MODEL_DIR}/unitree_g1_velocity_e2e.onnx"
AGILE_MODEL_URL="https://github.com/nvidia-isaac/WBC-AGILE/raw/main/agile/data/policy/velocity_g1/unitree_g1_velocity_e2e.onnx"
AGILE_MODEL_SHA256="8995f2462ba2d0d83afe08905148f6373990d50018610663a539225d268ef33b"

download_model() {
    local url="$1"
    local dest="$2"
    local expected_sha256="$3"

    if [ -f "$dest" ]; then
        echo "Model already exists: $dest"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    echo "Downloading $(basename "$dest") from ${url} ..."
    curl -L -o "$dest" "$url"

    actual_sha256=$(sha256sum "$dest" | awk '{print $1}')
    if [ "$actual_sha256" != "$expected_sha256" ]; then
        echo "ERROR: SHA256 mismatch for $dest"
        echo "  expected: $expected_sha256"
        echo "  actual:   $actual_sha256"
        rm -f "$dest"
        return 1
    fi

    echo "Downloaded and verified: $dest"
}

download_model "$AGILE_MODEL_URL" "$AGILE_MODEL_PATH" "$AGILE_MODEL_SHA256"

echo "All WBC models downloaded successfully."
