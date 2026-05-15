#!/usr/bin/env bash
# Compatibility wrapper for the currently published GR00T CI sidecar image.
# Its baked /workspace/ci_bootstrap.sh still looks for this path in the mounted
# repository. Keep this file until the sidecar image is rebuilt and rolled out.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${REPO_ROOT}/isaaclab_arena_gr00t/docker/ci_gr00t_train_and_serve.sh" "$@"
