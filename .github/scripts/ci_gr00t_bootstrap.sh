#!/usr/bin/env bash
# Compatibility wrapper for already-published GR00T CI images and older local
# build commands. The maintained script lives with the GR00T Docker assets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

exec bash "${REPO_ROOT}/isaaclab_arena_gr00t/docker/ci_gr00t_bootstrap.sh" "$@"
