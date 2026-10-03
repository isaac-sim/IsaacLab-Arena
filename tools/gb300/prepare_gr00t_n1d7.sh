#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

ARENA_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
GR00T_N1D7_ROOT=${GR00T_N1D7_ROOT:-$HOME/.cache/arena-gr00t-n17-src}
GR00T_N1D7_RUNTIME=${GR00T_N1D7_RUNTIME:-$HOME/.cache/arena-gr00t-n17-runtime}
GR00T_N1D7_REVISION=51d4c89f72fda44cbf77285c6a8114b52676b8a1
GR00T_PYTHON=${GR00T_PYTHON:-python3.12}
UV_COMMAND=${UV_COMMAND:-uv}

command -v "$UV_COMMAND" >/dev/null || { echo "Install uv before preparing the N1.7 runtime." >&2; exit 1; }
if [ ! -d "$GR00T_N1D7_ROOT" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone --filter=blob:none --no-checkout \
        https://github.com/NVIDIA/Isaac-GR00T.git "$GR00T_N1D7_ROOT"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$GR00T_N1D7_ROOT" checkout --detach "$GR00T_N1D7_REVISION"
fi
if [ "$(git -C "$GR00T_N1D7_ROOT" rev-parse HEAD)" != "$GR00T_N1D7_REVISION" ]; then
    echo "Expected Isaac-GR00T $GR00T_N1D7_REVISION at $GR00T_N1D7_ROOT; select a separate checkout." >&2
    exit 1
fi
if [ ! -x "$GR00T_N1D7_RUNTIME/bin/python" ]; then
    "$UV_COMMAND" venv --python "$GR00T_PYTHON" "$GR00T_N1D7_RUNTIME"
fi
# CUDA 13 wheels support the GB300; this environment is separate from Isaac Sim and N1.6.
"$UV_COMMAND" --no-config pip install --python "$GR00T_N1D7_RUNTIME/bin/python" \
    --index https://download.pytorch.org/whl/cu130 torch==2.9.0 torchvision==0.24.0
"$UV_COMMAND" --no-config pip install --python "$GR00T_N1D7_RUNTIME/bin/python" \
    -r "$ARENA_ROOT/tools/gb300/gr00t_n1d7_inference_requirements.txt"
"$UV_COMMAND" --no-config pip install --python "$GR00T_N1D7_RUNTIME/bin/python" \
    --no-deps -e "$GR00T_N1D7_ROOT"
