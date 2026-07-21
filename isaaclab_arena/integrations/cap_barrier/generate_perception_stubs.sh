#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Generate the CAP perception gRPC Python stubs for the Arena barrier producer.
#
# The frozen contract lives in the ROS module worktree:
#   cap_perception_bridge/proto/cap_perception.proto
# The Arena pinned .venv ships grpcio + protobuf at runtime but intentionally does
# NOT carry grpcio-tools, and we must not mutate that venv or uv.lock. This script
# runs protoc in an ephemeral uv overlay (grpcio-tools pinned to Arena's grpcio
# version) and writes the stubs into a git-ignored _generated/ tree. It also
# rewrites the flat grpc import to a package-relative one so the producer can
# import the stubs as a normal subpackage.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROTO="${HERE}/../../../../../workspaces/isaac-cap-module/ros_ws/src/isaac_ros_cap/cap_perception_bridge/proto/cap_perception.proto"
PROTO="${CAP_PERCEPTION_PROTO:-${DEFAULT_PROTO}}"
GRPCIO_TOOLS_VERSION="${CAP_GRPCIO_TOOLS_VERSION:-1.82.1}"
OUT_DIR="${HERE}/_generated/cap_perception_proto"

if [[ ! -f "${PROTO}" ]]; then
  echo "error: proto not found at ${PROTO}" >&2
  echo "set CAP_PERCEPTION_PROTO to the frozen cap_perception.proto path" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT
cp "${PROTO}" "${STAGE}/cap_perception.proto"

mkdir -p "${OUT_DIR}"
uv run --no-project --python 3.11 --with "grpcio-tools==${GRPCIO_TOOLS_VERSION}" -- \
  python -m grpc_tools.protoc \
  -I"${STAGE}" \
  --python_out="${OUT_DIR}" \
  --grpc_python_out="${OUT_DIR}" \
  "${STAGE}/cap_perception.proto"

# Make the package self-contained: the generated _grpc module imports the pb2
# module by flat name; rewrite it to a relative import.
python3 - "${OUT_DIR}/cap_perception_pb2_grpc.py" <<'PY'
import sys
path = sys.argv[1]
text = open(path, encoding="utf-8").read()
text = text.replace(
    "import cap_perception_pb2 as cap__perception__pb2",
    "from . import cap_perception_pb2 as cap__perception__pb2",
)
open(path, "w", encoding="utf-8").write(text)
PY

: > "${HERE}/_generated/__init__.py"
: > "${OUT_DIR}/__init__.py"
echo "CAP_PERCEPTION_STUBS_OK ${OUT_DIR}"
