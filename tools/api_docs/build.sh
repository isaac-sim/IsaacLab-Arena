#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Build the generated API reference as HTML and as plain text.
# Usage: build.sh <sphinx-build executable> <html output dir> [<text output dir>]
set -euo pipefail

SPHINX_BUILD="${1:?path to sphinx-build}"
HTML_DIR="${2:?html output dir}"
TEXT_DIR="${3:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SPHINX_BUILD}" -b html -q "${HERE}" "${HTML_DIR}"
if [[ -n "${TEXT_DIR}" ]]; then
    "${SPHINX_BUILD}" -b text -q "${HERE}" "${TEXT_DIR}"
fi
