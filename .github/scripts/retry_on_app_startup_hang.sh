#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run a test command, and rerun it when it fails while Isaac Sim (Kit) is starting up.
#
# Kit startup intermittently hangs on the CI GPU runners. With ISAACLAB_ARENA_APP_STARTUP_TIMEOUT_SEC set,
# get_app_launcher() turns such a hang into a stack dump and exit code 1. A failure counts as a startup hang
# when the output has more "initialization started" markers than "initialization complete" markers.
#
# Usage: retry_on_app_startup_hang.sh <command> [args...]
# Environment:
#   APP_STARTUP_HANG_RETRIES  Number of reruns after a startup hang. Defaults to 2.

set -uo pipefail

STARTED_MARKER="[isaaclab-arena] AppLauncher initialization started"
COMPLETE_MARKER="[isaaclab-arena] AppLauncher initialization complete"
MAX_ATTEMPTS=$((${APP_STARTUP_HANG_RETRIES:-2} + 1))

output_file="$(mktemp)"
trap 'rm -f "${output_file}"' EXIT

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    "$@" 2>&1 | tee "${output_file}"
    exit_code=${PIPESTATUS[0]}
    if [ "${exit_code}" -eq 0 ]; then
        exit 0
    fi

    started_count=$(grep -cF "${STARTED_MARKER}" "${output_file}")
    complete_count=$(grep -cF "${COMPLETE_MARKER}" "${output_file}")
    if [ "${started_count}" -le "${complete_count}" ]; then
        exit "${exit_code}"
    fi

    echo "::warning::Isaac Sim startup hang (attempt ${attempt}/${MAX_ATTEMPTS}, exit code ${exit_code})."
done

echo "::error::Isaac Sim startup hung on all ${MAX_ATTEMPTS} attempts."
exit "${exit_code}"
