#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run a command. If it is still running after HANG_DIAG_DELAY_S seconds, print the state of the
# pytest processes (process tree, per-thread kernel wait state, open descriptors, sockets, Python and
# native stacks when py-spy is available, GPU state) every HANG_DIAG_EVERY_S seconds until it ends.
# The command's exit code is preserved.
#
# Usage: run_with_hang_diagnostics.sh <command> [args...]

set -u
DELAY_S="${HANG_DIAG_DELAY_S:-900}"
EVERY_S="${HANG_DIAG_EVERY_S:-300}"

dump_process_state() {
    echo "::group::hang diagnostics $(date -u +%H:%M:%S) (uptime: $(cut -d' ' -f1 /proc/uptime)s)"
    ps -eo pid,ppid,stat,etime,pcpu,rss,wchan:24,args --forest 2>&1 | cut -c1-220
    free -m 2>&1
    df -h /tmp /dev/shm 2>&1
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>&1
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>&1

    for pid in $(pgrep -f "pytest" | grep -vx "$$"); do
        [ -d "/proc/$pid" ] || continue
        echo "--- pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-160)"
        echo "threads by (state, kernel wait channel):"
        for task in /proc/$pid/task/*; do
            state=$(awk '/^State:/ {print $2}' "$task/status" 2>/dev/null)
            echo "$state $(cat "$task/wchan" 2>/dev/null)"
        done | sort | uniq -c | sort -rn
        echo "open descriptors by type:"
        ls -l /proc/$pid/fd 2>/dev/null | awk '{print $NF}' | sed -E 's/[0-9]+/N/g' | sort | uniq -c | sort -rn | head -15
        echo "sockets:"
        ss -tnp 2>/dev/null | grep "pid=$pid," | head -20 || true

        if ! command -v py-spy > /dev/null 2>&1 && [ ! -f /tmp/py_spy_install_attempted ]; then
            touch /tmp/py_spy_install_attempted
            uv pip install -q py-spy > /dev/null 2>&1 || echo "py-spy install failed"
        fi
        if command -v py-spy > /dev/null 2>&1; then
            echo "py-spy stacks:"
            py-spy dump --native --pid "$pid" 2>&1 | head -150
        fi
        if command -v gdb > /dev/null 2>&1; then
            echo "gdb native stacks:"
            gdb -p "$pid" -batch -ex "thread apply all bt 20" 2>&1 | tail -120
        fi
    done
    echo "::endgroup::"
}

(
    sleep "$DELAY_S"
    while true; do
        dump_process_state
        sleep "$EVERY_S"
    done
) &
WATCHER_PID=$!

"$@"
EXIT_CODE=$?

kill "$WATCHER_PID" 2> /dev/null
wait "$WATCHER_PID" 2> /dev/null
exit "$EXIT_CODE"
