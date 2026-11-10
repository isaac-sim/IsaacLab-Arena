#!/usr/bin/env bash
# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run a command. If it is still running after HANG_DIAG_DELAY_S seconds, print the state of the
# pytest processes every HANG_DIAG_EVERY_S seconds until it ends: process tree, which process traces
# which (TracerPid), per-thread kernel wait state, kernel stacks, open descriptors, Python and native
# stacks (py-spy, gdb), a short strace, and the tail of the Kit logs and any crash reports.
# The command's exit code is preserved.
#
# Usage: run_with_hang_diagnostics.sh <command> [args...]

set -u
DELAY_S="${HANG_DIAG_DELAY_S:-900}"
EVERY_S="${HANG_DIAG_EVERY_S:-300}"
PROCESS_PATTERN="${HANG_DIAG_PROCESS_PATTERN:-pytest}"
KIT_DATA_DIR=/isaac-sim/kit/data/Kit/IsaacLab
KIT_LOG_DIR=/isaac-sim/kit/logs/Kit/IsaacLab

print_environment_limits() {
    echo "ptrace_scope: $(cat /proc/sys/kernel/yama/ptrace_scope 2>&1)"
    echo "this shell: $(id -u) uid, $(grep -E '^Cap(Eff|Bnd)' /proc/self/status | tr '\n' ' ')"
    echo "tools: $(for t in gdb strace lsof py-spy; do command -v $t > /dev/null 2>&1 && printf '%s ' "$t"; done)"
}

print_kit_logs_and_crash_reports() {
    echo "--- Kit data dir (dumps and crash reports)"
    ls -la --time-style=full-iso "$KIT_DATA_DIR"/*/ 2>&1 | grep -E "dmp|crash|\.txt|total" | head -20
    for crash_report in $(ls -t "$KIT_DATA_DIR"/*/crash_*.txt 2> /dev/null | head -2); do
        echo "--- crash report $crash_report"
        head -150 "$crash_report" | cut -c1-240
    done
    kit_log=$(ls -t "$KIT_LOG_DIR"/*/kit_*.log 2> /dev/null | head -1)
    if [ -n "$kit_log" ]; then
        echo "--- tail of $kit_log"
        tail -60 "$kit_log" | cut -c1-240
    fi
    for omni_log in $(ls -t /github/home/.nvidia-omniverse/logs/*.log 2> /dev/null | head -2); do
        echo "--- tail of $omni_log"
        tail -20 "$omni_log" | cut -c1-240
    done
}

print_one_process() {
    local pid="$1"
    echo "--- pid $pid: $(tr '\0' ' ' < /proc/$pid/cmdline | cut -c1-140)"
    grep -E '^(State|PPid|TracerPid|Threads):' /proc/$pid/status 2>&1 | tr '\n' ' '
    echo
    echo "syscall: $(cat /proc/$pid/syscall 2>&1 | cut -c1-120)"
    echo "kernel stack: $(cat /proc/$pid/stack 2>&1 | head -12 | tr '\n' ';')"
    echo "threads by (state, kernel wait channel):"
    for task in /proc/$pid/task/*; do
        state=$(awk '/^State:/ {print $2}' "$task/status" 2>/dev/null)
        echo "$state $(cat "$task/wchan" 2>/dev/null)"
    done | sort | uniq -c | sort -rn
    echo "open files (dumps, locks, sockets):"
    lsof -nP -p "$pid" 2>/dev/null | grep -vE "carb-RStringInternals|/dev/nvidia|\.so" | head -40 | cut -c1-200

    if command -v py-spy > /dev/null 2>&1; then
        echo "py-spy stacks:"
        py-spy dump --native --pid "$pid" 2>&1 | head -120
    fi
    if command -v gdb > /dev/null 2>&1; then
        echo "gdb native stacks:"
        gdb -p "$pid" -batch -ex "set pagination off" -ex "thread apply all bt 25" 2>&1 | tail -150 | cut -c1-240
    fi
    if command -v strace > /dev/null 2>&1; then
        echo "strace (3 s):"
        timeout 3 strace -f -tt -s 80 -p "$pid" 2>&1 | head -25
    fi
}

dump_process_state() {
    echo "::group::hang diagnostics $(date -u +%H:%M:%S) (uptime: $(cut -d' ' -f1 /proc/uptime)s)"
    print_environment_limits
    ps -eo pid,ppid,stat,etime,pcpu,rss,wchan:24,args --forest 2>&1 | cut -c1-220
    free -m 2>&1
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv 2>&1

    echo "--- tracing relationships (process -> its tracer)"
    for status in /proc/[0-9]*/status; do
        tracer=$(awk '/^TracerPid:/ {print $2}' "$status" 2>/dev/null)
        [ -n "$tracer" ] && [ "$tracer" != "0" ] && echo "pid $(basename "$(dirname "$status")") is traced by pid $tracer"
    done

    for pid in $(pgrep -f "$PROCESS_PATTERN" | grep -vx "$$"); do
        [ -d "/proc/$pid" ] || continue
        print_one_process "$pid"
    done
    print_kit_logs_and_crash_reports
    echo "::endgroup::"
}

if ! command -v py-spy > /dev/null 2>&1; then
    uv pip install -q py-spy > /dev/null 2>&1 || true
fi

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
