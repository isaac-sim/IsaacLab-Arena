#!/usr/bin/env bash
# Reproduce CI subprocess-test stalling locally.
#
# On CI, all 4 test sections run in the SAME container sequentially.
# In-process sections may leave orphaned Kit/GPU processes that block
# the subsequent subprocess section.  This script simulates that
# full sequence and then runs subprocess tests one-by-one to pinpoint
# which one stalls.
#
# Usage (inside Docker container):
#   bash scripts/repro_ci_subprocess_stall.sh              # full CI sequence
#   bash scripts/repro_ci_subprocess_stall.sh --subprocess-only  # skip in-process, jump to subprocess isolation
#   bash scripts/repro_ci_subprocess_stall.sh --loop        # repeat full sequence until failure
#
# Tunables (env vars):
#   CPUS              CPU range for taskset (default "0-3", ~4 cores like CI)
#   TIMEOUT           Per-subprocess timeout in seconds (default 900, same as CI)
#   SKIP_CACHE_WIPE   Set to 1 to keep caches (test residual-process effects only)
#   LOG_DIR           Directory for log files (default "logs/repro_ci")
#
# Logs are saved to $LOG_DIR/repro_<timestamp>.log (tee'd to stdout).

set -euo pipefail

CPUS="${CPUS:-0-3}"
TIMEOUT="${TIMEOUT:-900}"
SKIP_CACHE_WIPE="${SKIP_CACHE_WIPE:-0}"
LOG_DIR="${LOG_DIR:-logs/repro_ci}"

MODE="full"
[[ "${1:-}" == "--subprocess-only" ]] && MODE="subprocess-only"
[[ "${1:-}" == "--loop" ]] && MODE="loop"

OV_CACHE_DIR="${HOME}/.cache/ov"
PYTHON="/isaac-sim/python.sh"

# --- Logging setup ---
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE="${LOG_DIR}/repro_${TIMESTAMP}.log"
echo "[repro] Logging to ${LOGFILE}"
exec > >(tee -a "${LOGFILE}") 2>&1

# --- Subprocess test files and their individual test functions ---
SUBPROCESS_TESTS=(
    "isaaclab_arena/tests/test_policy_runner.py::test_zero_action_policy_press_button"
    "isaaclab_arena/tests/test_policy_runner.py::test_zero_action_policy_kitchen_pick_and_place"
    "isaaclab_arena/tests/test_policy_runner.py::test_zero_action_policy_galileo_pick_and_place"
    "isaaclab_arena/tests/test_policy_runner.py::test_zero_action_policy_gr1_open_microwave"
    "isaaclab_arena/tests/test_policy_runner.py::test_replay_policy_gr1_open_microwave"
    "isaaclab_arena/tests/test_external_environment.py::test_external_environment_franka_table"
    "isaaclab_arena/tests/test_external_environment.py::test_external_environment_franka_table_with_task"
    "isaaclab_arena/tests/test_sequential_task_mimic_data_generation.py::test_franka_put_and_close_door_mimic_data_generation_single_env"
    "isaaclab_arena/tests/test_sequential_task_mimic_data_generation.py::test_franka_put_and_close_door_mimic_data_generation_multi_env"
    "isaaclab_arena/tests/test_rsl_rl.py::test_rl_train_and_eval_lift_object"
    "isaaclab_arena/tests/test_action_chunking_client.py::test_action_chunking_client_end_to_end_with_dummy_chunking_server"
)

# ---------- helpers ----------

wipe_kit_caches() {
    if [[ "$SKIP_CACHE_WIPE" == "1" ]]; then
        echo "[repro] SKIP_CACHE_WIPE=1 — keeping caches"
        return
    fi
    echo "[repro] Wiping Kit caches to simulate CI cold start …"
    rm -rf "${OV_CACHE_DIR}/shaders"
    rm -rf "${OV_CACHE_DIR}/Kit"
    rm -rf "${OV_CACHE_DIR}/DerivedDataCache"
    rm -rf "${OV_CACHE_DIR}/ogn_generated"
    echo "[repro] Caches wiped.  Remaining:"
    du -sh "${OV_CACHE_DIR}"/* 2>/dev/null || echo "  (empty)"
}

show_orphans() {
    echo "[repro] Checking for orphaned python/kit processes …"
    # Show any python or kit-related processes (excluding this script and grep)
    local procs
    procs=$(ps aux | grep -E '(python|kit|carb|omni)' | grep -v -E '(grep|repro_ci)' || true)
    if [[ -n "$procs" ]]; then
        echo "[repro] *** ORPHANED PROCESSES DETECTED ***"
        echo "$procs"
        echo ""
    else
        echo "[repro] No orphans found."
    fi
}

kill_orphans() {
    echo "[repro] Killing any leftover python/kit child processes …"
    # Kill python processes that look like Isaac Sim (but not our own bash/pytest)
    pkill -9 -f 'isaac-sim/kit/python' 2>/dev/null || true
    pkill -9 -f 'omni.kit' 2>/dev/null || true
    sleep 2
}

run_timed() {
    # Run a command with taskset, print elapsed time, return exit code
    local label="$1"
    shift
    echo ""
    echo "--- [$label] ---"
    echo "[repro] Command: $*"
    local t0
    t0=$(date +%s)

    local rc=0
    taskset -c "${CPUS}" "$@" || rc=$?

    local elapsed=$(( $(date +%s) - t0 ))
    if [[ $rc -eq 0 ]]; then
        echo "[repro] [$label] PASSED in ${elapsed}s"
    else
        echo "[repro] [$label] FAILED (exit $rc) after ${elapsed}s"
    fi
    return $rc
}

# ---------- CI section runners ----------

run_inprocess_sections() {
    echo ""
    echo "================================================================"
    echo "[repro] Phase 1: In-process sections (mimicking CI steps 1-3)"
    echo "================================================================"

    # run_timed "Newton" \
    #     "$PYTHON" -m pytest -sv --durations=0 -m with_newton \
    #     isaaclab_arena/tests/ || true

    # show_orphans

    # run_timed "PhysX (no cameras)" \
    #     "$PYTHON" -m pytest -sv --durations=0 \
    #     -m "not with_cameras and not with_subprocess and not with_newton" \
    #     isaaclab_arena/tests/ || true

    # show_orphans

    run_timed "PhysX (with cameras)" \
        "$PYTHON" -m pytest -sv --durations=0 \
        -m "with_cameras and not with_subprocess and not with_newton" \
        isaaclab_arena/tests/ || true

    show_orphans
    echo ""
    echo "[repro] Phase 1 complete.  Any orphans above may affect Phase 2."
    echo ""
}

run_subprocess_bulk() {
    echo ""
    echo "================================================================"
    echo "[repro] Phase 2a: Subprocess tests as one pytest (like CI)"
    echo "================================================================"
    run_timed "Subprocess (bulk)" \
        env ISAACLAB_ARENA_SUBPROCESS_TIMEOUT="${TIMEOUT}" \
        "$PYTHON" -m pytest -sv --durations=0 -m with_subprocess \
        isaaclab_arena/tests/
}

run_subprocess_isolated() {
    echo ""
    echo "================================================================"
    echo "[repro] Phase 2b: Subprocess tests — ONE AT A TIME"
    echo "================================================================"
    echo "[repro] Running ${#SUBPROCESS_TESTS[@]} tests individually to find the culprit."
    echo ""

    local pass=0
    local fail=0
    local results=()

    for test_id in "${SUBPROCESS_TESTS[@]}"; do
        local short_name
        short_name=$(echo "$test_id" | sed 's|isaaclab_arena/tests/||')

        show_orphans

        local t0 rc
        t0=$(date +%s)
        rc=0

        taskset -c "${CPUS}" \
            env ISAACLAB_ARENA_SUBPROCESS_TIMEOUT="${TIMEOUT}" \
            "$PYTHON" -m pytest -sv --durations=0 "$test_id" || rc=$?

        local elapsed=$(( $(date +%s) - t0 ))

        if [[ $rc -eq 0 ]]; then
            results+=("PASS  ${elapsed}s  ${short_name}")
            pass=$((pass + 1))
        else
            results+=("FAIL  ${elapsed}s  ${short_name}  (exit $rc)")
            fail=$((fail + 1))
        fi
    done

    echo ""
    echo "================================================================"
    echo "[repro] SUBPROCESS TEST RESULTS"
    echo "================================================================"
    printf "%-6s %-8s %s\n" "Status" "Time" "Test"
    printf "%-6s %-8s %s\n" "------" "--------" "----"
    for r in "${results[@]}"; do
        echo "$r"
    done
    echo ""
    echo "[repro] Passed: $pass  Failed: $fail  Total: ${#SUBPROCESS_TESTS[@]}"
}

# ---------- main ----------

run_once() {
    local run_num="$1"
    echo ""
    echo "================================================================"
    echo "[repro] === RUN #${run_num} ===  CPUs=${CPUS}  timeout=${TIMEOUT}s  mode=${MODE}"
    echo "================================================================"

    wipe_kit_caches

    if [[ "$MODE" != "subprocess-only" ]]; then
        run_inprocess_sections
    else
        echo "[repro] Skipping in-process sections (--subprocess-only)"
    fi

    # First try the bulk run (exactly like CI).  If it passes, great.
    # If it fails/hangs, the isolated run below will pinpoint the culprit.
    local bulk_rc=0
    run_subprocess_bulk || bulk_rc=$?

    if [[ $bulk_rc -ne 0 ]]; then
        echo ""
        echo "[repro] Bulk subprocess run FAILED — re-running tests individually to isolate …"
        kill_orphans
        run_subprocess_isolated
    else
        echo ""
        echo "[repro] Bulk subprocess run PASSED."
    fi
}

case "$MODE" in
    full|subprocess-only)
        run_once 1
        ;;
    loop)
        echo "[repro] Loop mode — Ctrl-C to stop"
        n=1
        while true; do
            run_once "$n"
            n=$((n + 1))
        done
        ;;
esac
