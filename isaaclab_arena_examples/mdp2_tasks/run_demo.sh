# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run with bash. Each invocation keeps its inputs, measurements and comparison logs.
set -euo pipefail
demo_dir=$(cd "$(dirname "$0")" && pwd)
arena_root=$(cd "$demo_dir/../.." && pwd)
lab_root=${1:?Pass the Lab draft root containing source/isaaclab_experimental}
lab_python=${2:-"$lab_root/.venv/bin/python"}
shift
if (( $# )); then shift; fi
cases=("$@")
if (( ${#cases[@]} == 0 )); then cases=(single parallel choice sequence two_of_three); fi
result_dir=$(mktemp -d "${TMPDIR:-/tmp}/arena-mdp2-results.XXXXXXXX")
echo "Results: $result_dir"
arena_path="$arena_root:$arena_root/isaaclab_arena_embodiments"
lab_path="$arena_path"
for package in "$lab_root"/source/isaaclab*; do
    lab_path="$lab_path:$package"
done
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
for case_name in "${cases[@]}"; do
    echo "Running $case_name"
    rules="$result_dir/$case_name-rules.json"
    trace="$result_dir/$case_name-trace.json"
    PYTHONPATH="$arena_path" "$arena_root/.venv/bin/python" "$demo_dir/compare_arena.py" export "$case_name" "$rules"
    camera=()
    if [[ "$case_name" == parallel ]]; then camera=(--gif "$result_dir/parallel.gif"); fi
    PYTHONPATH="$lab_path" "$lab_python" "$demo_dir/run_mdp2.py" "$rules" "$trace" "${camera[@]}"
    PYTHONPATH="$arena_path" "$arena_root/.venv/bin/python" "$demo_dir/compare_arena.py" compare "$trace" | tee "$result_dir/$case_name.txt"
done
