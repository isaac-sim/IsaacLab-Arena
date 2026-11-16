# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Execute the maintained kitchen force recipes through their public entrypoint."""

import json
import time
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess


@pytest.mark.with_subprocess
@pytest.mark.parametrize("task", ["open_cabinet", "place_bananas_in_wooden_bowl"])
def test_kitchen_object_control_recipe(task, tmp_path):
    kitchen_dir = Path(TestConstants.arena_environments_dir) / "kitchen_bench"
    stem = f"kitchen_bench_lightwheel_{task}_u_shaped_with_island_farmhouse2"
    scene_path = kitchen_dir / f"{stem}.yaml"
    started = time.perf_counter()
    result = run_subprocess(
        [
            TestConstants.python_path,
            str(Path(TestConstants.repo_root) / "isaaclab_arena/agentic_tools/object_control_runner.py"),
            "--env_spec",
            str(scene_path),
            "--commands",
            str(kitchen_dir / f"{stem}_object_control.jsonl"),
            "--output",
            str(tmp_path),
            "--placement_seed",
            "42",
            "--no_record",
            "--visualizer",
            "none",
        ],
        capture_output=True,
        timeout_sec=300,
    )
    (tmp_path / "subprocess_timing.json").write_text(
        json.dumps({"subprocess_wall_time_s": time.perf_counter() - started}) + "\n", encoding="utf-8"
    )
    (tmp_path / "process.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    assert "TASK_SUCCESS" in result.stdout
    report = json.loads((tmp_path / "validation.json").read_text())
    assert report["success"] == [True]
    assert all(report["subtasks"][0])
    # Recorded milestones alone can hide a placement that was disturbed later.
    if task == "place_bananas_in_wooden_bowl":
        for values in report["current_predicates"].values():
            assert values[-1] == [True]
    assert not list(tmp_path.glob("failure_*.json"))
    timings = json.loads((tmp_path / "timings.json").read_text())
    assert timings["simulation_app_startup_s"] > 0
    timer_names = {timer["name"] for timer in timings["timers"]}
    assert {"scene_spec", "scene_config", "scene_loading", "scene_reset", "force_demo"} <= timer_names
