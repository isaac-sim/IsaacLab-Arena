# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the temporary Isaac CAP environment demos.

Remove this directory together with ``isaaclab_arena_environments/isaac_cap``.
"""

import os
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

pytestmark = [pytest.mark.isaac_cap, pytest.mark.with_subprocess]

_CABLE_DEMO_SCRIPT = f"{TestConstants.arena_environments_dir}/isaac_cap/cable_routing/cable_env_behaviour_demo.py"
_GEAR_DEMO_SCRIPT = f"{TestConstants.arena_environments_dir}/isaac_cap/gear_insertion/gear_env_behaviour_demo.py"
_TOOL_SORT_DEMO_SCRIPT = (
    f"{TestConstants.arena_environments_dir}/isaac_cap/tool_sorting/tool_sorting_env_behaviour_demo.py"
)
_TOOL_HANGING_DEMO_SCRIPT = (
    f"{TestConstants.arena_environments_dir}/isaac_cap/tool_hanging/tool_hanging_env_behaviour_demo.py"
)


@pytest.mark.parametrize("variant", ("easy", "medium"))
def test_cable_routing_behaviour_demo(variant: str) -> None:
    """Run one headless demo cycle and verify that success resets the environment."""
    result = run_subprocess(
        [
            TestConstants.python_path,
            _CABLE_DEMO_SCRIPT,
            "--variant",
            variant,
            "--cycles",
            "1",
            "--pause-steps",
            "25",
            "--no-real-time",
            "--visualizer",
            "none",
        ],
        capture_output=True,
    )

    assert result is not None
    expected = "[cable-behaviour-demo] cycle 1: success reset observed"
    assert expected in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("variant", ("easy", "medium"))
def test_gear_insertion_behaviour_demo(variant: str) -> None:
    """Run one headless demo cycle and verify that success resets every environment."""
    result = run_subprocess(
        [
            TestConstants.python_path,
            _GEAR_DEMO_SCRIPT,
            variant,
            "--cycles",
            "1",
            "--pause-steps",
            "25",
            "--no-real-time",
            "--visualizer",
            "none",
        ],
        capture_output=True,
    )

    assert result is not None
    expected = "[gear-validation] cycle 1: success reset observed in all 2 environments"
    assert expected in result.stdout, result.stdout + result.stderr


def test_tool_sorting_behaviour_demo() -> None:
    """Run one representative headless cycle and verify that success resets every environment."""
    result = run_subprocess(
        [
            TestConstants.python_path,
            _TOOL_SORT_DEMO_SCRIPT,
            "2",
            "--cycles",
            "1",
            "--teleport-only",
            "--num-envs",
            "1",
            "--pause-steps",
            "25",
            "--no-real-time",
            "--visualizer",
            "none",
        ],
        capture_output=True,
        timeout_sec=900,
    )

    assert result is not None
    expected = "[tool-sort-validation] cycle 1: success reset observed in all 1 environment"
    assert expected in result.stdout, result.stdout + result.stderr


@pytest.mark.with_cameras
def test_tool_hanging_behaviour_demo(tmp_path: Path) -> None:
    """Run one wrench-hanging cycle and verify reset plus camera recordings."""
    video_dir = Path(os.environ.get("ISAACLAB_ARENA_TEST_VIDEO_DIR", tmp_path / "tool_hanging_videos"))
    result = run_subprocess(
        [
            TestConstants.python_path,
            _TOOL_HANGING_DEMO_SCRIPT,
            "--cycles",
            "1",
            "--no-real-time",
            "--video-dir",
            str(video_dir),
            "--visualizer",
            "none",
        ],
        capture_output=True,
        timeout_sec=900,
    )

    assert result is not None
    expected = "[tool-hanging-validation] cycle 1: success reset observed"
    assert expected in result.stdout, result.stdout + result.stderr
    expected_cameras = {
        "top_camera_rgb",
        "side_camera_rgb",
        "left_wrist_camera_rgb",
        "right_wrist_camera_rgb",
    }
    recorded_cameras = {
        path.name.removeprefix("tool-hanging-env0-").removesuffix("-episode-0.mp4")
        for path in video_dir.glob("tool-hanging-env0-*-episode-0.mp4")
        if path.stat().st_size > 0
    }
    assert recorded_cameras == expected_cameras
