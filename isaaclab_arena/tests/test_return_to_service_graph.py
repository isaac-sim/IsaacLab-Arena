# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build and step the public DROID placement graph through its documented CLI."""

import subprocess
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.return_to_service import _require_assets


@pytest.mark.with_subprocess
def test_service_graph_builds_and_steps():
    _require_assets()
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            TestConstants.python_path,
            "isaaclab_arena_examples/agentic_environment_generation/cli_runner.py",
            "--mode",
            "build",
            "--visualizer",
            "none",
            "--num_envs",
            "1",
            "--num_steps",
            "3",
            "--env_spec",
            "isaaclab_arena_environments/return_to_service/authoring_examples/battery_in_bin.yaml",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[runner] built env 'service_battery_in_bin'" in result.stdout
    assert "[runner] done." in result.stdout
