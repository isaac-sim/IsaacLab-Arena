# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline clutter command-line parsing and cache generation."""

import subprocess
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants

CLUTTER_DIR = Path(__file__).parents[3] / "isaaclab_arena_environments/isaac_cap/clutter"
SCRIPT = CLUTTER_DIR / "generate_clutter_scene.py"


def _arguments(output):
    return [
        "--env_spec",
        str(CLUTTER_DIR / "clutter_scene.yaml"),
        "--output",
        str(output),
    ]


def _run(arguments):
    return subprocess.run(
        [TestConstants.python_path, str(SCRIPT), *arguments],
        capture_output=True,
        text=True,
        timeout=180,
    )


def test_help_lists_settling_controls():
    result = _run(["--help"])
    assert result.returncode == 0, result.stderr
    assert "--passive_move_thresh_m" in result.stdout
    assert "--required_quiet_windows" in result.stdout


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["--num_envs", "0"], "must be positive"),
        (["--attempts", "0"], "must be positive"),
        (["--register", "missing_colon"], "expected module:function"),
        (["--num_layouts", "0"], "must be positive"),
    ],
)
def test_invalid_options_fail_without_writing(tmp_path, arguments, message):
    output = tmp_path / "scene.yaml"
    result = _run([*_arguments(output), *arguments])
    assert result.returncode != 0
    assert message in result.stderr
    assert not output.exists()


def test_missing_required_arguments():
    result = _run([])
    assert result.returncode != 0
    assert "required" in result.stderr


def test_existing_output_is_rejected(tmp_path):
    output = tmp_path / "scene.yaml"
    output.write_text("existing scene")
    result = _run(_arguments(output))
    assert result.returncode != 0
    assert "Output already exists" in result.stderr
    assert output.read_text() == "existing scene"


@pytest.mark.with_subprocess
def test_cli_generates_scene_cache(tmp_path):
    import yaml

    output = tmp_path / "scene.yaml"
    result = _run([*_arguments(output), "--num_envs", "2", "--num_layouts", "3", "--viz", "none"])
    assert result.returncode == 0, result.stdout + result.stderr
    spec = yaml.safe_load(output.read_text())
    assert set(spec) == {f"cube_{i}" for i in range(4)}
    for poses in spec.values():
        assert len(poses) == 3
        for pose in poses:
            assert len(pose["position_xyz"]) == 3
            assert len(pose["rotation_xyzw"]) == 4
