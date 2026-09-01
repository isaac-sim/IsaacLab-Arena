# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Subprocess tests for Hydra variation overrides on policy_runner."""

import os
import subprocess

import pytest

from isaaclab_arena.tests.test_policy_runner import NUM_STEPS, run_policy_runner
from isaaclab_arena.tests.utils.constants import TestConstants


@pytest.mark.with_subprocess
def test_zero_action_policy_with_hydra_variation_overrides():
    """Boot pick_and_place_maple_table with both Hydra-configurable variations enabled."""
    run_policy_runner(
        policy_type="zero_action",
        example_environment="pick_and_place_maple_table",
        embodiment="droid_abs_joint_pos",
        num_steps=NUM_STEPS,
        enable_cameras=True,
        hydra_overrides=[
            "light.hdr_image.enabled=true",
            "droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true",
        ],
    )


@pytest.mark.with_subprocess
def test_zero_action_policy_graph_spec_with_variation():
    """Boot a graph-spec env with --enable_cameras and a camera-extrinsics variation enabled.

    Exercises that --enable_cameras propagates into the graph-spec embodiment so its wrist
    camera is spawned, letting the camera-extrinsics variation resolve its target scene entity.
    """
    run_policy_runner(
        policy_type="zero_action",
        example_environment=f"{TestConstants.test_data_dir}/pick_and_place_maple_table_env_graph.yaml",
        num_steps=NUM_STEPS,
        enable_cameras=True,
        hydra_overrides=[
            "light.hdr_image.enabled=true",
            "droid_abs_joint_pos.camera_extrinsics_wrist_camera.enabled=true",
        ],
    )


@pytest.mark.with_subprocess
def test_unknown_hydra_variation_override_fails_with_message():
    from isaaclab_arena.tests.utils.constants import TestConstants

    args = [
        TestConstants.python_path,
        f"{TestConstants.evaluation_dir}/policy_runner.py",
        "--policy_type",
        "zero_action",
        "--num_steps",
        str(NUM_STEPS),
        "--headless",
        "pick_and_place_maple_table",
        "--embodiment",
        "droid_abs_joint_pos",
        "light.nonexistent_variation.enabled=true",
    ]
    env = os.environ.copy()
    env["ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE"] = "1"
    result = subprocess.run(
        args,
        env=env,
        timeout=int(os.environ.get("ISAACLAB_ARENA_SUBPROCESS_TIMEOUT", "900")),
        capture_output=True,
        text=True,
        start_new_session=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "Unknown Hydra variation override" in output
    assert "light.hdr_image" in output
    assert "droid_abs_joint_pos.camera_extrinsics_wrist_camera" in output
