# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify goal-policy execution and moving-camera calibration in an isolated simulator."""

import sys

import pytest

from isaaclab_arena.tests.utils.subprocess import run_subprocess


def _test_vlm_experiment(simulation_app):
    import base64
    import io
    import json
    import numpy as np
    import tempfile
    from pathlib import Path
    from scipy.spatial.transform import Rotation
    from unittest.mock import Mock, patch

    from PIL import Image

    from isaaclab_arena.assets.registries import PolicyRegistry
    from isaaclab_arena.embodiments.droid.droid import DroidCameraCfg
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.arena_run import RunStatus
    from isaaclab_arena.evaluation.run_execution import build_and_run
    from isaaclab_arena.tests.utils.constants import TestConstants

    config_path = Path(TestConstants.arena_environments_dir) / "experiment_configs/droid_pnp_vlm_experiment.yaml"
    run_name = "droid_pnp_vlm_goal"
    experiment = load_arena_experiment_from_config_file(
        config_path,
        device="cuda:0",
        overrides=[
            f"runs.{run_name}.rollout_limit.num_episodes=null",
            f"runs.{run_name}.rollout_limit.num_steps=3",
            f"runs.{run_name}.policy.model=test-model",
            f"runs.{run_name}.policy.max_steps_per_goal=1",
        ],
    )
    run = experiment.runs[run_name]
    assert PolicyRegistry().get_policy_type_for_cfg(run.policy) is PolicyRegistry().get_policy("vlm_goal")
    backend = Mock()
    wrist_positions = []
    wrist_offset = DroidCameraCfg().wrist_camera.offset

    def move_nearby(request):
        content = request.messages[1]["content"]
        assert any(item["type"] == "image_url" for item in content)
        state = json.loads(content[1]["text"])
        assert "All quaternions in input and output use XYZW" in request.messages[0]["content"]
        assert len(state["camera_geometry"]) == 3
        for index in range(2, len(content), 2):
            name = content[index]["text"].removeprefix("camera=")
            geometry = state["camera_geometry"][name]
            url = content[index + 1]["image_url"]["url"]
            image = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
            assert tuple(geometry["image_size_px"]) == image.size == (384, 216)
            assert geometry["optical_frame"] == "x_right_y_down_z_forward"
            K = np.array(geometry["intrinsic_matrix"])
            assert K[0, 0] > 0 and K[1, 1] > 0
            np.testing.assert_allclose(K[2], [0, 0, 1])
            np.testing.assert_allclose(np.linalg.norm(geometry["quaternion_world_xyzw"]), 1, atol=1e-5)
            # DROID's fixed root is identity in this example, so world and root camera poses agree.
            root = geometry["poses_in_robot_roots"]["robot"]
            np.testing.assert_allclose(root["position_root_m"], geometry["position_world_m"], atol=1e-5)
            np.testing.assert_allclose(
                Rotation.from_quat(root["quaternion_root_xyzw"]).as_matrix(),
                Rotation.from_quat(geometry["quaternion_world_xyzw"]).as_matrix(),
                atol=1e-5,
            )
        pose = state["command_body_poses"]["arm_action"]
        np.testing.assert_allclose(state["policy"]["eef_quat"], pose["quaternion_root_xyzw"], atol=1e-5)
        # Reset can return an image captured before the joint reset. After a rendered step,
        # check the moving camera against its measured parent link, not its own cached data.
        wrist = state["camera_geometry"]["wrist_camera_rgb"]
        if wrist_positions:
            rotation_W_E = Rotation.from_quat(state["policy"]["eef_quat"])
            t_W_C = rotation_W_E.apply(wrist_offset.pos) + state["policy"]["eef_pos"]
            np.testing.assert_allclose(wrist["position_world_m"], t_W_C, atol=1e-4)
            rotation_W_C = rotation_W_E * Rotation.from_quat(wrist_offset.rot) * Rotation.from_euler("x", np.pi)
            # Allow sub-milliradian rounding in the authored USD mount quaternion.
            rotation_error = rotation_W_C.inv() * Rotation.from_quat(wrist["quaternion_world_xyzw"])
            assert rotation_error.magnitude() < 5e-4
        wrist_positions.append(wrist["position_world_m"])
        command = {"values": pose["position_root_m"] + pose["quaternion_root_xyzw"] + [0.0]}
        command["values"][0] += 0.01
        return command

    backend.infer.side_effect = move_nearby
    with tempfile.TemporaryDirectory() as output_dir:
        with patch("isaaclab_arena_vlm_agent.vlm_agent_policy.InferenceBackend", return_value=backend):
            result = build_and_run(run, output_dir)
        assert result.status is RunStatus.COMPLETED
        assert backend.infer.call_count == 3
        assert np.linalg.norm(np.array(wrist_positions[-1]) - wrist_positions[0]) > 1e-4
        backend.close.assert_called_once()
    return True


@pytest.mark.with_cameras
@pytest.mark.with_subprocess
def test_vlm_experiment():
    # Other VLM tests initialize the parent SimulationApp without camera support.
    run_subprocess([
        sys.executable,
        "-c",
        (
            "from isaaclab_arena.tests.utils.persistent_simulation_app import"
            " run_function_with_persistent_simulation_app; from"
            " isaaclab_arena_vlm_agent.tests.test_vlm_policy_configuration_simulation import _test_vlm_experiment;"
            " assert run_function_with_persistent_simulation_app(_test_vlm_experiment, enable_cameras=True)"
        ),
    ])
