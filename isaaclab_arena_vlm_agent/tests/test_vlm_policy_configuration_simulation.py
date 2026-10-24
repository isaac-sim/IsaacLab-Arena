# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify all three example strategies through typed loading and run execution."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_vlm_experiment(simulation_app, mode):
    import json
    import tempfile
    from pathlib import Path
    from unittest.mock import Mock, patch

    from isaaclab_arena.assets.registries import PolicyRegistry
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.arena_run import RunStatus
    from isaaclab_arena.evaluation.run_execution import build_and_run
    from isaaclab_arena.tests.utils.constants import TestConstants

    config_path = Path(TestConstants.arena_environments_dir) / "experiment_configs/droid_pnp_vlm_experiment.yaml"
    run_name = f"droid_pnp_vlm_{mode}"
    experiment = load_arena_experiment_from_config_file(
        config_path,
        device="cuda:0",
        overrides=[
            f"runs.{run_name}.rollout_limit.num_episodes=null",
            f"runs.{run_name}.rollout_limit.num_steps=3",
            f"runs.{run_name}.policy.model=test-model",
        ],
    )
    run = experiment.runs[run_name]
    assert PolicyRegistry().get_policy_type_for_cfg(run.policy) is PolicyRegistry().get_policy(f"vlm_{mode}")
    backend = Mock()

    def hold_current_pose(request):
        content = request.messages[1]["content"]
        assert any(item["type"] == "image_url" for item in content)
        state = json.loads(content[1]["text"])
        pose = state["command_body_poses"]["arm_action"]
        command = {"values": pose["position_root_m"] + pose["quaternion_root_xyzw"] + [0.0]}
        if mode == "chunk":
            assert f"exactly {run.policy.chunk_size} commands" in request.messages[0]["content"]
            return {"commands": [command] * run.policy.chunk_size}
        return command

    backend.infer.side_effect = hold_current_pose
    with tempfile.TemporaryDirectory() as output_dir:
        with patch("isaaclab_arena_vlm_agent.vlm_agent_policy.InferenceBackend", return_value=backend):
            result = build_and_run(run, output_dir)
        assert result.status is RunStatus.COMPLETED
        if mode == "step":
            assert backend.infer.call_count == 3
        elif mode == "chunk":
            assert backend.infer.call_count == 1
        else:
            assert 1 <= backend.infer.call_count <= 3
        backend.client.close.assert_called_once()
    return True


@pytest.mark.with_cameras
@pytest.mark.parametrize("mode", ["step", "chunk", "goal"])
def test_vlm_experiment(mode):
    assert run_function_with_persistent_simulation_app(_test_vlm_experiment, enable_cameras=True, mode=mode)
