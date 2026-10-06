# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest


def _test_goal_camera_contract(simulation_app, output):
    import json
    import os
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena.evaluation.run_execution import build_arena_builder_from_run_cfg
    from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_goal_policy import AgentCommandedGoalPolicy

    config = load_arena_experiment_from_config_file(
        "isaaclab_arena_environments/experiment_configs/droid_pnp_agent_commanded_goal_experiment.yaml", device="cuda:0"
    )
    run = config.runs["droid_pnp_agent_commanded_goal"]
    env = build_arena_builder_from_run_cfg(run).make_registered()
    arm = env.unwrapped.action_manager.get_term("arm_action")
    assert arm.cfg.body_name == "base_link" and arm.cfg.body_offset is None
    assert not arm.cfg.controller.use_relative_mode and arm.cfg.scale == 1.0
    assert arm.cfg.controller.command_type == "pose" and arm.cfg.controller.ik_method == "dls"
    calls = []

    def create(**kwargs):
        assert kwargs["messages"][0]["content"] == config.runs["droid_pnp_agent_commanded_goal"].policy.system_prompt
        state = json.loads(kwargs["messages"][-1]["content"][0]["text"])["proprioception"]
        assert len(state["camera_calibration"]) == 3
        assert "initial_layout" not in state and "objects" not in state
        for calibration in state["camera_calibration"].values():
            assert calibration["image_shape_hw"] == [216, 384]
        calls.append(state)
        payload = {
            "command": "wait",
            "position": None,
            "quaternion_xyzw": None,
            "gripper": None,
            "steps": 6,
            "note": "Hold to verify calibrated observation and command feedback",
        }
        message = SimpleNamespace(content=json.dumps(payload), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None)
    policy = None
    try:
        observation, _ = env.reset()
        with (
            patch.dict(os.environ, {"TEST_GOAL_KEY": "test-only"}),
            patch("isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy.OpenAI", return_value=client),
        ):
            policy = AgentCommandedGoalPolicy(
                replace(
                    config.runs["droid_pnp_agent_commanded_goal"].policy,
                    api_key_env_var="TEST_GOAL_KEY",
                    trace_directory=str(output),
                )
            )
        policy.set_task_description("Hold the current pose")
        for _ in range(12):
            action = policy.get_action(env, observation)
            observation, _, terminated, truncated, _ = env.step(action)
            assert not (terminated | truncated).any()
        assert len(calls) == 2
        assert calls[1]["previous_command_result"]["outcome"] == "completed"
        assert calls[1]["previous_command_result"]["executed_steps"] == 6
        return True
    finally:
        if policy is not None:
            policy.close()
        env.close()


@pytest.mark.with_cameras
def test_goal_camera_contract(tmp_path):
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(
        _test_goal_camera_contract,
        enable_cameras=True,
        force_disable_fabric=False,
        output=tmp_path,
    )
