# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import numpy as np
import torch
from types import SimpleNamespace

import pytest

from isaaclab_arena_vlm_agent_policy.embodiment_adapter.droid_eef_action_adapter import (
    DroidCalibratedChunkActionAdapter,
    validate_actions,
)
from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_chunk_policy import (
    AgentCommandedChunkPolicy,
    AgentCommandedChunkPolicyCfg,
)


@pytest.mark.parametrize("fault", ["horizon", "nan", "gripper", "quaternion", "translation", "rotation"])
def test_reject_invalid_chunk(fault):
    pose = np.array([0.4, 0, 0.3, 0, 0, 0, 1])
    actions = np.tile(np.append(pose, 0), (15, 1))
    if fault == "horizon":
        actions = actions[:-1]
    elif fault == "nan":
        actions[0, 0] = np.nan
    elif fault == "gripper":
        actions[0, 7] = 0.5
    elif fault == "quaternion":
        actions[0, 3:7] = 0
    elif fault == "translation":
        actions[0, 0] += 0.051
    else:
        actions[0, 3:7] = [0, 1, 0, 0]
    with pytest.raises(AssertionError):
        validate_actions({"actions": actions}, pose)


def test_config_rejects_invalid_chunk_size_and_history():
    with pytest.raises(AssertionError):
        AgentCommandedChunkPolicyCfg(decision_history=-1, system_prompt="Chunk test")
    with pytest.raises(AssertionError, match="chunk_size"):
        AgentCommandedChunkPolicyCfg(chunk_size=0, system_prompt="Chunk test")


@pytest.mark.parametrize("lose_tracking", [False, True])
@pytest.mark.parametrize("decision_history", [0, 2])
def test_replay_reobserves_with_optional_history_and_resets(monkeypatch, tmp_path, lose_tracking, decision_history):
    import isaaclab_arena_vlm_agent_policy.policy.vlm_agent_policy as agent_module

    pose = np.array([0.4, 0, 0.3, 0, 0, 0, 1])
    requests = []

    def create(**kwargs):
        requests.append(json.loads(json.dumps(kwargs)))
        states = json.loads(kwargs["messages"][-1]["content"][0]["text"])["proprioception"]
        assert "camera_calibration" in states
        actions = np.tile(np.append(states["eef_pose_root_xyz_xyzw"], 0), (policy.config.chunk_size, 1))
        message = SimpleNamespace(content=json.dumps({"actions": actions.tolist()}), refusal=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])

    monkeypatch.setattr(
        agent_module,
        "OpenAI",
        lambda **kwargs: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)), close=lambda: None
        ),
    )
    monkeypatch.setenv("TEST_CHUNK_KEY", "test-only")

    class Adapter(agent_module.VLMObservationAdapter):
        camera_keys = ("camera",)

        def extract_proprioception(self, env, observation):
            return [{"eef_pose_root_xyz_xyzw": pose.tolist(), "camera_calibration": {"camera": "calibrated"}}]

    policy = AgentCommandedChunkPolicy(
        AgentCommandedChunkPolicyCfg(
            api_key_env_var="TEST_CHUNK_KEY",
            trace_directory=str(tmp_path),
            decision_history=decision_history,
            system_prompt='Chunk experiment prompt with literal JSON: {"actions": []}.\nUse calibrated images.',
        ),
        observation_adapter=Adapter(image_max_edge=384),
        action_adapter=DroidCalibratedChunkActionAdapter(),
    )
    policy.action_adapter.validate_environment = lambda env: None
    policy.set_task_description("Move")
    env = SimpleNamespace(unwrapped=SimpleNamespace(num_envs=1, device="cpu"))
    observation = {"camera_obs": {"camera": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)}}
    try:
        action = policy.get_action(env, observation)
        np.testing.assert_allclose(action[0].numpy(), [0.4, 0, 0.3, 0, 0, 0, 1, 0], atol=1e-6)
        if lose_tracking:
            pose[0] += 0.06
        else:
            for _ in range(policy.config.chunk_size - 1):
                policy.get_action(env, observation)
        assert len(requests) == 1
        policy.get_action(env, observation)
        assert len(requests) == 2
        assert len(policy._histories[0]) == 1
        state = json.loads(requests[-1]["messages"][-1]["content"][0]["text"])["proprioception"]
        assert ("controller_feedback" in state) == lose_tracking
        for _ in range(30):
            policy.get_action(env, observation)
        assert len(requests) == 4
        for i, request in enumerate(requests):
            retained = min(i, decision_history)
            assert [message["role"] for message in request["messages"]] == (
                ["system"] + ["user", "assistant"] * retained + ["user"]
            )
            assert all(isinstance(message["content"], str) for message in request["messages"][:-1])
            assert sum(part["type"] == "image_url" for part in request["messages"][-1]["content"]) == 1
            assert request["messages"][0]["content"] == policy.config.system_prompt
        if decision_history:
            assert len(policy._decisions[0]) == decision_history
        else:
            assert not policy._decisions
        policy.reset(torch.tensor([0]))
        assert policy._chunks == [None] and policy._indices == [0] and policy._steps == [0]
        assert not policy._decisions
        policy.get_action(env, observation)
        assert len(requests[-1]["messages"]) == 2
    finally:
        policy.close()


def test_standalone_examples_preserve_successful_settings():
    from isaaclab_arena.evaluation.arena_experiment_config_loader import load_arena_experiment_from_config_file
    from isaaclab_arena_vlm_agent_policy.policy.agent_commanded_goal_policy import AgentCommandedGoalPolicyCfg

    runs = []
    for suffix in ("goal", "chunk"):
        config = load_arena_experiment_from_config_file(
            f"isaaclab_arena_environments/experiment_configs/droid_pnp_agent_commanded_{suffix}_experiment.yaml",
            device="cuda:0",
        )
        runs.append(next(iter(config.runs.values())))
    goal, chunk = runs
    assert isinstance(goal.policy, AgentCommandedGoalPolicyCfg) and goal.policy.decision_history == 16
    assert isinstance(chunk.policy, AgentCommandedChunkPolicyCfg) and chunk.policy.decision_history == 0
    assert chunk.policy.chunk_size == 15
    assert "Choose ONE command" in goal.policy.system_prompt
    assert "predict exactly 15 actions" in chunk.policy.system_prompt
    assert goal.policy.system_prompt != chunk.policy.system_prompt
    assert goal.environment.episode_length_s == chunk.environment.episode_length_s == 70.0
    for run in (goal, chunk):
        assert run.environment.embodiment == "droid_absolute_ik" and run.environment.enable_cameras
        assert run.environment_builder.placement_seed == 42 and run.environment_builder.seed == 42
        assert run.environment_builder.resolve_on_reset is False
        assert run.policy.model == "openai/openai/gpt-6-astra" and run.policy.image_max_edge == 384
        assert "Camera calibration" in run.policy.system_prompt
    module = "isaaclab_arena_vlm_agent_policy.embodiment_adapter.droid_eef_action_adapter"
    assert (
        goal.policy.observation_adapter
        == chunk.policy.observation_adapter
        == f"{module}.CalibratedDroidObservationAdapter"
    )
    assert goal.policy.action_adapter == f"{module}.DroidGoalActionAdapter"
    assert chunk.policy.action_adapter == f"{module}.DroidCalibratedChunkActionAdapter"
