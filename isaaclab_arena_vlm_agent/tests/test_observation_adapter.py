# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify embodiment-based state and image selection without robot-specific field names."""

import torch
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from isaaclab_arena_vlm_agent.observation_adapter import (
    ImageObservationAdapter,
    ImageStateObservationAdapter,
    StateObservationAdapter,
    VLMObservationAdapter,
)


def _inputs():
    state_cfg = SimpleNamespace(
        concatenate_terms=False,
        left_joints=SimpleNamespace(func=Mock()),
        right_joints=SimpleNamespace(func=Mock()),
        disabled_term=None,
    )
    camera_cfg = SimpleNamespace(
        head_color=SimpleNamespace(func=Mock(), params={"data_type": "rgb"}),
        hand_color=SimpleNamespace(func=Mock(), params={"data_type": "rgb"}),
        head_depth=SimpleNamespace(func=Mock(), params={"data_type": "distance_to_image_plane"}),
    )
    action_cfg = SimpleNamespace(
        left_target=SimpleNamespace(asset_name="left_robot", body_name="left_tool"),
        right_target=SimpleNamespace(asset_name="right_robot", body_name="right_tool"),
        joints=SimpleNamespace(asset_name="torso"),
    )
    embodiment = SimpleNamespace(
        get_observation_cfg=Mock(return_value=SimpleNamespace(policy=state_cfg, camera_obs=camera_cfg)),
        get_action_cfg=Mock(return_value=action_cfg),
    )

    def get_pose(asset_name, body_name):
        x = 1.0 if asset_name == "left_robot" else 2.0
        return torch.tensor([[x, 0, 0, 0, 0, 0, 1], [x + 10, 0, 0, 0, 0, 0, 1]])

    world = SimpleNamespace(get_body_pose_in_root=Mock(side_effect=get_pose))
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            num_envs=2,
            arena_world=world,
            observation_manager=SimpleNamespace(
                active_terms={"policy": ["left_joints", "right_joints"]}, group_obs_concatenate={"policy": False}
            ),
            action_manager=SimpleNamespace(
                active_terms=list(vars(action_cfg)),
                get_term=lambda name: SimpleNamespace(cfg=getattr(action_cfg, name)),
            ),
        )
    )
    observation = {
        "policy": {
            "left_joints": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "right_joints": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "task_only_term": torch.ones(2, 3),
        },
        "camera_obs": {
            "head_color": torch.zeros(2, 4, 8, 3, dtype=torch.uint8),
            "hand_color": torch.full((2, 8, 4, 3), 255, dtype=torch.uint8),
        },
    }
    return embodiment, env, observation


@pytest.mark.parametrize(
    "adapter_type,include_state,include_images",
    [
        (StateObservationAdapter, True, False),
        (ImageObservationAdapter, False, True),
        (ImageStateObservationAdapter, True, True),
    ],
)
def test_selects_only_requested_modalities_and_configured_terms(adapter_type, include_state, include_images):
    embodiment, env, observation = _inputs()
    if not include_state:
        del observation["policy"]
        embodiment.get_observation_cfg.return_value.policy = None
    if not include_images:
        del observation["camera_obs"]
        embodiment.get_observation_cfg.return_value.camera_obs = None
    adapter = adapter_type() if adapter_type is StateObservationAdapter else adapter_type(embodiment)
    images, state = adapter.extract_vlm_input_from_observation(env, observation, 1)
    if include_state:
        assert state["policy"] == {"left_joints": [3.0, 4.0], "right_joints": [7.0, 8.0]}
        assert set(state["command_body_poses"]) == {"left_target", "right_target"}
        assert state["command_body_poses"]["left_target"]["position_root_m"] == [11.0, 0.0, 0.0]
        assert state["command_body_poses"]["right_target"]["position_root_m"] == [12.0, 0.0, 0.0]
        env.unwrapped.arena_world.get_body_pose_in_root.assert_any_call("left_robot", "left_tool")
        env.unwrapped.arena_world.get_body_pose_in_root.assert_any_call("right_robot", "right_tool")
    else:
        embodiment.get_action_cfg.assert_not_called()
        env.unwrapped.arena_world.get_body_pose_in_root.assert_not_called()
    if include_images:
        assert list(images) == ["head_color", "hand_color"]
        assert images["head_color"].shape == (4, 8, 3)
        assert images["hand_color"].shape == (8, 4, 3)
    else:
        assert images == {}
    if not include_state:
        assert state == {}


def test_joint_only_state_does_not_require_command_body_queries():
    embodiment, env, observation = _inputs()
    env.unwrapped.action_manager.active_terms = ["joints"]
    adapter = StateObservationAdapter()
    state = adapter.extract_robot_state(env, observation, 0)
    assert state["command_body_poses"] == {}
    assert state["policy"]["right_joints"] == [5.0, 6.0]
    env.unwrapped.arena_world.get_body_pose_in_root.assert_not_called()


def test_state_modality_requires_named_observation_terms():
    _, env, observation = _inputs()
    env.unwrapped.observation_manager.group_obs_concatenate["policy"] = True
    with pytest.raises(AssertionError, match="unconcatenated"):
        StateObservationAdapter().extract_robot_state(env, observation, 0)


def test_rgb_modality_requires_configured_rgb_views():
    embodiment, _, _ = _inputs()
    camera_cfg = embodiment.get_observation_cfg.return_value.camera_obs
    del camera_cfg.head_color
    del camera_cfg.hand_color
    with pytest.raises(AssertionError, match="configured RGB"):
        ImageObservationAdapter(embodiment)


def test_observation_base_requires_a_modality_implementation():
    with pytest.raises(TypeError, match="extract_vlm_input_from_observation"):
        VLMObservationAdapter()


def test_state_uses_live_managers_after_environment_configuration_changes():
    embodiment, env, observation = _inputs()
    adapter = ImageStateObservationAdapter(embodiment)
    env.unwrapped.observation_manager.active_terms["policy"] = ["task_only_term"]
    env.unwrapped.action_manager.active_terms = ["new_arm"]
    env.unwrapped.action_manager.get_term = lambda name: SimpleNamespace(
        cfg=SimpleNamespace(asset_name="new_robot", body_name="new_tool")
    )
    _, state = adapter.extract_vlm_input_from_observation(env, observation, 0)
    assert state["policy"] == {"task_only_term": [1.0, 1.0, 1.0]}
    assert set(state["command_body_poses"]) == {"new_arm"}
    env.unwrapped.arena_world.get_body_pose_in_root.assert_called_once_with("new_robot", "new_tool")
