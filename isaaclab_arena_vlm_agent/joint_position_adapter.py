# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Articulation joint positions as privileged VLM policy input."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter


class JointPositionObservationAdapter(VLMObservationAdapter):
    """Add selected articulation joint positions to another adapter's prompt values."""

    def __init__(self, observation_adapter: VLMObservationAdapter, scene_keys: Sequence[str]):
        assert scene_keys, "JointPositionObservationAdapter requires at least one scene key"
        self.observation_adapter = observation_adapter
        self.scene_keys = tuple(scene_keys)

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return the wrapped adapter's input with privileged articulation joint positions."""
        images, prompt_values = self.observation_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        scene = env.unwrapped.scene
        joints = {}
        for scene_key in self.scene_keys:
            assert scene_key in scene.articulations, f"'{scene_key}' must name an articulation"
            asset = scene[scene_key]
            joints[scene_key] = dict(zip(asset.joint_names, asset.data.joint_pos.torch[env_id].tolist()))
        assistance = dict(prompt_values.get("simulator_ground_truth_assistance", {}))
        assert "joint_positions_m_or_rad" not in assistance, "joint positions already exist in prompt values"
        assistance["joint_positions_m_or_rad"] = joints
        return images, {**prompt_values, "simulator_ground_truth_assistance": assistance}
