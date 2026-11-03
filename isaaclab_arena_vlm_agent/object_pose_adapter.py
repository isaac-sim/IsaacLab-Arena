# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exact scene-object poses as privileged VLM policy input."""

from __future__ import annotations

from collections.abc import Sequence

from isaaclab_arena_vlm_agent.observation_adapter import VLMObservationAdapter
from isaaclab_arena_vlm_agent.pose_utils import poses_in_robot_roots, read_robot_root_poses_w


class ObjectPoseObservationAdapter(VLMObservationAdapter):
    """Add exact scene-object poses in the robot root frames to another adapter's prompt values.

    These poses are privileged state that cameras and robot sensors do not provide. They show what a policy
    achieves once it knows exactly where objects are, as a reference for policies that must locate them.
    """

    def __init__(self, observation_adapter: VLMObservationAdapter, scene_keys: Sequence[str]):
        """Wrap an adapter and select the objects to report.

        Args:
            observation_adapter: Adapter supplying the images and the other prompt values.
            scene_keys: Scene keys of the rigid objects, articulations, or scene extras to report.
        """
        assert scene_keys, "ObjectPoseObservationAdapter requires at least one scene key"
        self.observation_adapter = observation_adapter
        self.scene_keys = tuple(scene_keys)

    def extract_vlm_input_from_observation(self, env, observation, env_id: int):
        """Return the wrapped adapter's input with object_poses added to the prompt values."""
        images, prompt_values = self.observation_adapter.extract_vlm_input_from_observation(env, observation, env_id)
        base = env.unwrapped
        roots = read_robot_root_poses_w(base, env_id)
        object_poses = {}
        for scene_key in self.scene_keys:
            T_W_O = base.arena_world.get_pose_w(scene_key)[env_id].detach().cpu().numpy()
            object_poses[scene_key] = {"poses_in_robot_roots": poses_in_robot_roots(roots, T_W_O[:3], T_W_O[3:])}
        return images, {**prompt_values, "object_poses": object_poses}
