# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Snapshots of scene roots, joints and actuator commands."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SceneSnapshot:
    """Scene and control state for N environments, with J joints per articulation."""

    def __init__(self, env: ManagerBasedEnv):
        self.state = env.scene.get_state()
        """Scene state in Isaac Lab's nested state-dictionary format."""
        self.targets = {
            key: (
                asset.data.joint_pos_target.torch.clone(),
                asset.data.joint_vel_target.torch.clone(),
                asset.data.joint_effort_target.torch.clone(),
            )
            for key, asset in env.scene.articulations.items()
        }
        """Position, velocity and effort targets, each (N, J), by articulation key."""

    def restore(self, env: ManagerBasedEnv) -> None:
        """Restore physics state and actuator targets."""
        env.scene.reset_to(self.state)
        for key, (position, velocity, effort) in self.targets.items():
            asset = env.scene.articulations[key]
            asset.set_joint_position_target_index(target=position)
            asset.set_joint_velocity_target_index(target=velocity)
            asset.set_joint_effort_target_index(target=effort)
        env.scene.write_data_to_sim()
        env.sim.forward()


def articulation_link_poses_in_root_frame(env: ManagerBasedEnv) -> dict[str, torch.Tensor]:
    """Return link-to-root poses (N, B, 7), with N environments and B links per articulation."""
    from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul

    poses = {}
    for key, body in env.scene.articulations.items():
        links = body.data.body_link_pose_w.torch
        root = env.arena_world.get_pose_w(key)[:, None, :].expand_as(links)
        position = quat_apply_inverse(root[..., 3:], links[..., :3] - root[..., :3])
        rotation = quat_mul(quat_conjugate(root[..., 3:]), links[..., 3:])
        poses[key] = torch.cat((position, rotation), dim=-1)
    return poses
