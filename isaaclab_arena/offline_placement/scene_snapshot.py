# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene state and control targets for reversible offline physics trials."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SceneSnapshot:
    """Scene and control state for N environments, with J joints and B links per articulation."""

    def __init__(self, env: ManagerBasedEnv, geometry_keys: list[str]):
        self.state = env.scene.get_state()
        """Scene state in Isaac Lab's nested state-dictionary format."""
        self.poses = {key: env.arena_world.get_pose_e(key).clone() for key in geometry_keys}
        """Environment-local poses (N, 7), ordered xyz/xyzw, by scene key."""
        self.targets = {
            key: (
                asset.data.joint_pos_target.torch.clone(),
                asset.data.joint_vel_target.torch.clone(),
                asset.data.joint_effort_target.torch.clone(),
            )
            for key, asset in env.scene.articulations.items()
        }
        """Position, velocity and effort targets, each (N, J), by articulation key."""
        self.links = {key: asset.data.body_link_pose_w.torch.clone() for key, asset in env.scene.articulations.items()}
        """World-frame link poses (N, B, 7), ordered xyz/xyzw, by articulation key."""

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
