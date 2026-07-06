# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class CubeGoalPoseEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the cube goal-pose environment."""

    enable_cameras: bool = False
    object: str = "dex_cube"
    background: str = "table"
    embodiment: str = "franka_ik"
    teleop_device: str | None = None


@register_environment
class CubeGoalPoseEnvironment(ArenaEnvironmentFactory[CubeGoalPoseEnvironmentCfg]):
    """
    A environment for achieving the goal pose of a cube.
    """

    name = "cube_goal_pose"
    _legacy_argparse_cfg_type = CubeGoalPoseEnvironmentCfg

    def build(self, cfg: CubeGoalPoseEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.goal_pose_task import GoalPoseTask
        from isaaclab_arena.utils.pose import Pose

        # Add the asset registry from the arena migration package
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        light = self.asset_registry.get_asset_by_name("light")()
        object = self.asset_registry.get_asset_by_name(cfg.object)()
        object.set_initial_pose(
            Pose(
                position_xyz=(0.1, 0.0, 0.2),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.4, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        # order: [panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7, panda_finger_joint1, panda_finger_joint2]
        embodiment.set_initial_joint_pose(
            initial_joint_pose=[0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]
        )

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
            # increase sensitivity for teleop device
            teleop_device.pos_sensitivity = 0.25
            teleop_device.rot_sensitivity = 0.5
        else:
            teleop_device = None

        scene = Scene(assets=[background, light, object])

        task = GoalPoseTask(
            object,
            target_z_range=(0.2, 1),
            target_orientation_xyzw=(0.0, 0.0, 0.7071, 0.7071),  # yaw 90 degrees
            target_orientation_tolerance_rad=0.2,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment
