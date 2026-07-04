# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_cfg import ArenaEnvironmentCfg
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class LiftObjectEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the lift-object environment."""

    object: str = "dex_cube"
    teleop_device: str | None = None
    embodiment: str = "franka_joint_pos"
    """Use joint control by default because it is more reliable than IK for RL training."""
    rl_training_mode: bool = False


@register_environment
class LiftObjectEnvironment(ExampleEnvironmentBase[LiftObjectEnvironmentCfg]):

    name: str = "lift_object"
    _legacy_argparse_cfg_type = LiftObjectEnvironmentCfg

    def build(self, cfg: LiftObjectEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab_arena_examples.policy.base_rsl_rl_policy as base_rsl_rl_policy
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("table")()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()

        # Add ground plane and light to the scene
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        assets = [background, pick_up_object, ground_plane, light]

        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(concatenate_observation_terms=True)

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        # Set all positions
        background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_xyzw=(0, 0, 0.707, 0.707)))
        pick_up_object.set_initial_pose(Pose(position_xyz=(0.5, 0, 0.055), rotation_xyzw=(0, 0, 0, 1)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

        # Compose the scene
        # If using for an IL task, add the goal position as a marker to the scene
        scene = Scene(assets=assets)

        task = LiftObjectTaskRL(
            pick_up_object,
            background,
            embodiment,
            minimum_height_to_lift=0.04,
            episode_length_s=5.0,
            rl_training_mode=cfg.rl_training_mode,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg=f"{base_rsl_rl_policy.__name__}:RLPolicyCfg",
        )

        return isaaclab_arena_environment
