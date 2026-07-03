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
class PegInsertEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the peg-insert assembly environment."""

    enable_cameras: bool = False
    object: str = "peg"
    destination_object: str = "hole"
    background: str = "table"
    embodiment: str = "franka_ik"
    teleop_device: str | None = None


@register_environment
class PegInsertEnvironment(ArenaEnvironmentFactory[PegInsertEnvironmentCfg]):

    name: str = "peg_insert"
    _legacy_argparse_cfg_type = PegInsertEnvironmentCfg

    def build(self, cfg: PegInsertEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.sim as sim_utils

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.assembly_task import AssemblyTask
        from isaaclab_arena.utils.pose import Pose
        from isaaclab_arena_environments import mdp

        background = self.asset_registry.get_asset_by_name(cfg.background)()
        pick_up_object = self.asset_registry.get_asset_by_name(cfg.object)()
        destination_object = self.asset_registry.get_asset_by_name(cfg.destination_object)()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
        embodiment.scene_config.robot = mdp.FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        background.set_initial_pose(Pose(position_xyz=(0.55, 0.0, 0.0), rotation_xyzw=(0, 0, 0.707, 0.707)))

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.45, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        destination_object.set_initial_pose(
            Pose(
                position_xyz=(0.45, 0.1, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object, destination_object, light])

        task = AssemblyTask(
            task_description="Assemble the peg with the hole",
            fixed_asset=pick_up_object,
            held_asset=destination_object,
            auxiliary_asset_list=[],
            background_scene=background,
            pose_range={
                "x": (0.2, 0.6),
                "y": (-0.20, 0.20),
                "z": (0.0, 0.0),
                "yaw": (-1.0, 1.0),
            },
            min_separation=0.1,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
        )
        return isaaclab_arena_environment
