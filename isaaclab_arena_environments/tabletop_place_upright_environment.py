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
class TableTopPlaceUprightEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the tabletop place-upright environment."""

    enable_cameras: bool = False
    object: str = "mug"
    background: str = "table"
    embodiment: str = "agibot"
    teleop_device: str | None = "keyboard"


@register_environment
class TableTopPlaceUprightEnvironment(ArenaEnvironmentFactory[TableTopPlaceUprightEnvironmentCfg]):
    """
    A place upright environment for the Seattle Lab table.
    """

    name = "tabletop_place_upright"
    _legacy_argparse_cfg_type = TableTopPlaceUprightEnvironmentCfg

    def build(self, cfg: TableTopPlaceUprightEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""
        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils.configclass import configclass
        from isaaclab_tasks.manager_based.manipulation.stack.mdp.franka_stack_events import randomize_object_pose

        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.place_upright_task import PlaceUprightTask
        from isaaclab_arena.utils.pose import Pose

        @configclass
        class EventCfgPlaceUprightMug:
            """Configuration for events."""

            reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset", params={"reset_joint_targets": True})

            randomize_mug_positions = EventTerm(
                func=randomize_object_pose,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.05, 0.2),
                        "y": (-0.10, 0.10),
                        "z": (0.75, 0.75),
                        "roll": (-1.57, -1.57),
                        "yaw": (-0.57, 0.57),
                    },
                    "asset_cfgs": [SceneEntityCfg("mug")],
                },
            )

        # Add the asset registry from the arena migration package
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        placeable_object = self.asset_registry.get_asset_by_name(cfg.object)(
            initial_pose=Pose(position_xyz=(0.05, 0.0, 0.75), rotation_xyzw=(1.0, 0.0, 0.0, 0.0))
        )
        if cfg.embodiment == "agibot":
            embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(
                enable_cameras=cfg.enable_cameras, arm_mode=ArmMode.LEFT
            )
        else:
            raise NotImplementedError(
                f"Embodiment {cfg.embodiment} not supported for tabletop place upright environment"
            )

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
        else:
            teleop_device = None

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.60, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        background.set_initial_pose(Pose(position_xyz=(0.50, 0.0, 0.625), rotation_xyzw=(0, 0, 0.7071, 0.7071)))
        background.object_cfg.spawn.scale = (1.0, 1.0, 0.60)

        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        scene = Scene(assets=[background, placeable_object, ground_plane, light])

        task = PlaceUprightTask(placeable_object)
        task.events_cfg = EventCfgPlaceUprightMug()

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment
