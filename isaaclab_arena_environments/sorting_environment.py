# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


@dataclass
class TableTopSortCubesEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the tabletop cube-sorting environment."""

    enable_cameras: bool = False
    objects: list[str] = field(default_factory=lambda: ["red_cube", "green_cube"])
    destinations: list[str] = field(default_factory=lambda: ["red_container", "green_container"])
    background: str = "table"
    embodiment: str = "franka_ik"
    teleop_device: str | None = None


@register_environment
class TableTopSortCubesEnvironment(ArenaEnvironmentFactory[TableTopSortCubesEnvironmentCfg]):
    """
    A pick and place environment for the Seattle Lab table.
    """

    name = "tabletop_sort_cubes"
    _legacy_argparse_cfg_type = TableTopSortCubesEnvironmentCfg

    def build(self, cfg: TableTopSortCubesEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build the environment from its typed configuration."""

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
        from isaaclab_arena.utils.pose import Pose

        assert (
            len(cfg.destinations) == len(cfg.objects) == 2
        ), "Only 2 objects and 2 destinations are supported in this environment."

        # Add the asset registry from the arena migration package
        light = self.asset_registry.get_asset_by_name("light")()
        background = self.asset_registry.get_asset_by_name(cfg.background)()
        background.set_initial_pose(
            Pose(
                position_xyz=(0.3, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        if cfg.embodiment == "franka_ik":
            embodiment = self.asset_registry.get_asset_by_name(cfg.embodiment)(enable_cameras=cfg.enable_cameras)
            # reset initial pose of embodiment
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

        else:
            raise NotImplementedError

        if cfg.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(cfg.teleop_device)()
            # increase sensitivity for teleop device
            teleop_device.pos_sensitivity = 0.25
            teleop_device.rot_sensitivity = 0.5
        else:
            teleop_device = None

        destination_location_1 = self.asset_registry.get_asset_by_name(cfg.destinations[0])()
        destination_location_1.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.1, 0.1),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        destination_location_2 = self.asset_registry.get_asset_by_name(cfg.destinations[1])()
        destination_location_2.set_initial_pose(
            Pose(
                position_xyz=(0.0, -0.1, 0.1),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        pick_up_object_1 = self.asset_registry.get_asset_by_name(cfg.objects[0])()
        pick_up_object_1.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.3, 0.1),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        pick_up_object_2 = self.asset_registry.get_asset_by_name(cfg.objects[1])()
        pick_up_object_2.set_initial_pose(
            Pose(
                position_xyz=(0.0, -0.3, 0.1),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        scene = Scene(
            assets=[
                background,
                light,
                pick_up_object_1,
                pick_up_object_2,
                destination_location_1,
                destination_location_2,
            ]
        )

        task = SortMultiObjectTask(
            pick_up_object_list=[pick_up_object_1, pick_up_object_2],
            destination_location_list=[destination_location_1, destination_location_2],
            background_scene=background,
        )

        # add custom force threshold for success termination
        task.termination_cfg.success.params["force_threshold"] = 0.1

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment
