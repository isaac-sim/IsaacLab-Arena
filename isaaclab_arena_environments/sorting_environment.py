# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class TableTopSortCubesEnvironment(ExampleEnvironmentBase):
    """
    A pick and place environment for the Seattle Lab table.
    """

    name = "tabletop_sort_cubes"

    def get_env(self, args_cli: argparse.Namespace):

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sorting_task import SortMultiObjectTask
        from isaaclab_arena.utils.pose import Pose

        assert len(args_cli.destination) == len(args_cli.object)

        # Add the asset registry from the arena migration package
        light = self.asset_registry.get_asset_by_name("light")()
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        background.set_initial_pose(
            Pose(
                position_xyz=(0.3, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        if args_cli.embodiment == "franka":
            embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
                enable_cameras=args_cli.enable_cameras
            )
            # reset initial pose of embodiment
            embodiment.set_initial_pose(
                Pose(
                    position_xyz=(-0.4, 0.0, 0.0),
                    rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
                )
            )

            # order: [panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5, panda_joint6, panda_joint7, panda_finger_joint1, panda_finger_joint2]
            embodiment.set_initial_joint_pose(
                initial_joint_pose=[0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400]
            )

        else:
            raise NotImplementedError

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
            # increase sensitivity for teleop device
            teleop_device.pos_sensitivity = 0.25
            teleop_device.rot_sensitivity = 0.5
        else:
            teleop_device = None

        destination_location1 = self.asset_registry.get_asset_by_name(args_cli.destination[0])()
        destination_location1.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.1, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        destination_location2 = self.asset_registry.get_asset_by_name(args_cli.destination[1])()
        destination_location2.set_initial_pose(
            Pose(
                position_xyz=(0.0, -0.1, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        pick_up_object1 = self.asset_registry.get_asset_by_name(args_cli.object[0])()
        pick_up_object1.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.3, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        pick_up_object2 = self.asset_registry.get_asset_by_name(args_cli.object[1])()
        pick_up_object2.set_initial_pose(
            Pose(
                position_xyz=(0.0, -0.3, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        scene = Scene(
            assets=[background, light, pick_up_object1, pick_up_object2, destination_location1, destination_location2]
        )

        task = SortMultiObjectTask(
            [pick_up_object1, pick_up_object2], [destination_location1, destination_location2], background
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

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            nargs="*",
            default=["red_cube", "green_cube"],
            help="object list (example: --object red_cube green_cube)",
        )
        parser.add_argument(
            "--destination",
            nargs="*",
            default=["red_basket", "green_basket"],
            help="destination list (example: --destination red_basket green_basket)",
        )
        parser.add_argument("--background", type=str, default="table")
        parser.add_argument("--embodiment", type=str, default="franka")
        parser.add_argument("--enable_cameras", type=bool, default=False)
        parser.add_argument("--teleop_device", type=str, default=None)
