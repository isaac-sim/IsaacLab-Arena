# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class LMBoxTableToShelfPnP(ExampleEnvironmentBase):

    name: str = "LMBoxTableToShelfPnP"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from .events import set_relative_initial_pose
        from .g1_locomanip_pnp_task import G1LocomanipPnPTask
        from .pose_utils import pose_range_from_quat

        assets = []

        TABLE_X = 1.4
        TABLE_Y = 0
        SHELF_X = 0.3
        SHELF_Y = -1.1

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        assets.append(background)

        table = self.asset_registry.get_asset_by_name("locomanip_white_table")()
        table.name = "office_table"
        table.prim_path = "{ENV_REGEX_NS}/office_table"
        table.scale = (0.8, 1.0, 0.95)
        table.object_cfg = table._init_object_cfg()
        table.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TABLE_X - 0.1, TABLE_Y - 0.1, -0.8),
                position_xyz_max=(TABLE_X + 0.1, TABLE_Y + 0.1, -0.8),
                rotation_xyzw=(0, 0, -1, 0),
                yaw_jitter=0.02,
            )
        )
        assets.append(table)

        object = self.asset_registry.get_asset_by_name("locomanip_cardbox")()
        set_relative_initial_pose(
            obj=object,
            reference=table,
            offset_range={"x": (-0.341, 0.159), "y": (-0.278, 0.322), "z": (0.762, 0.762)},
            rotation_xyzw=(0, 0, 0, 1),
            yaw_jitter=0.36,
        )
        assets.append(object)

        industrial_shelf = self.asset_registry.get_asset_by_name("locomanip_industrial_shelf")()
        industrial_shelf.name = "industrial_shelf"
        industrial_shelf.prim_path = "{ENV_REGEX_NS}/industrial_shelf"
        industrial_shelf.scale = (1.0, 1.0, 0.9)
        industrial_shelf.object_cfg = industrial_shelf._init_object_cfg()
        industrial_shelf.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(SHELF_X - 0.18, SHELF_Y - 0.18, -0.8),
                position_xyz_max=(SHELF_X + 0.18, SHELF_Y + 0.18, -0.8),
                rotation_xyzw=(0, 0, 0.70711, 0.70711),
                yaw_jitter=0.36,
            )
        )
        assets.append(industrial_shelf)

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipPnPTask(
                object,
                industrial_shelf,
                max_x_separation=0.16,
                max_y_separation=0.14,
                max_z_separation=0.9227,
                require_static=True,
                gripper_far_threshold=0.1,
                require_contact=True,
            ),
            teleop_device=teleop_device,
        )

        isaaclab_arena_environment.task.robot_pose_range = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (-math.pi / 18, math.pi / 18),
        }

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="brown_box", help="Object to lift")
        parser.add_argument(
            "--background",
            type=str,
            default="factory_room",
            help="Background scene (factory_room, ground_plane, kitchen, etc.)",
        )
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
