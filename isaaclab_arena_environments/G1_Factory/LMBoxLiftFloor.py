# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class LMBoxLiftFloor(ExampleEnvironmentBase):

    name: str = "LMBoxLiftFloor"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from .g1_locomanip_lift_task import G1LocomanipLiftTask
        from .pose_utils import pose_range_from_quat

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        assets = []

        BOX_X = 2
        BOX_Y = 0.0

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        assets.append(background)

        object = self.asset_registry.get_asset_by_name(args_cli.object)()
        assets.append(object)

        # Optional table (use --table none or --table "" to disable)
        if args_cli.table and args_cli.table.lower() != "none":
            table = self.asset_registry.get_asset_by_name(args_cli.table)()
            assets.append(table)

        object.scale = (1.0, 1.0, 1.0)
        object.object_cfg = object._init_object_cfg()
        object.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(BOX_X - 0.16, BOX_Y - 0.16, -0.8),
                position_xyz_max=(BOX_X + 0.16, BOX_Y + 0.16, -0.8),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                yaw_jitter=0.16,
            )
        )

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipLiftTask(
                object,
                lift_height=args_cli.lift_height,
                upright_threshold=0.9,
                upright_z_axis_up=True,
                gripper_far_threshold=0.6,
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
        parser.add_argument("--object", type=str, default="locomanip_longbox", help="Object to lift")
        parser.add_argument(
            "--background",
            type=str,
            default="factory_room",
            help="Background scene (factory_room, ground_plane, kitchen, etc.)",
        )
        parser.add_argument("--table", type=str, default="none", help="Table asset (use 'none' or '' to disable)")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument(
            "--lift_height", type=float, default=-0.5, help="Height to lift object for success (meters)"
        )
