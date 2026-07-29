# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class LMPushShelfForward(ExampleEnvironmentBase):

    name: str = "LMPushShelfForward"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from .g1_locomanip_pnp_task import G1LocomanipPnPTask
        from .pose_utils import pose_range_from_quat

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        assets = []
        TARGET_X = 1.5
        TARGET_Y = 0.0

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        background.set_initial_pose(Pose(position_xyz=(7.0, 7.2, -0.785), rotation_xyzw=(0.0, 0.0, 0.70711, 0.70711)))
        assets.append(background)

        object = self.asset_registry.get_asset_by_name("locomanip_mobile_cart")()
        object.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TARGET_X - 0.3, TARGET_Y - 0.3, -0.8),
                position_xyz_max=(TARGET_X + 0.3, TARGET_Y + 0.3, -0.8),
                rotation_xyzw=(0.0, 0.0, -0.70711, 0.70711),
                yaw_jitter=0.06,
            )
        )
        assets.append(object)

        target_zone = self.asset_registry.get_asset_by_name("locomanip_target_zone")()
        target_zone.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(TARGET_X * 2.87 - 0.1, TARGET_Y - 0.1, -0.8),
                position_xyz_max=(TARGET_X * 2.87 + 0.1, TARGET_Y + 0.1, -0.8),
                rotation_xyzw=(0, 0, 0, 1),
                yaw_jitter=0.02,
            )
        )
        assets.append(target_zone)

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
                target_zone,
                drop_height=None,
                max_x_separation=0.3,
                max_y_separation=0.3,
                max_z_separation=0.01,
                upright_threshold=0.9,
                upright_z_axis_up=True,
                gripper_near_threshold=1.1,
                require_static=True,
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
        parser.add_argument(
            "--table", type=str, default="white_table", help="Table asset (use 'none' or '' to disable)"
        )
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--lift_height", type=float, default=0.2, help="Height to lift object for success (meters)")
