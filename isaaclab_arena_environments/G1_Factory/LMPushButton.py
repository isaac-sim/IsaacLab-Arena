# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class LMPushButton(ExampleEnvironmentBase):

    name: str = "LMPushButton"

    def get_env(self, args_cli: argparse.Namespace):
        import math

        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        from .events import set_relative_initial_pose
        from .pose_utils import pose_range_from_quat
        from .press_button_task import G1FactoryPressButtonTask

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -0.015), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        assets = []
        CONTROL_BOX_X = 2
        CONTROL_BOX_Y = 1

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        assets.append(background)

        control_box = self.asset_registry.get_asset_by_name("control_box")()
        control_box.set_initial_pose(
            pose_range_from_quat(
                position_xyz_min=(CONTROL_BOX_X - 1, CONTROL_BOX_Y - 1.2, -0.79),
                position_xyz_max=(CONTROL_BOX_X - 0.2, CONTROL_BOX_Y - 0.2, -0.79),
                rotation_xyzw=(0.0, 0.0, 0.70711, 0.70711),
                yaw_jitter=0.0,
            )
        )
        assets.append(control_box)

        button = self.asset_registry.get_asset_by_name("button")()
        set_relative_initial_pose(
            obj=button,
            reference=control_box,
            offset_range={"x": (-0.3, -0.3), "y": (-0.48, -0.38), "z": (0.8, 0.8)},
            rotation_xyzw=(0.0, -0.70711, 0.0, 0.70711),
            yaw_jitter=0.0,
        )
        assets.append(button)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1FactoryPressButtonTask(
                button,
                reset_pressedness=0.8,
                gripper_far_threshold=0.2,
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
        parser.add_argument("--background", type=str, default="factory_room", help="Background scene")
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
