# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class KitchenMoveObjectEnvironment(ExampleEnvironmentBase):

    name: str = "kitchen_move_object"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.move_object_task import MoveObjectTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        movable_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        movable_object.set_initial_pose(Pose(position_xyz=(-2, -0.1, 0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        scene = Scene(assets=[background, movable_object])
        move_object_task = MoveObjectTask(
            movable_object=movable_object,
            background_scene=background,
            displacement_threshold=args_cli.displacement_threshold,
            episode_length_s=args_cli.episode_length,
        )
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=move_object_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default="mobile_shelving_cart",
            help="Movable articulated object to push.",
        )
        parser.add_argument("--embodiment", type=str, default="gr1_joint")
        parser.add_argument(
            "--displacement_threshold",
            type=float,
            default=0.5,
            help="XY displacement (meters) to count as success.",
        )
        parser.add_argument(
            "--episode_length",
            type=float,
            default=10.0,
            help="Episode length in seconds.",
        )
        parser.add_argument("--teleop_device", type=str, default=None)
