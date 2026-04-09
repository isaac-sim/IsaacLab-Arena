# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class G1AgileTabletopAppleToPlateEnvironment(ExampleEnvironmentBase):
    """G1 robot with WBC-AGILE policy doing tabletop apple-to-plate manipulation.

    The robot stands stationary at a table and moves an apple onto a plate.
    The AGILE whole-body-control policy handles balance while the upper body
    performs manipulation via direct joint control.
    """

    name: str = "g1_agile_tabletop_apple_to_plate"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("table")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination_location = self.asset_registry.get_asset_by_name("clay_plates_hot3d_robolab")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Position objects on the table surface within the robot's reach.
        # The Seattle Lab table surface is near z=0; objects get a small z offset above it.
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.15, 0.15, 0.05),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )
        destination_location.set_initial_pose(
            Pose(
                position_xyz=(0.15, -0.15, 0.02),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        # Robot stands behind the table facing forward.
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.4, 0.0, 0.0),
                rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object, destination_location])
        task = PickAndPlaceTask(
            pick_up_object=pick_up_object,
            destination_location=destination_location,
            background_scene=background,
            episode_length_s=30.0,
            task_description="Pick up the apple from the table and place it onto the plate.",
        )
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
        parser.add_argument("--object", type=str, default="apple_01_objaverse_robolab")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_agile_joint")
        parser.add_argument("--teleop_device", type=str, default=None)
