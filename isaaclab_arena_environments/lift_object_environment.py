# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class LiftObjectEnvironment(ExampleEnvironmentBase):

    name: str = "lift_object"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.lift_object_task import LiftObjectTaskRL
        from isaaclab_arena.utils.pose import Pose, PoseRange

        background = self.asset_registry.get_asset_by_name("table")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()

        # Add ground plane and light to the scene
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()

        assets = [background, pick_up_object, ground_plane, light]

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(concatenate_observation_terms=True)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set all positions
        background.set_initial_pose(Pose(position_xyz=(0.5, 0, 0), rotation_wxyz=(0.707, 0, 0, 0.707)))
        pick_up_object.set_initial_pose(PoseRange(position_xyz_min=(-0.1, -0.25, 0.0), position_xyz_max=(0.1, 0.25, 0.0)))
        ground_plane.set_initial_pose(Pose(position_xyz=(0.0, 0.0, -1.05)))

        # Compose the scene
        scene = Scene(assets=assets)

        task = LiftObjectTaskRL(
            pick_up_object,
            background,
            embodiment,
            minimum_height_to_lift=0.04,
            episode_length_s=5.0,
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
        parser.add_argument("--object", type=str, default="dex_cube")
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="franka")
