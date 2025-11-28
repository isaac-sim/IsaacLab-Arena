# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

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
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("packing_table")()
        pick_up_object = self.asset_registry.get_asset_by_name("tomato_soup_can")()

        assets = [background, pick_up_object]

        embodiment = self.asset_registry.get_asset_by_name("franka")()
        embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Put the microwave on the packing table.
        pick_up_object_pose = Pose(
            position_xyz=(0.4, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
        pick_up_object.set_initial_pose(pick_up_object_pose)

        embodiment_information = {
            "body_name": "panda_hand",
            "eef_prim_path": "{ENV_REGEX_NS}/Robot/panda_link0",
            "target_prim_path": "{ENV_REGEX_NS}/Robot/panda_hand",
            "target_frame_name": "end_effector",
            "target_offset": (0.0, 0.0, 0.1034),
        }

        # Compose the scene
        scene = Scene(assets=assets)

        task = LiftObjectTaskRL(
            pick_up_object,
            background,
            embodiment_information,
            minimum_height_to_lift=1.5,
            maximum_height_to_lift=2,
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
        parser.add_argument("--object", type=str, default=None)
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
        # Note (xinjieyao, 2025.10.06): Add the embodiment argument for PINK IK EEF control or Joint positional control
        parser.add_argument("--embodiment", type=str, default="gr1_pink")
