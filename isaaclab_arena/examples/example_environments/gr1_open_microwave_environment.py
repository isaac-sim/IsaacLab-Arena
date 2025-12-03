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


class Gr1OpenMicrowaveEnvironment(ExampleEnvironmentBase):

    name: str = "gr1_open_microwave"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.open_door_task import OpenDoorTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        assets = [background, microwave]
        assert args_cli.embodiment in ["gr1_pink", "gr1_joint"], "Invalid GR1T2 embodiment {}".format(
            args_cli.embodiment
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.4, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Put the microwave on the packing table.
        microwave_pose = Pose(
            position_xyz=(0.4, -0.00586, 0.22773),
            rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
        )
        microwave.set_initial_pose(microwave_pose)

        # Optionally add another object
        if args_cli.object is not None:
            object = self.asset_registry.get_asset_by_name(args_cli.object)()
            object_pose = Pose(
                position_xyz=(0.466, -0.437, 0.154),
                rotation_wxyz=(0.5, -0.5, 0.5, -0.5),
            )
            object.set_initial_pose(object_pose)
            assets.append(object)

        # Compose the scene
        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2, episode_length_s=2.0),
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
