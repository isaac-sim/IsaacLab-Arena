# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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


class Gr1KitchenSequentialTaskEnvironment(ExampleEnvironmentBase):

    name: str = "gr1_kitchen_sequential_task"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.sequential_composite_task_testing.open_door_task_sequential_test import OpenDoorTaskSequentialTest
        from isaaclab_arena.tasks.sequential_composite_task_testing.pick_task_sequential_test import PickTaskSequentialTest
        from isaaclab_arena.tasks.sequential_composite_task_testing.gr1_kitchen_sequential_task import Gr1KitchenSequentialTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("kitchen")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        assets = [background, microwave, pick_up_object]
        assert args_cli.embodiment in ["gr1_pink", "gr1_joint"], "Invalid GR1T2 embodiment {}".format(
            args_cli.embodiment
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.set_initial_pose(Pose(position_xyz=(-0.34, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        microwave.set_initial_pose(
            Pose(
                position_xyz=(0.4, 0.16442, 0.20518),
                rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
            )
        )

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.1498, -0.26401, 0.0351),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Add the subtasks
        subtask_1 = OpenDoorTaskSequentialTest(openable_object=microwave, openness_threshold=0.6, reset_openness=0.15, episode_length_s=2.0)
        subtask_2 = PickTaskSequentialTest(pick_up_object=pick_up_object, background_scene=background, min_z=0.2, episode_length_s=2.0)

        # Compose the scene
        scene = Scene(assets=assets)

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=Gr1KitchenSequentialTask(subtasks=[subtask_1, subtask_2], openable_object=microwave),
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="red_beaker")
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
        # Note (xinjieyao, 2025.10.06): Add the embodiment argument for PINK IK EEF control or Joint positional control
        parser.add_argument("--embodiment", type=str, default="gr1_pink")
