# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
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


class FrankaPutAndCloseDoorEnvironment(ExampleEnvironmentBase):
    """
    A sequential task environment with two subtasks:
    1. Pick and place object into the microwave
    2. Close the microwave door
    The microwave starts open, the robot places the object inside, then closes it.
    """

    name = "franka_put_and_close_door"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.utils.pose import Pose

        # Get assets
        background = self.asset_registry.get_asset_by_name("kitchen")()
        container = self.asset_registry.get_asset_by_name("microwave")()
        pick_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set initial poses
        container.set_initial_pose(
            Pose(
                position_xyz=(0.4, -0.00586, 0.22773),
                rotation_wxyz=(0.7071068, 0, 0, -0.7071068),
            )
        )

        pick_object.set_initial_pose(
            Pose(
                position_xyz=(0.2, -0.437, 0.154),
                rotation_wxyz=(0.5, -0.5, 0.5, -0.5),
            )
        )

        embodiment.set_initial_pose(
            Pose(
                position_xyz=(-0.3, 0.0, -0.5),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        if args_cli.embodiment == "franka":
            # Set Franka arm pose for kitchen setup
            embodiment.set_initial_joint_pose([0.0, -1.309, 0.0, -2.793, 0.0, 3.037, 0.740, 0.04, 0.04])

        # Create destination reference
        destination_ref = ObjectReference(
            name="microwave_disc",
            parent_asset=container,
            prim_path="{ENV_REGEX_NS}/microwave/Microwave039_Disc001",
            object_type=ObjectType.RIGID,
        )

        # Task descriptions
        task_description_pick = "Pick the object and place it into the microwave."
        task_description_close = "Close the microwave door."

        # Create scene
        scene = Scene(assets=[background, container, pick_object])

        # Create close door task
        close_door_task = CloseDoorTask(
            openable_object=container,
            closedness_threshold=0.05,
            reset_openness=0.9,
            task_description=task_description_close,
        )

        # Create pick and place task
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_location=destination_ref,
            background_scene=background,
            task_description=task_description_pick,
        )

        # Create a sequential task wrapper class
        class SequentialPutAndCloseDoorTask(SequentialTaskBase):
            def __init__(self, subtasks, episode_length_s=None):
                super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

            def get_metrics(self):
                return []

            def get_mimic_env_cfg(self, arm_mode):
                return None

            def get_viewer_cfg(self):
                return self.subtasks[1].get_viewer_cfg()

        # Create the sequential task
        sequential_task = SequentialPutAndCloseDoorTask(
            subtasks=[pick_and_place_task, close_door_task],
            episode_length_s=90.0,  # Episode for two subtasks
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=sequential_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="dex_cube", help="Object to pick and place in the microwave")
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment to use")
        parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
