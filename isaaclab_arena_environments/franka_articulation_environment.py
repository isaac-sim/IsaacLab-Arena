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


class FrankaPutAndCloseDrawerEnvironment(ExampleEnvironmentBase):
    """
    A sequential task environment with two subtasks:
    1. Pick and place object from top of cabinet into the drawer
    2. Close the drawer (goal is to close it to 5% or less)
    The drawer starts open and the goal is to place the object and then close it.
    """

    name = "franka_put_and_close_drawer"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.utils.pose import Pose

        # Get assets from registry
        cabinet = self.asset_registry.get_asset_by_name("cabinet")()
        cabinet.set_initial_pose(Pose(position_xyz=(0.6, 0.0, 0.4), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

        # Get the pick-up object (place it on top of the cabinet)
        pick_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        pick_object.set_initial_pose(
            Pose(
                position_xyz=(0.35, 0.0, 1.05),  # On top of cabinet
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Create a simple background object with object_min_z
        class SimpleBackground:
            def __init__(self, object_min_z: float):
                self.object_min_z = object_min_z

        minimal_background = SimpleBackground(object_min_z=0.05)

        # Create object reference to the drawer bottom as destination
        drawer_bottom = ObjectReference(
            parent_asset=cabinet,
            name="drawer_bottom",
            prim_path="{ENV_REGEX_NS}/cabinet/cabinet/drawer_bottom",
            object_type=ObjectType.RIGID,
        )

        # Get embodiment based on selection
        if args_cli.embodiment == "franka":
            embodiment = self.asset_registry.get_asset_by_name("franka")(enable_cameras=args_cli.enable_cameras)
        else:
            raise NotImplementedError(f"Embodiment {args_cli.embodiment} not supported")

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set robot initial pose
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(0.0, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Scene with cabinet, pick object, ground plane, and light
        ground_plane = self.asset_registry.get_asset_by_name("ground_plane")()
        light = self.asset_registry.get_asset_by_name("light")()
        scene = Scene(assets=[cabinet, pick_object, ground_plane, light])

        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=pick_object,
            destination_location=drawer_bottom,
            background_scene=minimal_background,
            task_description="Pick the object from the top of the cabinet and place it in the drawer.",
        )

        close_drawer_task = CloseDoorTask(
            openable_object=cabinet,
            closedness_threshold=0.05,
            reset_openness=0.3,
            task_description="Close the cabinet drawer.",
        )

        # Create a sequential task wrapper class
        class SequentialPutAndCloseDrawerTask(SequentialTaskBase):
            def __init__(self, subtasks, episode_length_s=None):
                super().__init__(subtasks=subtasks, episode_length_s=episode_length_s)

            def get_metrics(self):
                return []

            def get_mimic_env_cfg(self, arm_mode):
                return None

            def get_viewer_cfg(self):
                return self.subtasks[1].get_viewer_cfg()

        # Create the sequential task
        sequential_task = SequentialPutAndCloseDrawerTask(
            subtasks=[pick_and_place_task, close_drawer_task],
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
        parser.add_argument("--object", type=str, default="dex_cube", help="Object to pick and place in the drawer")
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment to use")
        parser.add_argument("--enable_cameras", action="store_true", default=False, help="Enable camera sensors")
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
