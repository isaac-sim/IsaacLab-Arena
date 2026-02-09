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
        import math

        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_composite_tasks.franka_put_and_close_door_task import (
            FrankaPutAndCloseDoorTask,
        )
        from isaaclab_arena.utils.pose import Pose

        # Get assets
        background = self.asset_registry.get_asset_by_name("kitchen")()
        table_top_reference = ObjectReference(
            name="table_top_reference",
            prim_path="{ENV_REGEX_NS}/kitchen/Kitchen_Counter/TRS_Base/TRS_Static/Counter_Top_A",
            parent_asset=background,
        )
        table_top_reference.add_relation(IsAnchor())
        container = self.asset_registry.get_asset_by_name("microwave")()
        pick_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        teleop_device = None
        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()

        container.add_relation(On(table_top_reference, clearance_m=0.02))
        container.add_relation(AtPosition(x=0.4, y=0.0))
        container.add_relation(RotateAroundSolution(yaw_rad=-math.pi / 2))

        pick_object.add_relation(On(table_top_reference, clearance_m=0.02))
        pick_object.add_relation(AtPosition(x=0.225, y=-0.487))
        pick_object.add_relation(RandomAroundSolution(x_half_m=0.075, y_half_m=0.15))
        pick_object.add_relation(RotateAroundSolution(roll_rad=-math.pi / 2, yaw_rad=-math.pi / 2))

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
        scene = Scene(assets=[background, table_top_reference, container, pick_object])

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
            destination_object=container,
            destination_location=destination_ref,
            background_scene=background,
            task_description=task_description_pick,
        )

        sequential_task = FrankaPutAndCloseDoorTask(
            subtasks=[pick_and_place_task, close_door_task], openable_object=container
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
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
