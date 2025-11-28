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


class PressButtonEnvironment(ExampleEnvironmentBase):

    name: str = "press_button"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_base import ObjectType
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.press_button_task import PressButtonTaskRL
        from isaaclab_arena.utils.pose import Pose

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

        background = self.asset_registry.get_asset_by_name("packing_table")()
        press_object = self.asset_registry.get_asset_by_name("coffee_machine")()

        # Reference this and add it to the scene to use it in the reward function
        pressable_object_prim = ObjectReference(
            name="pressable_object_prim",
            prim_path="{ENV_REGEX_NS}/" + press_object.name + "/" + press_object.pressable_prim_path,
            parent_asset=press_object,
            object_type=ObjectType.RIGID,
        )

        embodiment_end_effector_frame = ObjectReference(
            name="embodiment_end_effector_frame",
            prim_path=embodiment.scene_config.robot.prim_path + "/" + embodiment.rl_end_effector_frame,
            parent_asset=embodiment,
            object_type=ObjectType.RIGID,
        )

        assets = [background, press_object, pressable_object_prim, embodiment_end_effector_frame]

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Put the coffee_machine on the packing table.
        press_object_pose = Pose(position_xyz=(0.7, 0.4, 0.19), rotation_wxyz=(0.7071, 0.0, 0.0, -0.7071))
        press_object.set_initial_pose(press_object_pose)

        # Compose the scene
        scene = Scene(assets=assets)

        task = PressButtonTaskRL(
            press_object,
            pressable_object_prim,
            embodiment_end_effector_frame,
            reset_pressedness=0.8,
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
        parser.add_argument("--embodiment", type=str, default="franka")
