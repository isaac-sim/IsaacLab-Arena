# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from isaac_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class SequentialTaskExampleEnvironment(ExampleEnvironmentBase):

    name: str = "sequential_task_example"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacArenaEnvironment:
        from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
        from isaac_arena.scene.scene import Scene
        from isaac_arena.tasks.open_door_task import OpenDoorTask
        from isaac_arena.tasks.press_button_task import PressButtonTask
        from isaac_arena.tasks.sequential_task import SequentialTask
        from isaac_arena.utils.pose import Pose

        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

        background = self.asset_registry.get_asset_by_name("packing_table")()

        # Put a toaster on the packing table.
        pressable_object = self.asset_registry.get_asset_by_name("toaster")()
        pressable_object_pose = Pose(position_xyz=(0.7, 0.4, 0.19), rotation_wxyz=(0.7071, 0.0, 0.0, -0.7071))
        pressable_object.set_initial_pose(pressable_object_pose)

        openable_object = self.asset_registry.get_asset_by_name("microwave")()

        press_button_task = PressButtonTask(pressable_object, reset_pressedness=0.8)
        open_door_task = OpenDoorTask(openable_object, openness_threshold=0.8, reset_openness=0.2)

        sequential_task = SequentialTask([press_button_task, open_door_task])

        assets = [background, pressable_object, openable_object]

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Compose the scene
        scene = Scene(assets=assets)

        isaac_arena_environment = IsaacArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=sequential_task,
            teleop_device=teleop_device,
        )
        return isaac_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=None)
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
        parser.add_argument("--embodiment", type=str, default="franka")
