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


class Gr1OpenMicrowaveEnvironment(ExampleEnvironmentBase):

    name: str = "gr1_open_microwave"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacArenaEnvironment:
        from isaac_arena.embodiments.gr1t2.gr1t2 import GR1T2Embodiment
        from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
        from isaac_arena.geometry.pose import Pose
        from isaac_arena.scene.scene import Scene
        from isaac_arena.tasks.open_door_task import OpenDoorTask

        background = self.asset_registry.get_asset_by_name("packing_table")()
        microwave = self.asset_registry.get_asset_by_name("microwave")()
        assets = [background, microwave]

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Put the microwave on the packing table.
        microwave_pose = Pose(
            position_xyz=(0.8, -0.00586, 0.22773),
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

        isaac_arena_environment = IsaacArenaEnvironment(
            name=self.name,
            embodiment=GR1T2Embodiment(),
            scene=scene,
            task=OpenDoorTask(microwave, openness_threshold=0.8, reset_openness=0.2),
            teleop_device=teleop_device,
        )

        return isaac_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default=None)
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
