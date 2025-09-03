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
from typing import TYPE_CHECKING

from isaac_arena.examples.example_environments.example_environment_base import (
    ExampleEnvironmentBase,
    add_argument_if_missing,
)

if TYPE_CHECKING:
    from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment


class PickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacArenaEnvironment:
        from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
        from isaac_arena.geometry.pose import Pose
        from isaac_arena.scene.scene import Scene
        from isaac_arena.tasks.pick_and_place_task import PickAndPlaceTask

        assert args_cli.background is not None
        assert args_cli.object is not None
        assert args_cli.embodiment is not None

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.4, 0.0, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object])
        isaac_arena_environment = IsaacArenaEnvironment(
            name="pick_and_place",
            embodiment=embodiment,
            scene=scene,
            task=PickAndPlaceTask(pick_up_object, background),
            teleop_device=teleop_device,
        )
        return isaac_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        add_argument_if_missing(parser, "--object", type=str, default=None)
        add_argument_if_missing(parser, "--background", type=str, default=None)
        add_argument_if_missing(parser, "--embodiment", type=str, default=None)
        add_argument_if_missing(parser, "--teleop_device", type=str, default=None)
