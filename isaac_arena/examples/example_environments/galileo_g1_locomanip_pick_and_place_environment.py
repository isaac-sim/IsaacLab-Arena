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


class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "galileo_g1_locomanip_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacArenaEnvironment:
        from isaac_arena.environments.isaac_arena_environment import IsaacArenaEnvironment
        from isaac_arena.geometry.pose import Pose
        from isaac_arena.scene.scene import Scene
        from isaac_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask

        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        blue_sorting_bin = self.asset_registry.get_asset_by_name("blue_sorting_bin")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)()

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.55, -0.04, 0.2259),
                rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
            )
        )
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(-0.2450, -1.5815, -0.2641),
                rotation_wxyz=(0.0, 0, 0, 10),
            )
        )

        # NOTE(alexmillane, 2025.09.08): This is a sub-optimal destination location
        # in the room. I'd like to use the bottom shelf, however, the whole shelf is
        # a single prim and therefore I cannot pick out the bottom shelf specifically.
        # NOTE(alexmillane, 2025.09.08): I've also had to apply the rigid body API to
        # the lid via the UI.
        # TODO(alexmillane, 2025.09.08): Separate the self into prims so we can reference
        # the bottom shelf specifically.
        # destination_location = ObjectReference(
        #     name="destination_location",
        #     prim_path="{ENV_REGEX_NS}/galileo/BackgroundAssets/bins/small_bin_grid_01/lid",
        #     parent_asset=background,
        # )

        scene = Scene(assets=[background, pick_up_object, blue_sorting_bin])
        isaac_arena_environment = IsaacArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipPickAndPlaceTask(pick_up_object, blue_sorting_bin, background),
            teleop_device=teleop_device,
        )
        return isaac_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="power_drill")
        parser.add_argument("--embodiment", type=str, default="gr1")
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
