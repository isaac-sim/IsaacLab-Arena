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

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase

# NOTE(alexmillane, 2025.09.04): There is an issue with type annotation in this file.
# We cannot annotate types which require the simulation app to be started in order to
# import, because this file is used to retrieve CLI arguments, so it must be imported
# before the simulation app is started.
# TODO(alexmillane, 2025.09.04): Fix this.


class LightwheelKitchenPotPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "lightwheel_kitchen_pot_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        from isaaclab_arena.assets.object_reference import ObjectReference
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("lightwheel_kitchen")()
        pot = self.asset_registry.get_asset_by_name("lightwheel_pot_51")()
        robot = self.asset_registry.get_asset_by_name("gr1_pink")(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Put the pot on the bench.
        pot.set_initial_pose(Pose(position_xyz=(1.06, -1.01, 0.00), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        # Put the pot above the stovetop to test the termination condition
        # pot.set_initial_pose(Pose(position_xyz=(2.21, -0.47, 1.36), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        stovetop = ObjectReference(
            name="stovetop",
            prim_path="{ENV_REGEX_NS}/lightwheel_kitchen/stovetop_main_group/Stovetop004",
            parent_asset=background,
        )

        # Compose the scene
        scene = Scene(assets=[background, pot, stovetop])

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=robot,
            scene=scene,
            task=PickAndPlaceTask(pot, stovetop, background),
            teleop_device=teleop_device,
        )

        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        # NOTE(alexmillane, 2025.09.04): We need a teleop device argument in order
        # to be used in the record_demos.py script.
        parser.add_argument("--teleop_device", type=str, default=None)
