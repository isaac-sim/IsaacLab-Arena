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


class OrcaG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "orca_g1_locomanip_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.g1_orca_task import G1OrcaTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("orca")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination_cart = self.asset_registry.get_asset_by_name("orca_cart")(
            kinematic_enabled=False,  # Dynamic: can be pushed by robot
            mass=1.0,                  # 1 kg - light cart
            linear_damping=0.005,       # Very low damping - easy to push
            angular_damping=0.005,      # Very low damping
        )
        plate = self.asset_registry.get_asset_by_name("orca_plate")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # TODO: Adjust these positions based on your ORCA scene layout
        background.set_initial_pose(Pose(position_xyz=(4.0, 0.0, -0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(1.0, 0.0, 0.1),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        plate.set_initial_pose(
            Pose(
                position_xyz=(1.0, -0.7, 0.0),  # Position plate near the box
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        destination_cart.set_initial_pose(
            Pose(
                position_xyz=(0.137, -1.05, -0.7875),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        if (
            args_cli.embodiment == "g1_wbc_pink"
            and hasattr(args_cli, "mimic")
            and args_cli.mimic
            and not hasattr(args_cli, "auto")
        ):
            # Patch the Mimic generate function for locomanip use case
            from isaaclab_arena.utils.locomanip_mimic_patch import patch_g1_locomanip_mimic

            patch_g1_locomanip_mimic()

            # Set navigation p-controller for locomanip use case
            action_cfg = embodiment.get_action_cfg()
            action_cfg.g1_action.use_p_control = True
            # Set nav subgoals (x,y,heading) and turning_in_place flag for G1 WBC Pink navigation p-controller
            # TODO: Adjust these navigation subgoals based on your ORCA scene layout
            action_cfg.g1_action.navigation_subgoals = [
                ([0.18, 0.18, 0.0], False),
                ([0.18, 0.18, -1.78], True),
                ([-0.0955, -1.1070, -1.78], False),
                ([-0.0955, -1.1070, -1.78], False),
            ]

        scene = Scene(assets=[background, pick_up_object, plate, destination_cart])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1OrcaTask(pick_up_object, destination_cart, background),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="brown_box")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
