# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse

from isaaclab_arena.examples.example_environments.example_environment_base import ExampleEnvironmentBase


class GalileoG1LocomanipPickAndPlaceEnvironment(ExampleEnvironmentBase):

    name: str = "galileo_g1_locomanip_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.g1_locomanip_pick_and_place_task import G1LocomanipPickAndPlaceTask
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("galileo_locomanip")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        blue_sorting_bin = self.asset_registry.get_asset_by_name("blue_sorting_bin")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.5785, 0.18, 0.0707),
                rotation_wxyz=(0.0, 0.0, 1.0, 0.0),
            )
        )
        blue_sorting_bin.set_initial_pose(
            Pose(
                position_xyz=(-0.2450, -1.6272, -0.2641),
                rotation_wxyz=(0.0, 0.0, 0.0, 1.0),
            )
        )
        embodiment.set_initial_pose(Pose(position_xyz=(0.0, 0.18, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

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
            action_cfg.g1_action.navigation_subgoals = [
                ([0.18, 0.18, 0.0], False),
                ([0.18, 0.18, -1.78], True),
                ([-0.0955, -1.1070, -1.78], False),
                ([-0.0955, -1.1070, -1.78], False),
            ]

        scene = Scene(assets=[background, pick_up_object, blue_sorting_bin])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1LocomanipPickAndPlaceTask(pick_up_object, blue_sorting_bin, background, episode_length_s=20.0),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="brown_box")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)
