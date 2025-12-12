# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class GearMeshEnvironment(ExampleEnvironmentBase):
    """
    Gear mesh assembly environment with 4 gears:
    - gear_base: gear base (to be meshed with)
    - medium_gear: medium gear (to be picked and assembled)
    - small_gear: small reference gear
    - large_gear: large reference gear
    """

    name: str = "factory_gear_mesh"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import configclass

        import isaaclab.sim as sim_utils
        from isaaclab_arena.tasks.assembly_task import AssemblyTask
        from isaaclab_arena.assets.background_library import Table
        from isaaclab_arena.assets.object_library import GearBase, MediumGear, SmallGear, LargeGear
        from isaaclab_arena.assets.object_library import Light
        from isaaclab_arena_environments import mdp
        from isaaclab_arena.embodiments.franka.franka import FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG


        # Get assets from registry
        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        gear_base = self.asset_registry.get_asset_by_name("gear_base")()
        medium_gear = self.asset_registry.get_asset_by_name("medium_gear")()
        small_gear = self.asset_registry.get_asset_by_name("small_gear")()
        large_gear = self.asset_registry.get_asset_by_name("large_gear")()
        light_spawner_cfg = sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0)
        light = self.asset_registry.get_asset_by_name("light")(spawner_cfg=light_spawner_cfg)
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)
        embodiment.scene_config.robot = FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None
        
        background.set_initial_pose(
            Pose(
                position_xyz=(0.55, 0.0, 0.0), 
                rotation_wxyz=(0.707, 0, 0, 0.707)
            )
        )

        # Set initial poses for all 4 gears
        # Based on reference FactoryGearMeshSceneCfg configuration
        gear_base.set_initial_pose(
            Pose(
                position_xyz=(0.6, 0.0, 0.0),  # Gear base position
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        medium_gear.set_initial_pose(
            Pose(
                position_xyz=(0.5, 0.2, 0.0),  # Medium gear to be assembled
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        small_gear.set_initial_pose(
            Pose(
                position_xyz=(0.6, 0.0, 0.0),  # Small reference gear
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        large_gear.set_initial_pose(
            Pose(
                position_xyz=(0.6, 0.0, 0.0),  # Large reference gear
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        # Create scene with all 4 gears and background
        scene = Scene(assets=[background, gear_base, medium_gear, small_gear, large_gear, light])

        # Create gear mesh task
        task = AssemblyTask(
            fixed_asset=gear_base,
            held_asset=medium_gear,
            auxiliary_asset_list=[small_gear, large_gear],
            background_scene=background,
            pose_range={
                "x": (0.25, 0.6), 
                "y": (-0.20, 0.20), 
                "z": (0.0, 0.0), 
                "yaw": (-1.0, 1.0)
            },
            min_separation=0.18,
            randomization_mode=1,
        )

        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=task,
            teleop_device=teleop_device,
            env_cfg_callback=mdp.assembly_env_cfg_callback,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        """Add CLI arguments for gear mesh environment."""
        parser.add_argument("--background", type=str, default="table", help="Background scene (table)")
        parser.add_argument("--embodiment", type=str, default="franka", help="Robot embodiment")
        parser.add_argument(
            "--teleop_device", type=str, default=None, help="Teleoperation device (e.g., keyboard, spacemouse)"
        )
