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


class PegInsertEnvironment(ExampleEnvironmentBase):

    name: str = "factory_peg_insert"

    def get_env(self, args_cli: argparse.Namespace):  # -> IsaacLabArenaEnvironment:
        import isaaclab.sim as sim_utils
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import configclass
        import isaaclab.sim as sim_utils
        from isaaclab_arena_environments import mdp
        from isaaclab_arena.tasks.assembly_task import AssemblyTask

        from isaaclab_arena.assets.background_library import Table
        from isaaclab_arena.assets.object_library import Peg, Hole
        from isaaclab_arena.assets.object_library import Light
        from isaaclab_arena.embodiments.franka.franka import FrankaEmbodiment
        from isaaclab_arena.embodiments.franka.franka import FRANKA_PANDA_ASSEMBLY_HIGH_PD_CFG

        # Set pinocchio configuration on the passed args_cli.
        # For peg insert task, we need to test the task with pinocchio disabled due to the "peg" and "hole" assets are not compatible with pinocchio.
        args_cli.enable_pinocchio = False

        background = self.asset_registry.get_asset_by_name(args_cli.background)()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination_object = self.asset_registry.get_asset_by_name(args_cli.destination_object)()       
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

        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(0.45, 0.0, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        destination_object.set_initial_pose(
            Pose(
                position_xyz=(0.45, 0.1, 0.0),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )

        scene = Scene(assets=[background, pick_up_object, destination_object, light])

        task=AssemblyTask(
            task_description="Assemble the peg with the hole",
            fixed_asset=pick_up_object,
            held_asset=destination_object,
            auxiliary_asset_list=[],
            background_scene=background,
            pose_range={
                "x": (0.2, 0.6),
                "y": (-0.20, 0.20),
                "z": (0.0, 0.0),
                "yaw": (-1.0, 1.0),
            },
            min_separation=0.1,
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
        parser.add_argument("--object", type=str, default="peg")
        parser.add_argument("--destination_object", type=str, default="hole")
        parser.add_argument("--background", type=str, default="table")
        parser.add_argument("--embodiment", type=str, default="franka")
        parser.add_argument("--teleop_device", type=str, default=None)
