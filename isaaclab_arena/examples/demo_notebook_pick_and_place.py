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


# This notebook is solely used for demo purposes.
# Demo steps
# 1. Launch the simulation app
# 2. Get the embodiment, background from the registry
# 3. Create a scene with the assets
# 4. Get the task
# 5. Create the Arena environment
# 6. Compile to a Arena Manager-based RL environment configuration and register it.
# 7. Run some zero actions or teleop the robot
# 8. Clean up the simulation context and close the environment
# 9. Repeat the process with different tasks, embodiments, backgrounds, etc.

# %%
# 1. Launch the simulation app

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%
# 2. Get the embodiment, background from the registry

# Reload the arena modules

from isaaclab_arena.assets.asset_registry import AssetRegistry
from isaaclab_arena.assets.object_reference import ObjectReference
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

asset_registry = AssetRegistry()

background = asset_registry.get_asset_by_name("galileo")() # Change background
embodiment = asset_registry.get_asset_by_name("gr1_joint")() # Change embodiment
object_of_interest = asset_registry.get_asset_by_name("cracker_box")() # For pick and place task

# Setup destination location according to the background
# # For kitchen pick and place
# destination_location = ObjectReference(
#             name="destination_location",
#             prim_path="{ENV_REGEX_NS}/kitchen/Cabinet_B_02",
#             parent_asset=background,
#             object_type=ObjectType.RIGID,
#         )
# For galileo pick and place
destination_location = ObjectReference(
            name="destination_location",
            prim_path="{ENV_REGEX_NS}/galileo/BackgroundAssets/bins/small_bin_grid_01/lid",
            parent_asset=background,
        )

# # For kitchen pick and place
# object_of_interest.set_initial_pose(Pose(position_xyz=(0.4, 0.0, 0.1), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
# For galileo pick and place
object_of_interest.set_initial_pose(
            Pose(
                position_xyz=(0.55, 0.0, 0.33),
                rotation_wxyz=(0.0, 0.0, 0.7071068, -0.7071068),
            )
        )

# %%
# 3. Create a scene with the assets

from isaaclab_arena.scene.scene import Scene

# For pick and place
scene = Scene(assets=[background, object_of_interest, destination_location])


# %%
# 4. Get the task

from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
from isaaclab_arena.tasks.open_door_task import OpenDoorTask

task = PickAndPlaceTask(object_of_interest, destination_location, background)

# %%
# 5. Create the Arena environment

from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment

isaaclab_arena_environment = IsaacLabArenaEnvironment(
    name="demo_environment",
    embodiment=embodiment,
    scene=scene,
    task=task,
    teleop_device=None,
)

# %%
# 6. Compile the environment to create the Arena Environment configuration

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder

args_cli = get_isaaclab_arena_cli_parser().parse_args([])
env_builder = ArenaEnvBuilder(isaaclab_arena_environment, args_cli)
env = env_builder.make_registered()
env.reset()

# %%
# 7. Run some zero actions or teleop the robot

# Run some zero actions.
NUM_STEPS = 20
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)

# %%
# 8. Clean up the simulation context and close the environment

env.close()
from isaaclab_arena.tests.utils.subprocess import safe_teardown

safe_teardown()

# %%
