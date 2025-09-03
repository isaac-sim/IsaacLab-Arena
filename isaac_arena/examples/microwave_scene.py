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

# %%

import gymnasium as gym
import torch
import tqdm

import pinocchio  # noqa: F401

# Launching the simulation app
from isaaclab.app import AppLauncher

from isaac_arena.cli.isaac_arena_cli import get_isaac_arena_cli_parser

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


from isaac_arena.examples.example_environments.cli import get_env_cfg_from_cli

args_parser = get_isaac_arena_cli_parser()
# args_cli = args_parser.parse_args([
#     "--example_environment",
#     "gr1_open_microwave",
#     "--object",
#     "cracker_box",
# ])
args_cli = args_parser.parse_args([
    "--example_environment",
    "pick_and_place",
    "--object",
    "cracker_box",
    "--background",
    "kitchen_pick_and_place",
    "--embodiment",
    "franka",
])

cfg, name = get_env_cfg_from_cli(args_cli)
env = gym.make(name, cfg=cfg).unwrapped
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)


# %%
