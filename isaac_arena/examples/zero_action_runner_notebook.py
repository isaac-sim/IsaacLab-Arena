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

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


from isaac_arena.examples.example_environments.cli import (
    get_arena_builder_from_cli,
    get_isaac_arena_example_environment_cli_parser,
)

args_parser = get_isaac_arena_example_environment_cli_parser()

# GR1 Open Microwave
args_cli = args_parser.parse_args([
    "gr1_open_microwave",
    "--object",
    "cracker_box",
])

# Pick and Place
# args_cli = args_parser.parse_args([
#     "kitchen_pick_and_place",
#     "--object",
#     "cracker_box",
#     "--background",
#     "kitchen",
#     "--embodiment",
#     "franka",
# ])

arena_builder = get_arena_builder_from_cli(args_cli)
env = arena_builder.make_registered()
env.reset()

# %%

# Run some zero actions.
NUM_STEPS = 100
for _ in tqdm.tqdm(range(NUM_STEPS)):
    with torch.inference_mode():
        actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        env.step(actions)


# %%
