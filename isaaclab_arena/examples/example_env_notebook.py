# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%

import torch
import tqdm

import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()


# %%
from isaaclab_arena.utils.reload_modules import reload_arena_modules

reload_arena_modules()
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

args_parser = get_isaaclab_arena_environments_cli_parser()

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
from isaaclab_arena.utils.isaaclab_utils.simulation_app import teardown_simulation_app

teardown_simulation_app(suppress_exceptions=False, make_new_stage=True)
