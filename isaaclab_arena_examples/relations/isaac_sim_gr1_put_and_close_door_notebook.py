# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Multi-object example: GR1 put-item-in-fridge-and-close-door with multiple objects on the counter.

Loads gr1_multi_object_put_and_close_door (one target + distractors), relation solver places them
with random reachable positions and NoCollision. Run with zero actions to verify the scene.
"""

# NOTE: When running as a notebook, first run this cell to launch the simulation app:
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%
# Optional: list knife block and toaster prim paths in the kitchen USD (to get correct paths for NoCollision).
# Run this cell to print the suffix to use in gr1_multi_object_put_and_close_door_environment.py.
def list_kitchen_knife_toaster_prims(layout_id: int = 1, style_id: int = 2):
    from lightwheel_sdk.loader import floorplan_loader
    from pxr import Usd

    usd_path = str(floorplan_loader.get_usd(scene="robocasakitchen", layout_id=layout_id, style_id=style_id, backend="robocasa")[0])
    print(f"Kitchen USD: {usd_path}\n")
    stage = Usd.Stage.Open(usd_path)
    default_prim = stage.GetDefaultPrim()
    default_path = str(default_prim.GetPath()) if default_prim else ""
    print(f"Default prim: {default_path}\n")
    for prim in stage.Traverse():
        path_str = str(prim.GetPath()).lower()
        if "knife" in path_str or "toaster" in path_str:
            full = str(prim.GetPath())
            suffix = full[len(default_path) + 1:] if full.startswith(default_path + "/") else full
            print(f"  path: {full}")
            print(f"  -> suffix for env: {suffix}")
            print(f"  -> prim_path: {{ENV_REGEX_NS}}/lightwheel_robocasa_kitchen/{suffix}\n")

# list_kitchen_knife_toaster_prims()  # uncomment only if you need to find knife/toaster paths for another layout

# %%

import torch
import tqdm

from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser


def run_gr1_multi_object_put_and_close_door(
    num_steps: int = 5000,
    reset_every_n_steps: int = 0,
    object_name: str = "ranch_dressing_bottle",
    distractor_objects: list[str] | None = None,
    embodiment: str = "gr1_pink",
    kitchen_style: int = 2,
):
    """Load the multi-object put-and-close-door environment and run with zero actions.

    Uses gr1_multi_object_put_and_close_door: one target object (picked into fridge) plus
    optional distractors on the counter; all placed with NoCollision. Smaller distractors
    (e.g. cracker_box, tomato_soup_can) reduce overlap issues after placement.

    Args:
        num_steps: Number of simulation steps to run.
        reset_every_n_steps: Reset every N steps (0 = no periodic reset).
        object_name: Target object to place in fridge.
        distractor_objects: Object names on the counter (default: cracker_box, tomato_soup_can).
        embodiment: Robot embodiment (e.g. gr1_pink).
        kitchen_style: Kitchen style ID.
    """
    if distractor_objects is None:
        distractor_objects = ["cracker_box", "tomato_soup_can"]
    args_parser = get_isaaclab_arena_environments_cli_parser()
    args_list = [
        "gr1_multi_object_put_and_close_door",
        "--object",
        object_name,
        "--embodiment",
        embodiment,
        "--kitchen_style",
        str(kitchen_style),
        "--distractor_objects",
        *distractor_objects,
    ]
    args_cli = args_parser.parse_args(args_list)

    arena_builder = get_arena_builder_from_cli(args_cli)
    env = arena_builder.make_registered()
    env.reset()

    for step in tqdm.tqdm(range(num_steps)):
        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        if reset_every_n_steps > 0 and (step + 1) % reset_every_n_steps == 0:
            env.reset()


# %%
# Run multi-object example (target + distractors on counter, random reachable positions):
run_gr1_multi_object_put_and_close_door()

# Optional: change target or distractors, e.g.:
# run_gr1_multi_object_put_and_close_door(object_name="jug", distractor_objects=["ranch_dressing_bottle", "sweet_potato"])

# %%
