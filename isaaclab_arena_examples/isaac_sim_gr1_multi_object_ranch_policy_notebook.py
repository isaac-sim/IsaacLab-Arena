# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# %%
from __future__ import annotations

# pyright: reportArgumentType=false, reportCallIssue=false, reportAttributeAccessIssue=false

"""Multi-object put-and-close-door with ranch bottle policy.

Spawns multiple objects on the kitchen counter (ranch dressing bottle as target +
distractors). Run with zero actions to verify the scene, or use the policy_runner
command below to evaluate the ranch bottle policy in this multi-object setting.
"""

# NOTE: When running as a notebook, first run this cell to launch the simulation app:
import pinocchio  # noqa: F401
from isaaclab.app import AppLauncher

print("Launching simulation app once in notebook")
simulation_app = AppLauncher()

# %%

import torch
import tqdm

from isaaclab_arena_environments.cli import (
    ExampleEnvironments,
    get_arena_builder_from_cli,
    get_isaaclab_arena_environments_cli_parser,
)
from isaaclab_arena_environments.gr1_multi_object_ranch_policy_environment import (
    GR1MultiObjectRanchPolicyEnvironment,
)

# Register the new environment so the CLI parser includes it (no modification to cli.py on disk).
ExampleEnvironments[GR1MultiObjectRanchPolicyEnvironment.name] = GR1MultiObjectRanchPolicyEnvironment


def run_gr1_multi_object_ranch_policy(
    num_steps: int = 5000,
    reset_every_n_steps: int = 0,
    object_name: str = "ranch_dressing_bottle",
    distractor_objects: list[str] | None = None,
    embodiment: str = "gr1_joint",
    kitchen_style: int = 2,
):
    """Load the multi-object ranch policy environment and run with zero actions.

    Uses gr1_multi_object_ranch_policy: target object (ranch bottle) + distractors
    on the counter, placed with NoCollision. To run the actual policy, use the
    policy_runner command in the docstring below.

    Args:
        num_steps: Number of simulation steps to run.
        reset_every_n_steps: Reset every N steps (0 = no periodic reset).
        object_name: Target object to place in fridge.
        distractor_objects: Object names on the counter (default: cracker_box, tomato_soup_can).
        embodiment: Robot embodiment (e.g. gr1_joint).
        kitchen_style: Kitchen style ID.
    """
    if distractor_objects is None:
        distractor_objects = ["cracker_box", "tomato_soup_can"]
    args_parser = get_isaaclab_arena_environments_cli_parser()
    args_list = [
        "gr1_multi_object_ranch_policy",
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
# Run multi-object scene (ranch bottle + distractors on counter):
run_gr1_multi_object_ranch_policy()

# %%
# To run the ranch bottle *policy* in this multi-object environment, use policy_runner
# with --environment so the new env is loaded (no changes to existing cli.py):
#
#   python isaaclab_arena/evaluation/policy_runner.py \
#     --policy_type isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy \
#     --policy_config_yaml_path isaaclab_arena_gr00t/policy/config/gr1_manip_ranch_bottle_gr00t_closedloop_config.yaml \
#     --num_steps 2000 \
#     --enable_cameras \
#     --environment isaaclab_arena_environments.gr1_multi_object_ranch_policy_environment:GR1MultiObjectRanchPolicyEnvironment \
#     gr1_multi_object_ranch_policy \
#     --object ranch_dressing_bottle \
#     --distractor_objects cracker_box tomato_soup_can \
#     --embodiment gr1_joint
#
# Optional: change target or distractors in the notebook, e.g.:
# run_gr1_multi_object_ranch_policy(object_name="ranch_dressing_bottle", distractor_objects=["cracker_box", "tomato_soup_can"])
