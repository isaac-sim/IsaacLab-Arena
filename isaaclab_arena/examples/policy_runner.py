# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import random
import torch
import tqdm
from importlib import import_module
from typing import TYPE_CHECKING

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.examples.policy_runner_cli import add_policy_runner_arguments
from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext
from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

if TYPE_CHECKING:
    from isaaclab_arena.policy.policy_base import PolicyBase


def get_policy_cls(policy_type: str) -> type["PolicyBase"]:
    """Get the policy class for the given policy type name.

    Note that this function:
    - first: checks for a registered policy type in the PolicyRegistry
    - if not found, it tries to dynamically import the policy class, treating
      the policy_type argument as a string representing the module path and class name.

    """
    from isaaclab_arena.assets.asset_registry import PolicyRegistry

    policy_registry = PolicyRegistry()
    if policy_registry.is_registered(policy_type):
        return policy_registry.get_policy(policy_type)
    else:
        print(f"Policy {policy_type} is not registered. Dynamically importing from path: {policy_type}")
        assert "." in policy_type, (
            "policy_type must be a dotted Python import path of the form 'module.submodule.ClassName', got:"
            f" {policy_type}"
        )
        # Dynamically import the class from the string path
        module_path, class_name = policy_type.rsplit(".", 1)
        module = import_module(module_path)
        policy_cls = getattr(module, class_name)
        return policy_cls


def main():
    """Script to run an IsaacLab Arena environment with a zero-action agent."""
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Add policy-related arguments to the parser
        args_parser = setup_policy_argument_parser(args_parser)
        args_cli = args_parser.parse_args()
        # Build scene
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        obs, _ = env.reset()

        # NOTE(xinjieyao, 2025-09-29): General rule of thumb is to have as many non-standard python
        # library imports after app launcher as possible, otherwise they will likely stall the sim
        # app. Given current SimulationAppContext setup, use lazy import to handle policy-related
        # deps inside create_policy() function to bringup sim app.
        policy, num_steps = create_policy(args_cli)
        # set task description (could be None) from the task being evaluated
        policy.set_task_description(env.cfg.isaaclab_arena_env.task.get_task_description())

        # NOTE(xinjieyao, 2025-10-07): lazy import to prevent app stalling caused by omni.kit
        from isaaclab_arena.metrics.metrics import compute_metrics

        for _ in tqdm.tqdm(range(num_steps)):
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    # only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)

        metrics = compute_metrics(env)
        print(f"Metrics: {metrics}")

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
