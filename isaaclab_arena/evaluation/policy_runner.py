# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import random
import torch
import tqdm
from importlib import import_module
from typing import TYPE_CHECKING, Any

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments
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


def rollout_policy(env, policy: "PolicyBase", num_steps: int | None, num_episodes: int | None) -> dict[str, Any]:
    assert num_steps is not None or num_episodes is not None, "Either num_steps or num_episodes must be provided"
    assert num_steps is None or num_episodes is None, "Only one of num_steps or num_episodes must be provided"

    try:
        obs, _ = env.reset()
        policy.reset()
        # set task description (could be None) from the task being evaluated
        policy.set_task_description(env.cfg.isaaclab_arena_env.task.get_task_description())

        # Setup progress bar based on num_steps or num_episodes
        if num_steps is not None:
            pbar = tqdm.tqdm(total=num_steps, desc="Steps", unit="step")
        else:
            pbar = tqdm.tqdm(total=num_episodes, desc="Episodes", unit="episode")

        num_episodes_completed = 0
        num_steps_completed = 0
        while True:
            with torch.inference_mode():
                actions = policy.get_action(env, obs)
                obs, _, terminated, truncated, _ = env.step(actions)

                if terminated.any() or truncated.any():
                    # Only reset policy for those envs that are terminated or truncated
                    print(
                        f"Resetting policy for terminated env_ids: {terminated.nonzero().flatten()}"
                        f" and truncated env_ids: {truncated.nonzero().flatten()}"
                    )
                    env_ids = (terminated | truncated).nonzero().flatten()
                    policy.reset(env_ids=env_ids)
                    # Break if number of episodes is reached
                    completed_episodes = env_ids.shape[0]
                    num_episodes_completed += completed_episodes
                    if num_episodes is not None:
                        pbar.update(completed_episodes)
                        if num_episodes_completed >= num_episodes:
                            break
                # Break if number of steps is reached
                num_steps_completed += 1
                if num_steps is not None:
                    pbar.update(1)
                    if num_steps_completed >= num_steps:
                        break

        pbar.close()

    except Exception as e:
        pbar.close()
        raise RuntimeError(f"Error rolling out policy: {e}")

    else:
        # only compute metrics if env has metrics registered
        if hasattr(env.cfg, "metrics"):
            # NOTE(xinjieyao, 2025-10-07): lazy import to prevent app stalling caused by omni.kit
            from isaaclab_arena.metrics.metrics import compute_metrics

            metrics = compute_metrics(env)
            return metrics
        return None


def main():
    """Script to run an IsaacLab Arena environment with a zero-action agent."""
    args_parser = get_isaaclab_arena_cli_parser()
    # We do this as the parser is shared between the example environment and policy runner
    args_cli, unknown = args_parser.parse_known_args()

    # Start the simulation app
    with SimulationAppContext(args_cli):
        # Get the policy-type flag before preceding to other arguments
        add_policy_runner_arguments(args_parser)
        args_cli, _ = args_parser.parse_known_args()

        # Get the policy class from the policy type
        policy_cls = get_policy_cls(args_cli.policy_type)
        print(f"Requested policy type: {args_cli.policy_type} -> Policy class: {policy_cls}")

        # Add the example environment arguments + policy-related arguments to the parser
        args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
        args_parser = policy_cls.add_args_to_parser(args_parser)
        args_cli = args_parser.parse_args()

        # Build scene
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        if args_cli.seed is not None:
            env.seed(args_cli.seed)
            torch.manual_seed(args_cli.seed)
            np.random.seed(args_cli.seed)
            random.seed(args_cli.seed)

        # Create the policy from the arguments
        policy = policy_cls.from_args(args_cli)

        # Simulation length.
        if policy.has_length():
            num_steps = policy.length()
            num_episodes = None
        else:
            if args_cli.num_steps is not None:
                num_steps = args_cli.num_steps
                num_episodes = None
                print(f"Simulation length: {num_steps} steps")
            elif args_cli.num_episodes is not None:
                num_steps = None
                num_episodes = args_cli.num_episodes
                print(f"Simulation length: {num_episodes} episodes")
            else:
                raise ValueError("Either num_steps or num_episodes must be provided")

        metrics = rollout_policy(env, policy, num_steps, num_episodes)
        if metrics is not None:
            print(f"Metrics: {metrics}")

        # Close the environment.
        env.close()


if __name__ == "__main__":
    main()
