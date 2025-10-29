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

from isaaclab_arena.examples.example_environments.cli import get_isaaclab_arena_example_environment_cli_parser
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena.policy.replay_action_policy import ReplayActionPolicy
from isaaclab_arena.policy.zero_action_policy import ZeroActionPolicy


def add_zero_action_arguments(parser: argparse.ArgumentParser) -> None:
    """Add zero action policy specific arguments to the parser."""
    zero_action_group = parser.add_argument_group("Zero Action Policy", "Arguments for zero action policy")
    zero_action_group.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to run the policy for (only used with zero action policy)",
    )


def add_replay_arguments(parser: argparse.ArgumentParser) -> None:
    """Add replay action policy specific arguments to the parser."""
    replay_group = parser.add_argument_group("Replay Action Policy", "Arguments for replay action policy")
    replay_group.add_argument(
        "--replay_file_path",
        type=str,
        help="Path to the HDF5 file containing the episode (required with --policy_type replay)",
    )
    replay_group.add_argument(
        "--episode_name",
        type=str,
        default=None,
        help=(
            "Name of the episode to replay. If not provided, the first episode will be"
            "replayed (only used with --policy_type replay)"
        ),
    )


def add_replay_lerobot_arguments(parser: argparse.ArgumentParser) -> None:
    """Add replay Lerobot action policy specific arguments to the parser."""
    replay_lerobot_group = parser.add_argument_group(
        "Replay Lerobot Action Policy", "Arguments for replay Lerobot dataset action policy"
    )
    replay_lerobot_group.add_argument(
        "--config_yaml_path",
        type=str,
        help="Path to the Lerobot action policy config YAML file (required with --policy_type replay_lerobot)",
    )
    replay_lerobot_group.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of steps to run the policy for (only used with --policy_type replay_lerobot)",
    )
    replay_lerobot_group.add_argument(
        "--trajectory_index",
        type=int,
        default=0,
        help="Index of the trajectory to run the policy for (only used with --policy_type replay_lerobot)",
    )


def add_gr00t_closedloop_arguments(parser: argparse.ArgumentParser) -> None:
    """Add gr00t closedloop policy specific arguments to the parser."""
    gr00t_closedloop_group = parser.add_argument_group(
        "Gr00t Closedloop Policy", "Arguments for gr00t closedloop policy"
    )
    gr00t_closedloop_group.add_argument(
        "--policy_config_yaml_path",
        type=str,
        help="Path to the Gr00t closedloop policy config YAML file (required with --policy_type gr00t_closedloop)",
    )


def setup_policy_argument_parser(args_parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    """Set up and configure the argument parser with all policy-related arguments."""
    # Get the base parser from IsaacLab Arena
    args_parser = get_isaaclab_arena_example_environment_cli_parser(args_parser)

    args_parser.add_argument(
        "--policy_type",
        type=str,
        choices=["zero_action", "replay", "replay_lerobot", "gr00t_closedloop"],
        required=True,
        help="Type of policy to use: 'zero_action' or 'replay' or 'replay_lerobot' or 'gr00t_closedloop'",
    )

    # Add policy-specific argument groups
    add_zero_action_arguments(args_parser)
    add_replay_arguments(args_parser)
    add_replay_lerobot_arguments(args_parser)
    add_gr00t_closedloop_arguments(args_parser)
    parsed_args = args_parser.parse_args()

    if parsed_args.policy_type == "replay" and parsed_args.replay_file_path is None:
        raise ValueError("--replay_file_path is required when using --policy_type replay")
    if parsed_args.policy_type == "replay_lerobot" and parsed_args.config_yaml_path is None:
        raise ValueError("--config_yaml_path is required when using --policy_type replay_lerobot")
    if parsed_args.policy_type == "gr00t_closedloop" and parsed_args.policy_config_yaml_path is None:
        raise ValueError("--policy_config_yaml_path is required when using --policy_type gr00t_closedloop")
    return args_parser


def create_policy(args: argparse.Namespace) -> tuple[PolicyBase, int]:
    """Create the appropriate policy based on the arguments and return (policy, num_steps)."""
    if args.policy_type == "replay":
        policy = ReplayActionPolicy(args.replay_file_path, args.episode_name)
        num_steps = len(policy)
    elif args.policy_type == "zero_action":
        policy = ZeroActionPolicy()
        num_steps = args.num_steps
    elif args.policy_type == "replay_lerobot":
        # NOTE(xinjie.yao, 2025-09-28): lazy import to prevent app stalling
        # due to import GR00T py dependencies that are conflicting with omni.kit
        # see functional import sequence here https://github.com/isaac-sim/IsaacLabEvalTasks/blob/main/scripts/evaluate_gn1.py#L38
        from isaaclab_arena_gr00t.replay_lerobot_action_policy import ReplayLerobotActionPolicy

        policy = ReplayLerobotActionPolicy(
            args.config_yaml_path, num_envs=args.num_envs, device=args.device, trajectory_index=args.trajectory_index
        )
        # Use custom max_steps if provided to optionally playing partial sequence in one trajectory
        if args.max_steps is not None:
            num_steps = args.max_steps
        else:
            num_steps = policy.get_trajectory_length(policy.get_trajectory_index())

    elif args.policy_type == "gr00t_closedloop":
        from isaaclab_arena_gr00t.gr00t_closedloop_policy import Gr00tClosedloopPolicy

        policy = Gr00tClosedloopPolicy(args.policy_config_yaml_path, num_envs=args.num_envs, device=args.device)
        num_steps = args.num_steps
    else:
        raise ValueError(f"Unknown policy type: {args.type}")
    return policy, num_steps
