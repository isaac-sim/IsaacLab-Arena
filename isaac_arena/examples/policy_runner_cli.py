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

from isaac_arena.examples.example_environments.cli import get_isaac_arena_example_environment_cli_parser
from isaac_arena.policy.policy_base import PolicyBase
from isaac_arena.policy.replay_action_policy import ReplayActionPolicy
from isaac_arena.policy.zero_action_policy import ZeroActionPolicy
from isaac_arena.policy.gr00t.replay_lerobot_action_policy import ReplayLerobotActionPolicy
from isaac_arena.policy.gr00t.policy_config import LerobotReplayActionPolicyConfig


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
    replay_lerobot_group = parser.add_argument_group("Replay Lerobot Action Policy", "Arguments for replay Lerobot action policy")
    replay_lerobot_group.add_argument(
        "--replay_lerobot_dataset_path",
        type=str,
        help="Path to the Lerobot file containing the episode (required with --policy_type replay_lerobot)",
    )



def setup_policy_argument_parser() -> argparse.ArgumentParser:
    """Set up and configure the argument parser with all policy-related arguments."""
    # Get the base parser from Isaac Arena
    args_parser = get_isaac_arena_example_environment_cli_parser()

    args_parser.add_argument(
        "--policy_type",
        type=str,
        choices=["zero_action", "replay", "replay_lerobot"],
        required=True,
        help="Type of policy to use: 'zero_action' or 'replay' or 'replay_lerobot'",
    )

    # Add policy-specific argument groups
    add_zero_action_arguments(args_parser)
    add_replay_arguments(args_parser)
    add_replay_lerobot_arguments(args_parser)
    parsed_args = args_parser.parse_args()

    if parsed_args.policy_type == "replay" and parsed_args.replay_file_path is None:
        raise ValueError("--replay_file_path is required when using --policy_type replay")
    if parsed_args.policy_type == "replay_lerobot" and parsed_args.replay_lerobot_dataset_path is None:
        raise ValueError("--replay_lerobot_dataset_path is required when using --policy_type replay_lerobot")

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
        # init dataset_path member in policy_cfg class
        policy_cfg = LerobotReplayActionPolicyConfig(dataset_path=args.replay_lerobot_dataset_path)
        policy = ReplayLerobotActionPolicy(policy_cfg)
        num_steps = args.num_steps
    else:
        raise ValueError(f"Unknown policy type: {args.type}")
    return policy, num_steps
