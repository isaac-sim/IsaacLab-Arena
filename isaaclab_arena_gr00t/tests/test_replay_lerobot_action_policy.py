# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function
from isaaclab_arena_gr00t.tests.utils.constants import TestConstants as Gr00tTestConstants

HEADLESS = True
ENABLE_CAMERAS = True
NUM_STEPS = 10
# Only 1 traj in test data
TRAJECTORY_INDEX = 0


def _run_replay_lerobot_policy(
    simulation_app,
    config_yaml_path: str,
    environment: str,
    object_name: str,
    embodiment: str,
    max_steps: int,
    trajectory_index: int = 0,
) -> bool:
    """Run the replay lerobot action policy within a shared simulation app."""
    import sys

    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
    from isaaclab_arena.evaluation.policy_runner import get_policy_cls, rollout_policy
    from isaaclab_arena_environments.cli import get_arena_builder_from_cli, get_isaaclab_arena_environments_cli_parser

    try:
        # Get the policy class
        policy_type = "isaaclab_arena_gr00t.policy.replay_lerobot_action_policy.ReplayLerobotActionPolicy"
        policy_cls = get_policy_cls(policy_type)
        print(f"Requested policy type: {policy_type} -> Policy class: {policy_cls}")

        # Build argument list FIRST - needed before parser setup because
        # get_isaaclab_arena_environments_cli_parser internally calls parse_known_args()
        # which parses sys.argv
        arg_list = [
            "--policy_type",
            policy_type,
            "--config_yaml_path",
            str(config_yaml_path),
            "--max_steps",
            str(max_steps),
            "--trajectory_index",
            str(trajectory_index),
            "--headless",
            "--enable_cameras",
            # Environment subparser command and its specific args
            environment,
            "--object",
            object_name,
            "--embodiment",
            embodiment,
        ]

        # Temporarily set sys.argv so that internal parse_known_args() calls work correctly
        old_argv = sys.argv
        sys.argv = [sys.argv[0]] + arg_list

        try:
            # Build the arguments for the policy runner
            args_parser = get_isaaclab_arena_cli_parser()

            # Add environment and policy arguments
            from isaaclab_arena.evaluation.policy_runner_cli import add_policy_runner_arguments

            add_policy_runner_arguments(args_parser)
            args_parser = get_isaaclab_arena_environments_cli_parser(args_parser)
            args_parser = policy_cls.add_args_to_parser(args_parser)

            args_cli = args_parser.parse_args(arg_list)
        finally:
            sys.argv = old_argv

        # Build the environment
        arena_builder = get_arena_builder_from_cli(args_cli)
        env = arena_builder.make_registered()

        # Create the policy
        policy = policy_cls.from_args(args_cli)

        # Use policy length if available, otherwise use max_steps
        num_steps = policy.length() if policy.has_length() else max_steps
        assert num_steps is not None, "num_steps cannot be None"

        # Run the rollout
        rollout_policy(env, policy, num_steps, num_episodes=None)

        # Close the environment
        env.close()
        return True

    except Exception as e:
        print(f"Error running replay lerobot policy: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_g1_locomanip_replay_lerobot_policy_runner_single_env():
    result = run_simulation_app_function(
        _run_replay_lerobot_policy,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        config_yaml_path=Gr00tTestConstants.test_data_dir
        + "/test_g1_locomanip_lerobot/test_g1_locomanip_replay_action_config.yaml",
        environment="galileo_g1_locomanip_pick_and_place",
        object_name="brown_box",
        embodiment="g1_wbc_joint",
        max_steps=NUM_STEPS,
        trajectory_index=TRAJECTORY_INDEX,
    )
    assert result, "Test test_g1_locomanip_replay_lerobot_policy_runner_single_env failed"


@pytest.mark.skip(reason="Fails on CI for reasons under investigation.")
def test_gr1_manip_replay_lerobot_policy_runner_single_env():
    result = run_simulation_app_function(
        _run_replay_lerobot_policy,
        headless=HEADLESS,
        enable_cameras=ENABLE_CAMERAS,
        config_yaml_path=Gr00tTestConstants.test_data_dir
        + "/test_gr1_manip_lerobot/test_gr1_manip_replay_action_config.yaml",
        environment="gr1_open_microwave",
        object_name="microwave",
        embodiment="gr1_joint",
        max_steps=NUM_STEPS,
        trajectory_index=TRAJECTORY_INDEX,
    )
    assert result, "Test test_gr1_manip_replay_lerobot_policy_runner_single_env failed"


if __name__ == "__main__":
    test_g1_locomanip_replay_lerobot_policy_runner_single_env()
    test_gr1_manip_replay_lerobot_policy_runner_single_env()
