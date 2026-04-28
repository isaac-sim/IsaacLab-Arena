# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
NUM_STEPS = 2


def run_policy_runner(
    policy_type: str,
    example_environment: str,
    num_steps: int,
    embodiment: str | None = None,
    background: str | None = None,
    object_name: str | None = None,
    replay_file_path: str | None = None,
    checkpoint_path: str | None = None,
    episode_name: str | None = None,
):
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append(policy_type)
    if policy_type == "replay":
        assert replay_file_path is not None, f"replay_file_path must be provided for policy_type {policy_type}"
        args.append("--replay_file_path")
        args.append(replay_file_path)
        if episode_name is not None:
            args.append("--episode_name")
            args.append(episode_name)
    if policy_type == "rsl_rl":
        assert checkpoint_path is not None, f"checkpoint_path must be provided for policy_type {policy_type}"
        args.append("--checkpoint_path")
        args.append(checkpoint_path)
    args.append("--num_steps")
    args.append(str(num_steps))
    if HEADLESS:
        args.append("--headless")

    args.append(example_environment)
    if embodiment is not None:
        args.append("--embodiment")
        args.append(embodiment)
    if background is not None:
        args.append("--background")
        args.append(background)
    if object_name is not None:
        args.append("--object")
        args.append(object_name)
    run_subprocess(args)


@pytest.mark.with_subprocess
def test_zero_action_policy():
    run_policy_runner(
        policy_type="zero_action",
        example_environment="kitchen_pick_and_place",
        embodiment="franka_ik",
        object_name="cracker_box",
        num_steps=NUM_STEPS,
    )


@pytest.mark.with_subprocess
def test_replay_policy():
    run_policy_runner(
        policy_type="replay",
        replay_file_path=TestConstants.test_data_dir + "/test_demo_gr1_open_microwave.hdf5",
        example_environment="gr1_open_microwave",
        embodiment="gr1_pink",
        num_steps=NUM_STEPS,
    )


@pytest.mark.with_subprocess
@pytest.mark.parametrize("object_name", ["cracker_box", "tomato_soup_can"])
def test_zero_action_policy_with_objects(object_name):
    run_policy_runner(
        policy_type="zero_action",
        example_environment="gr1_open_microwave",
        embodiment="gr1_pink",
        object_name=object_name,
        num_steps=NUM_STEPS,
    )
