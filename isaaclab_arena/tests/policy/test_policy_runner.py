# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
    episode_name: str | None = None,
):
    args = [TestConstants.python_path, f"{TestConstants.examples_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append(policy_type)
    if policy_type == "replay":
        assert replay_file_path is not None, f"replay_file_path must be provided for policy_type {policy_type}"
        args.append("--replay_file_path")
        args.append(replay_file_path)
        if episode_name is not None:
            args.append("--episode_name")
            args.append(episode_name)
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


def test_zero_action_policy_press_button():
    run_policy_runner(
        policy_type="zero_action",
        example_environment="press_button",
        num_steps=NUM_STEPS,
    )


def test_zero_action_policy_kitchen_pick_and_place():
    # TODO(alexmillane, 2025.07.29): Get an exhaustive list of all scenes and embodiments
    # from a registry when we have one.
    example_environment = "kitchen_pick_and_place"
    embodiments = ["franka", "gr1_pink", "gr1_joint"]
    object_names = ["cracker_box", "tomato_soup_can"]
    for embodiment in embodiments:
        for object_name in object_names:
            run_policy_runner(
                policy_type="zero_action",
                example_environment=example_environment,
                embodiment=embodiment,
                object_name=object_name,
                num_steps=NUM_STEPS,
            )


def test_zero_action_policy_galileo_pick_and_place():
    # TODO(alexmillane, 2025.07.29): Get an exhaustive list of all scenes and embodiments
    # from a registry when we have one.
    # NOTE(alexmillane, 2025.09.04): Only test one configuration here to keep
    # the test fast.
    run_policy_runner(
        policy_type="zero_action",
        example_environment="galileo_pick_and_place",
        embodiment="gr1_pink",
        object_name="power_drill",
        num_steps=NUM_STEPS,
    )


def test_zero_action_policy_gr1_open_microwave():
    # TODO(alexmillane, 2025.07.29): Get an exhaustive list of all scenes and embodiments
    # from a registry when we have one.
    example_environment = "gr1_open_microwave"
    object_name = ["cracker_box", "tomato_soup_can", "mustard_bottle"]
    for object_name in object_name:
        run_policy_runner(
            policy_type="zero_action",
            example_environment=example_environment,
            embodiment="gr1_pink",
            background=None,
            object_name=object_name,
            num_steps=NUM_STEPS,
        )


def test_replay_policy_gr1_open_microwave():
    run_policy_runner(
        policy_type="replay",
        replay_file_path=TestConstants.test_data_dir + "/test_demo_gr1_open_microwave.hdf5",
        example_environment="gr1_open_microwave",
        embodiment="gr1_pink",
        num_steps=NUM_STEPS,
    )
