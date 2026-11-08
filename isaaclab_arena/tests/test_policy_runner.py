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
    object_name: str | None = None,
    replay_file_path: str | None = None,
    enable_cameras: bool = False,
    hydra_overrides: list[str] | None = None,
) -> None:
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append(policy_type)
    if policy_type == "replay":
        assert replay_file_path is not None, f"replay_file_path must be provided for policy_type {policy_type}"
        args.append("--replay_file_path")
        args.append(replay_file_path)
    args.append("--num_steps")
    args.append(str(num_steps))
    if not HEADLESS:
        args.extend(["--viz", "kit"])
    if enable_cameras:
        args.append("--enable_cameras")
    # automatically detect if the env source is a graph spec yaml or an example-environment name
    # and pass the appropriate argument.
    # NOTE: --env_spec takes the next arg as its value, so example_environment must be
    # appended directly after it. Do not insert any other arg between these two lines.
    if example_environment.endswith((".yaml", ".yml")):
        args.append("--env_spec")
    args.append(example_environment)
    if embodiment is not None:
        args.append("--embodiment")
        args.append(embodiment)
    if object_name is not None:
        args.append("--object")
        args.append(object_name)
    if hydra_overrides:
        args.extend(hydra_overrides)
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
        replay_file_path=TestConstants.test_data_dir + "/test_sequential_task_mimic_data_generation.hdf5",
        example_environment="franka_put_and_close_door",
        embodiment="franka_ik",
        num_steps=NUM_STEPS,
    )
