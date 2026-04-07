# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
NUM_STEPS = 2

EXTERNAL_ENV_IMPORT_PATH = "isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment"


def run_policy_runner_with_external_environment(
    policy_type: str,
    environment_import_path: str,
    example_environment: str,
    num_steps: int,
    object_name: str | None = None,
):
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/policy_runner.py"]
    args.append("--policy_type")
    args.append(policy_type)
    args.append("--num_steps")
    args.append(str(num_steps))
    if HEADLESS:
        args.append("--headless")
    else:
        args.append("--visualizer")
        args.append("kit")
    args.append("--environment")
    args.append(environment_import_path)
    args.append(example_environment)
    if object_name is not None:
        args.append("--object")
        args.append(object_name)
    run_subprocess(args)


@pytest.mark.with_subprocess
def test_external_environment_franka_table():
    run_policy_runner_with_external_environment(
        policy_type="zero_action",
        environment_import_path=EXTERNAL_ENV_IMPORT_PATH,
        example_environment="franka_table",
        object_name="cracker_box",
        num_steps=NUM_STEPS,
    )
