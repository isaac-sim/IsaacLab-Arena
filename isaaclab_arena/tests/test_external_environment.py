# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
NUM_STEPS = 2

EXTERNAL_ENV_BASIC_IMPORT_PATH = "isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment"
EXTERNAL_ENV_ADVANCED_IMPORT_PATH = (
    "isaaclab_arena_examples.external_environments.advanced:ExternalFrankaTableWithTaskEnvironment"
)


def _test_external_environment_registration_callback(_) -> bool:
    """Verify the Isaac Lab callback registers an externally defined Arena environment."""
    import gymnasium as gym

    from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback

    callback_args = [
        "external_environment_interop_test",
        "--task",
        "franka_table",
        "--external_environment_class_path",
        EXTERNAL_ENV_BASIC_IMPORT_PATH,
        "--object",
        "cracker_box",
    ]
    with patch.object(sys, "argv", callback_args):
        remaining_args = environment_registration_callback()

    assert remaining_args == []
    assert gym.spec("franka_table").id == "franka_table"
    return True


def test_external_environment_registration_callback():
    """Isaac Lab scripts can register external Arena environments through the callback."""
    result = run_function_with_persistent_simulation_app(_test_external_environment_registration_callback)
    assert result


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
    args.append("--external_environment_class_path")
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
        environment_import_path=EXTERNAL_ENV_BASIC_IMPORT_PATH,
        example_environment="franka_table",
        object_name="cracker_box",
        num_steps=NUM_STEPS,
    )


@pytest.mark.with_subprocess
def test_external_environment_franka_table_with_task():
    run_policy_runner_with_external_environment(
        policy_type="zero_action",
        environment_import_path=EXTERNAL_ENV_ADVANCED_IMPORT_PATH,
        example_environment="franka_table_with_task",
        object_name="cracker_box",
        num_steps=NUM_STEPS,
    )
