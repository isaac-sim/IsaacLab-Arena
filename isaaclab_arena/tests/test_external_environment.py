# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.test_experiment_runner import read_run_statuses, run_experiment_runner
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
NUM_STEPS = 2

EXTERNAL_ENV_ADVANCED_IMPORT_PATH = (
    "isaaclab_arena_examples.external_environments.advanced:ExternalFrankaTableWithTaskEnvironment"
)
EXTERNAL_ENVIRONMENTS_EXPERIMENT_PATH = (
    f"{TestConstants.repo_root}/isaaclab_arena_examples/external_environments/external_environments_experiment.yaml"
)


def _test_external_environment_registration_callback(_) -> bool:
    """Verify the Isaac Lab callback registers an externally defined Arena environment."""
    import gymnasium as gym

    from isaaclab_arena.assets.registries import EnvironmentRegistry
    from isaaclab_arena.environments.isaaclab_interop import environment_registration_callback

    # Only --external_environment_class_path may import the module, or this test would pass without the flag.
    assert not EnvironmentRegistry().is_registered("franka_table_with_task")
    callback_args = [
        "external_environment_interop_test",
        "--task",
        "franka_table_with_task",
        "--external_environment_class_path",
        EXTERNAL_ENV_ADVANCED_IMPORT_PATH,
        "--object",
        "cracker_box",
    ]
    with patch.object(sys, "argv", callback_args):
        remaining_args = environment_registration_callback()

    assert remaining_args == []
    assert gym.spec("franka_table_with_task").id == "franka_table_with_task"
    return True


def test_external_environment_registration_callback():
    """Isaac Lab scripts can register external Arena environments through the callback."""
    result = run_function_with_persistent_simulation_app(_test_external_environment_registration_callback)
    assert result


@pytest.mark.with_subprocess
def test_experiment_runner_evaluates_external_environments_selected_by_class_path(tmp_path):
    """Run the example Experiment, which selects both external environments by dotted class path."""
    experiment_output_directory = tmp_path / "output"

    run_experiment_runner(
        EXTERNAL_ENVIRONMENTS_EXPERIMENT_PATH,
        headless=HEADLESS,
        extra_args=[
            "--experiment_output_directory",
            str(experiment_output_directory),
            f"shared.rollout_limit.num_steps={NUM_STEPS}",
        ],
    )

    run_statuses = read_run_statuses(experiment_output_directory)
    assert run_statuses == {"franka_table": "completed", "franka_table_with_task": "completed"}
