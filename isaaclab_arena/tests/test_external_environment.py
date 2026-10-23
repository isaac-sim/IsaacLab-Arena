# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from unittest.mock import patch

import pytest

from isaaclab_arena.evaluation.arena_experiment_result import ARENA_EXPERIMENT_RESULT_FILENAME
from isaaclab_arena.tests.test_experiment_runner import run_experiment_runner
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True
NUM_STEPS = 2

EXTERNAL_ENV_BASIC_IMPORT_PATH = "isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment"
EXTERNAL_ENVIRONMENTS_EXPERIMENT_PATH = (
    f"{TestConstants.repo_root}/isaaclab_arena_examples/external_environments/external_environments_experiment.yaml"
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


@pytest.mark.with_subprocess
def test_experiment_runner_with_external_environments(tmp_path):
    """The Experiment Runner imports environments that an Experiment selects by class path."""
    experiment_output_directory = tmp_path / "output"
    run_experiment_runner(
        EXTERNAL_ENVIRONMENTS_EXPERIMENT_PATH,
        headless=HEADLESS,
        config_option="--experiment_config",
        extra_args=[
            "--experiment_output_directory",
            str(experiment_output_directory),
            f"shared.rollout_limit.num_steps={NUM_STEPS}",
        ],
    )

    experiment_result_path = experiment_output_directory / ARENA_EXPERIMENT_RESULT_FILENAME
    experiment_result = json.loads(experiment_result_path.read_text(encoding="utf-8"))
    run_statuses = {run_name: run_result["status"] for run_name, run_result in experiment_result["runs"].items()}
    assert run_statuses == {"franka_table": "completed", "franka_table_with_task": "completed"}
