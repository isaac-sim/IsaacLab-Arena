# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import re
import subprocess

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants

HEADLESS = True
NUM_STEPS = 2


def write_jobs_config_to_file(jobs: list[dict], tmp_file_path: str):
    jobs_config = {"jobs": jobs}

    with open(tmp_file_path, "w", encoding="utf-8") as f:
        json.dump(jobs_config, f, indent=4)


def run_eval_runner_and_check_no_failures(jobs_config_path: str, headless: bool = HEADLESS):
    """Run the eval_runner and verify no jobs failed.

    Args:
        jobs_config_path: Path to the jobs config JSON file
        headless: Whether to run in headless mode

    Raises:
        AssertionError: If any jobs failed
    """
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/eval_runner.py"]
    args.append("--eval_jobs_config")
    args.append(jobs_config_path)
    if headless:
        args.append("--headless")

    result = subprocess.run(args, capture_output=True, text=True, check=True)
    output = result.stdout + result.stderr

    # Parse the output to find job statuses in the table
    # The table format is:
    # |                Job Name               |   Status  | ...
    # |     gr1_open_microwave_cracker_box    | completed | ...
    status_pattern = r"\|\s+([^|]+?)\s+\|\s+(pending|running|completed|failed)\s+\|"
    matches = re.findall(status_pattern, output, re.IGNORECASE)

    # Filter out the header row
    job_statuses = [(name.strip(), status.strip()) for name, status in matches if name.strip() != "Job Name"]

    # Check for failed jobs
    failed_jobs = [name for name, status in job_statuses if status.lower() == "failed"]

    if failed_jobs:
        print("\n" + output)  # Print full output for debugging
        raise AssertionError(f"The following jobs failed: {', '.join(failed_jobs)}\nAll job statuses: {job_statuses}")


def test_eval_runner_two_jobs_zero_action(tmp_path):
    """Test eval_runner with 2 jobs using zero_action policy on different objects."""
    jobs = [
        {
            "name": "gr1_open_microwave_cracker_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "gr1_open_microwave_sugar_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "object": "sugar_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_eval_runner_two_jobs_zero_action.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner_and_check_no_failures(temp_config_path)


def test_eval_runner_multiple_environments(tmp_path):
    """Test eval_runner with jobs across different environments."""
    jobs = [
        {
            "name": "kitchen_pick_cracker_box",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "galileo_pick_power_drill",
            "arena_env_args": {
                "environment": "galileo_pick_and_place",
                "object": "power_drill",
                "embodiment": "gr1_pink",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_eval_runner_multiple_environments.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner_and_check_no_failures(temp_config_path)


def test_eval_runner_different_embodiments(tmp_path):
    """Test eval_runner with jobs using different embodiments."""
    jobs = [
        {
            "name": "kitchen_pick_gr1_pink",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "tomato_soup_can",
                "embodiment": "gr1_pink",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "kitchen_pick_franka",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "tomato_soup_can",
                "embodiment": "franka",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_eval_runner_different_embodiments.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner_and_check_no_failures(temp_config_path)


def test_eval_runner_from_existing_config():
    """Test eval_runner using the zero_action_jobs_config.json and verify no jobs failed."""
    config_path = f"{TestConstants.arena_environments_dir}/eval_jobs_configs/zero_action_jobs_config.json"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    run_eval_runner_and_check_no_failures(config_path)


@pytest.mark.with_cameras
def test_eval_runner_enable_cameras(tmp_path):
    """Test eval_runner with enable_cameras set to true."""
    jobs = [
        {
            "name": "kitchen_pick_and_place_no_cameras",
            "arena_env_args": {
                "environment": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "franka",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
        {
            "name": "kitchen_pick_and_place",
            "arena_env_args": {
                "enable_cameras": True,
                "environment": "kitchen_pick_and_place",
                "object": "cracker_box",
                "embodiment": "franka",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": {},
        },
    ]

    temp_config_path = str(tmp_path / "test_eval_runner_enable_cameras.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner_and_check_no_failures(temp_config_path)
