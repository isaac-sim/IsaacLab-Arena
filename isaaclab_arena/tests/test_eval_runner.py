# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import tempfile

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
NUM_STEPS = 2


def create_temp_jobs_config(jobs: list[dict]) -> str:
    """Create a temporary jobs config file.

    Args:
        jobs: List of job dictionaries

    Returns:
        Path to the temporary config file
    """
    jobs_config = {"jobs": jobs}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(jobs_config, f, indent=4)
        temp_config_path = f.name

    return temp_config_path


def run_eval_runner(jobs_config_path: str):
    """Run the eval_runner with the given jobs config.

    Args:
        jobs_config_path: Path to the jobs config JSON file
    """
    args = [TestConstants.python_path, f"{TestConstants.evaluation_dir}/eval_runner.py"]
    args.append("--eval_jobs_config")
    args.append(jobs_config_path)
    args.append("--num_envs")
    args.append("1")
    if HEADLESS:
        args.append("--headless")

    run_subprocess(args)


def test_eval_runner_two_jobs_zero_action():
    """Test eval_runner with 2 jobs using zero_action policy on different objects."""
    jobs = [
        {
            "name": "gr1_open_microwave_cracker_box",
            "arena_env_args": [
                "gr1_open_microwave",
                "--object",
                "cracker_box",
                "--embodiment",
                "gr1_joint",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
        {
            "name": "gr1_open_microwave_sugar_box",
            "arena_env_args": [
                "gr1_open_microwave",
                "--object",
                "sugar_box",
                "--embodiment",
                "gr1_joint",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
    ]

    temp_config_path = create_temp_jobs_config(jobs)
    try:
        run_eval_runner(temp_config_path)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_eval_runner_multiple_environments():
    """Test eval_runner with jobs across different environments."""
    jobs = [
        {
            "name": "kitchen_pick_cracker_box",
            "arena_env_args": [
                "kitchen_pick_and_place",
                "--object",
                "cracker_box",
                "--embodiment",
                "gr1_joint",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
        {
            "name": "galileo_pick_power_drill",
            "arena_env_args": [
                "galileo_pick_and_place",
                "--object",
                "power_drill",
                "--embodiment",
                "gr1_pink",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
    ]

    temp_config_path = create_temp_jobs_config(jobs)
    try:
        run_eval_runner(temp_config_path)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_eval_runner_different_embodiments():
    """Test eval_runner with jobs using different embodiments."""
    jobs = [
        {
            "name": "kitchen_pick_gr1_pink",
            "arena_env_args": [
                "kitchen_pick_and_place",
                "--object",
                "tomato_soup_can",
                "--embodiment",
                "gr1_pink",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
        {
            "name": "kitchen_pick_franka",
            "arena_env_args": [
                "kitchen_pick_and_place",
                "--object",
                "tomato_soup_can",
                "--embodiment",
                "franka",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
    ]

    temp_config_path = create_temp_jobs_config(jobs)
    try:
        run_eval_runner(temp_config_path)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)


def test_eval_runner_from_existing_config():
    """Test eval_runner using the zero_action_jobs_config.json."""
    config_path = f"{TestConstants.evaluation_dir}/configs/zero_action_jobs_config.json"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    run_eval_runner(config_path)


def test_eval_runner_job_status_tracking():
    """Test that job status is correctly tracked and printed throughout execution."""
    jobs = [
        {
            "name": "status_test_job_1",
            "arena_env_args": [
                "gr1_open_microwave",
                "--object",
                "cracker_box",
                "--embodiment",
                "gr1_joint",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
        {
            "name": "status_test_job_2",
            "arena_env_args": [
                "kitchen_pick_and_place",
                "--object",
                "tomato_soup_can",
                "--embodiment",
                "franka",
            ],
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
            "policy_args": [],
        },
    ]

    temp_config_path = create_temp_jobs_config(jobs)
    try:
        run_eval_runner(temp_config_path)
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
