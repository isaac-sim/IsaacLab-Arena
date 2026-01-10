# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab_arena.tests.test_eval_runner import write_jobs_config_to_file
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True
NUM_STEPS = 2


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


def test_eval_runner_gr00t_config(tmp_path: str):
    """Test eval_runner using the gr00t_jobs_config.json."""
    # create a temporary config file only has two jobs for g1_locomanipulation task
    jobs = [
        {
            "name": "g1_locomanip_pick_and_place_brown_box",
            "arena_env_args": {
                "environment": "galileo_g1_locomanip_pick_and_place",
                "object": "brown_box",
                "embodiment": "g1_wbc_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "isaaclab_arena_gr00t.policy.gr00t_closedloop_policy.Gr00tClosedloopPolicy",
            "policy_args": {
                "policy_config_yaml_path": (
                    "isaaclab_arena_gr00t/policy/config/g1_locomanip_gr00t_closedloop_config.yaml"
                ),
                "policy_device": "cuda:0",
            },
        },
        {
            "name": "gr1_open_microwave_cracker_box",
            "arena_env_args": {
                "environment": "gr1_open_microwave",
                "object": "cracker_box",
                "embodiment": "gr1_joint",
            },
            "num_steps": NUM_STEPS,
            "policy_type": "zero_action",
        },
    ]
    temp_config_path = os.path.join(tmp_path, "test_eval_runner_gr00t_config.json")
    write_jobs_config_to_file(jobs, temp_config_path)
    run_eval_runner(temp_config_path)
