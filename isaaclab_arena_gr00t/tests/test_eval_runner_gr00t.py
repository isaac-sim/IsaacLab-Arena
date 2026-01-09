# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True


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


def test_eval_runner_gr00t_config():
    """Test eval_runner using the gr00t_jobs_config.json."""
    config_path = f"{TestConstants.evaluation_dir}/configs/gr00t_jobs_config.json"
    assert os.path.exists(config_path), f"Config file not found: {config_path}"
    run_eval_runner(config_path)

