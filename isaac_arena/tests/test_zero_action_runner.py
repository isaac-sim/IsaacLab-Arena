# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from isaac_arena.tests.utils.constants import TestConstants
from isaac_arena.tests.utils.subprocess import run_subprocess


def test_zero_action_runner_franka():
    run_subprocess([
        TestConstants.python_path,
        f"{TestConstants.examples_dir}/zero_action_runner.py",
        "--headless",
        "--embodiment",
        "franka",
        "--num_steps",
        "2",
    ])


def test_zero_action_runner_gr1():
    run_subprocess([
        TestConstants.python_path,
        f"{TestConstants.examples_dir}/zero_action_runner.py",
        "--headless",
        "--embodiment",
        "gr1t2",
        "--num_steps",
        "2",
    ])
