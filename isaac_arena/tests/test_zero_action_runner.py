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

from isaac_arena.examples.zero_action_runner import main


def test_before():
    print("TEST BEFORE")
    assert False, "Fail!"


def test_zero_action_runner():
    run_subprocess([
        TestConstants.python_path,
        f"{TestConstants.examples_dir}/zero_action_runner.py",
        "--headless",
        "--embodiment",
        "franka",
        "--scene",
        "kitchen",
        "--arena_task",
        "pick_and_place",
        "--task",
        "test",
        "--num_steps",
        "2",
    ])
