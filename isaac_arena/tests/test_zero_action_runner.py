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

HEADLESS = True


def run_zero_action_runner(embodiment: str, background: str):

    args = [
        TestConstants.python_path,
        f"{TestConstants.examples_dir}/zero_action_runner.py",
        "--embodiment",
        embodiment,
        "--background",
        background,
        "--num_steps",
        "2",
    ]
    if HEADLESS:
        args.append("--headless")

    run_subprocess(args)


def test_zero_action_runner_franka_kitchen():
    run_zero_action_runner("franka", "kitchen_pick_and_place")


def test_zero_action_runner_gr1_kitchen():
    run_zero_action_runner("gr1", "kitchen_pick_and_place")


def test_zero_action_runner_franka_packing_table():
    run_zero_action_runner("franka", "packing_table_pick_and_place")


def test_zero_action_runner_gr1_packing_table():
    run_zero_action_runner("gr1", "packing_table_pick_and_place")
