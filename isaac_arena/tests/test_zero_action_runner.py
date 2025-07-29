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


def test_zero_action_runner():
    # TODO(alexmillane, 2025.07.29): Get an exhaustive list of all scenes and embodiments
    # from a registry when we have one.
    embodiments = ["franka", "gr1"]
    backgrounds = ["kitchen_pick_and_place", "packing_table_pick_and_place"]
    for embodiment in embodiments:
        for background in backgrounds:
            run_zero_action_runner(embodiment, background)
