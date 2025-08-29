# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from isaac_arena.tests.utils.constants import TestConstants
from isaac_arena.tests.utils.subprocess import run_subprocess

HEADLESS = True


def run_zero_action_runner(embodiment: str, background: str, object_name: str):

    args = [
        TestConstants.python_path,
        f"{TestConstants.examples_dir}/zero_action_runner.py",
        "--embodiment",
        embodiment,
        "--background",
        background,
        "--object",
        object_name,
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
    object_name = "cracker_box"
    for embodiment in embodiments:
        for background in backgrounds:
            run_zero_action_runner(embodiment, background, object_name)
