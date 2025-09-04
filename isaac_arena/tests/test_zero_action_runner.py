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
NUM_STEPS = 2


def run_zero_action_runner(
    example_environment: str,
    num_steps: int,
    embodiment: str | None = None,
    background: str | None = None,
    object_name: str | None = None,
):

    args = [TestConstants.python_path, f"{TestConstants.examples_dir}/zero_action_runner.py"]
    args.append("--num_steps")
    args.append(str(num_steps))
    if HEADLESS:
        args.append("--headless")

    args.append(example_environment)
    if embodiment is not None:
        args.append("--embodiment")
        args.append(embodiment)
    if background is not None:
        args.append("--background")
        args.append(background)
    if object_name is not None:
        args.append("--object")
        args.append(object_name)

    run_subprocess(args)


def test_zero_action_runner_pick_and_place():
    # TODO(alexmillane, 2025.07.29): Get an exhaustive list of all scenes and embodiments
    # from a registry when we have one.
    example_environment = "pick_and_place"
    embodiments = ["franka", "gr1"]
    object_names = ["cracker_box", "tomato_soup_can"]
    for embodiment in embodiments:
        for object_name in object_names:
            run_zero_action_runner(
                example_environment=example_environment,
                embodiment=embodiment,
                object_name=object_name,
                num_steps=NUM_STEPS,
            )


def test_zero_action_runner_gr1_open_microwave():
    # TODO(alexmillane, 2025.07.29): Get an exhaustive list of all scenes and embodiments
    # from a registry when we have one.
    example_environment = "gr1_open_microwave"
    object_name = ["cracker_box", "tomato_soup_can", "mustard_bottle"]
    for object_name in object_name:
        run_zero_action_runner(
            example_environment=example_environment,
            embodiment=None,
            background=None,
            object_name=object_name,
            num_steps=NUM_STEPS,
        )
