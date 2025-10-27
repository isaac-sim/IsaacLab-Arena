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

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

TEST_ARG = 123


def simulation_app_running(simulation_app) -> bool:
    print("Hello, simulation app test!")
    return simulation_app.is_running()


def test_simulation_app_context():
    # Run a function which returns True if the simulation app is running.
    test_passed = run_simulation_app_function(
        simulation_app_running,
    )
    assert test_passed, "Tested function returned False"


def got_argument(_, test_arg: int) -> bool:
    print(f"Got argument: {test_arg}")
    return test_arg == TEST_ARG


def test_run_simulation_app_function_with_arg():
    # Run a function which returns True if the simulation app is running.
    test_passed = run_simulation_app_function(
        got_argument,
        test_arg=TEST_ARG,
    )
    assert test_passed, "Tested function returned False"
