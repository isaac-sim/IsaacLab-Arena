# Copyright (c) 2025, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

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
