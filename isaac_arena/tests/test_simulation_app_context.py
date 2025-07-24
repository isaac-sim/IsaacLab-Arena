# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

from isaac_arena.tests.utils.subprocess import run_simulation_app_function_in_separate_process

TEST_ARG = 123


def simulation_app_running(simulation_app) -> bool:
    print("Hello, simulation app test!")
    return simulation_app.is_running()


def test_simulation_app_context():
    # Run a function which returns True if the simulation app is running.
    test_passed = run_simulation_app_function_in_separate_process(
        simulation_app_running,
    )
    assert test_passed, "Tested function returned False"


def got_argument(_, test_arg: int) -> bool:
    print(f"Got argument: {test_arg}")
    return test_arg == TEST_ARG


def test_run_simulation_app_function_in_separate_process_with_arg():
    # Run a function which returns True if the simulation app is running.
    test_passed = run_simulation_app_function_in_separate_process(
        got_argument,
        TEST_ARG,
    )
    assert test_passed, "Tested function returned False"
