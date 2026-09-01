# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from isaaclab_arena.tests.utils import persistent_simulation_app
from isaaclab_arena.tests.utils import subprocess as subprocess_utils
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

TEST_ARG = 123


def simulation_app_running(simulation_app) -> bool:
    print("Hello, simulation app test!")
    return simulation_app.is_running()


def test_simulation_app_context():
    # Run a function which returns True if the simulation app is running.
    test_passed = run_function_with_persistent_simulation_app(
        simulation_app_running,
    )
    assert test_passed, "Tested function returned False"


def got_argument(_, test_arg: int) -> bool:
    print(f"Got argument: {test_arg}")
    return test_arg == TEST_ARG


def test_run_function_with_persistent_simulation_app_with_arg():
    # Run a function which returns True if the simulation app is running.
    test_passed = run_function_with_persistent_simulation_app(
        got_argument,
        test_arg=TEST_ARG,
    )
    assert test_passed, "Tested function returned False"


@pytest.mark.parametrize(("tests_failed", "expected_exit_code"), [(False, 0), (True, 1)])
def test_persistent_simulation_app_force_exit_without_pytest_session(monkeypatch, tests_failed, expected_exit_code):
    """A direct test subprocess must not enter Kit's native shutdown path."""
    app = SimpleNamespace(close=lambda: pytest.fail("Force-exit mode must not call app.close()"))
    monkeypatch.setattr(persistent_simulation_app, "_PERSISTENT_SIM_APP_LAUNCHER", SimpleNamespace(app=app))
    monkeypatch.setattr(persistent_simulation_app, "PYTEST_SESSION", None)
    monkeypatch.setattr(subprocess_utils, "_AT_LEAST_ONE_TEST_FAILED", tests_failed)
    monkeypatch.setenv("ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE", "1")

    def raise_system_exit(exit_code):
        raise SystemExit(exit_code)

    monkeypatch.setattr(persistent_simulation_app.os, "_exit", raise_system_exit)

    with pytest.raises(SystemExit) as exc_info:
        persistent_simulation_app._close_persistent()

    assert exc_info.value.code == expected_exit_code
