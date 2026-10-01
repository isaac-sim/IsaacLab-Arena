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


def _leave_in_memory_stage(simulation_app) -> bool:
    from isaaclab.sim.utils import create_new_stage
    from pxr import UsdGeom

    stage = create_new_stage()
    UsdGeom.Xform.Define(stage, "/LeakedScene")
    return True


def _check_fresh_stage(simulation_app) -> bool:
    import omni.usd
    from isaaclab.sim.utils import get_current_stage

    stage = get_current_stage()
    assert stage is not None
    assert not stage.GetPrimAtPath("/LeakedScene"), "Isaac Lab reused the preceding test's scene"
    assert stage == omni.usd.get_context().get_stage(), "Isaac Lab and Kit must use the same fresh stage"
    return True


def test_persistent_simulation_app_clears_in_memory_stage():
    """A test that authors an in-memory scene must not leak it into the next test."""
    assert run_function_with_persistent_simulation_app(_leave_in_memory_stage)
    assert run_function_with_persistent_simulation_app(_check_fresh_stage)


@pytest.mark.parametrize(
    ("pytest_failed", "fallback_failed", "expected_exit_code"),
    [(None, False, 0), (None, True, 1), (False, True, 0), (True, False, 1)],
)
def test_persistent_simulation_app_force_exit_preserves_test_result(
    monkeypatch, pytest_failed, fallback_failed, expected_exit_code
):
    """Force-exit mode must preserve pytest or direct-execution results."""
    app = SimpleNamespace(close=lambda: pytest.fail("Force-exit mode must not call app.close()"))
    monkeypatch.setattr(persistent_simulation_app, "_PERSISTENT_SIM_APP_LAUNCHER", SimpleNamespace(app=app))
    pytest_session = None if pytest_failed is None else SimpleNamespace(tests_failed=pytest_failed)
    monkeypatch.setattr(persistent_simulation_app, "PYTEST_SESSION", pytest_session)
    monkeypatch.setattr(subprocess_utils, "_AT_LEAST_ONE_TEST_FAILED", fallback_failed)
    monkeypatch.setenv("ISAACLAB_ARENA_FORCE_EXIT_ON_COMPLETE", "1")

    def raise_system_exit(exit_code):
        raise SystemExit(exit_code)

    monkeypatch.setattr(persistent_simulation_app.os, "_exit", raise_system_exit)

    with pytest.raises(SystemExit) as exc_info:
        persistent_simulation_app._close_persistent()

    assert exc_info.value.code == expected_exit_code
