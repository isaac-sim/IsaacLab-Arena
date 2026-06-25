# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Isaac Sim makes testing complicated. During shutdown Isaac Sim will
# terminate the surrounding pytest process with exit code 0, regardless
# of whether the tests passed or failed.
# To work around this, we stash the session object and set a flag
# when a test fails. This flag is checked in isaaclab_arena.tests.utils.subprocess.py
# prior to closing the simulation app, in order to generate the correct exit code.


PYTEST_SESSION = None


def pytest_sessionstart(session):
    # This function is called before the first test is run (and before collection).
    # We stash the session object so we can access later to determine if any tests failed.
    global PYTEST_SESSION
    PYTEST_SESSION = session
    session.tests_failed = False

    # Boot the persistent SimulationApp before collection. With the Isaac Lab pip wheel,
    # ``@configclass`` resolves lazily and only works once the app exists; a test importing
    # such a cfg at module level otherwise leaves Isaac Lab half-imported, duplicating
    # classes (e.g. ``ArticulationCfg``) and breaking ``isinstance`` in later-collected tests.
    # Note: this means every pytest invocation here starts Isaac Sim (~5s), even quick runs.
    markexpr = getattr(session.config.option, "markexpr", "") or ""

    def _selects(marker: str) -> bool:
        return (marker in markexpr) and (f"not {marker}" not in markexpr)

    # The with_subprocess phase manages its own Isaac Sim child processes (see pytest.ini).
    if _selects("with_subprocess"):
        return
    from isaaclab_arena.tests.utils.subprocess import get_persistent_simulation_app

    get_persistent_simulation_app(headless=True, enable_cameras=_selects("with_cameras"))


def pytest_runtest_logreport(report):
    # This function is called after each test is run.
    # We set the tests_failed flag to True if the test failed.
    if report.when == "call" and report.failed:
        PYTEST_SESSION.tests_failed = True
