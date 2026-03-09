# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Re-exec pytest with PYTHONPATH so GR00T deps load first (set GROOT_DEPS_DIR when running GR00T tests).
import os
import sys

_groot_deps = os.environ.get("GROOT_DEPS_DIR")
if _groot_deps and os.environ.get("_GROOT_PYTHONPATH_APPLIED") != "1":
    os.environ["PYTHONPATH"] = _groot_deps + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["_GROOT_PYTHONPATH_APPLIED"] = "1"
    os.execv(sys.executable, [sys.executable, "-m", "pytest"] + sys.argv[1:])

# Isaac Sim exits with 0 on shutdown; stash session and set tests_failed so subprocess.py can report failure.
import isaaclab_arena.tests.conftest as arena_conftest


def pytest_sessionstart(session):
    # This function is called before the first test is run.
    # We stash the session object so we can access later to determine if any tests failed.
    # Update the arena conftest so subprocess.py sees the session.
    arena_conftest.PYTEST_SESSION = session
    session.tests_failed = False


def pytest_runtest_logreport(report):
    # This function is called after each test is run.
    # We set the tests_failed flag to True if the test failed.
    if report.when == "call" and report.failed:
        arena_conftest.PYTEST_SESSION.tests_failed = True
