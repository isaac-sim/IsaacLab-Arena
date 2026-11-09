# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Isaac Sim makes testing complicated. During shutdown Isaac Sim will
# terminate the surrounding pytest process with exit code 0, regardless
# of whether the tests passed or failed.
# To work around this, we stash the session object and set a flag
# when a test fails. This flag is checked in
# isaaclab_arena.tests.utils.persistent_simulation_app.py
# prior to closing the simulation app, in order to generate the correct exit code.

from __future__ import annotations

import pytest

from isaaclab_arena.tests.utils.hang_guard import exit_if_stuck

# Expose agentic LLM test fixtures (``stub_openai``, etc.) to the suite.
pytest_plugins = ["isaaclab_arena.tests.utils.agentic_environment_generation"]

PYTEST_SESSION = None


def pytest_sessionstart(session):
    # This function is called before the first test is run.
    # We stash the session object so we can access later to determine if any tests failed.
    global PYTEST_SESSION
    PYTEST_SESSION = session
    session.tests_failed = False


def pytest_runtest_logreport(report):
    # This function is called after each test is run.
    # We set the tests_failed flag to True if the test failed.
    if report.when == "call" and report.failed:
        PYTEST_SESSION.tests_failed = True


# The slowest passing tests in CI take about 10 minutes (the gear insertion camera test and the
# asset registry test). A test running longer than this is stuck.
TEST_TIMEOUT_S = 1200


@pytest.fixture(autouse=True)
def fail_when_test_is_stuck():
    """Exit with a stack dump of every thread when a test (including its simulation app startup) is stuck."""
    with exit_if_stuck(TEST_TIMEOUT_S):
        yield
