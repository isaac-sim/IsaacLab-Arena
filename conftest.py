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
#
# This conftest.py lives at the repo root so that it applies to the tests of every package.

from __future__ import annotations

from isaaclab_arena.tests.utils import pytest_session

# Expose agentic LLM test fixtures (``stub_openai``, etc.) to the suite.
pytest_plugins = ["isaaclab_arena.tests.utils.agentic_environment_generation"]


def pytest_sessionstart(session):
    # This function is called before the first test is run.
    # We stash the session object so we can access later to determine if any tests failed.
    pytest_session.PYTEST_SESSION = session
    session.tests_failed = False


def pytest_runtest_logreport(report):
    # This function is called after each test is run.
    # We set the tests_failed flag to True if the test failed.
    if report.when == "call" and report.failed:
        pytest_session.PYTEST_SESSION.tests_failed = True
