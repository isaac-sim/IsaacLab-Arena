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

import os

# Expose agentic LLM test fixtures (``stub_openai``, etc.) to the suite.
pytest_plugins = ["isaaclab_arena.tests.utils.agentic_environment_generation"]

PYTEST_SESSION = None


def pytest_sessionstart(session):
    # This function is called before the first test is run.
    # We stash the session object so we can access later to determine if any tests failed.
    global PYTEST_SESSION
    PYTEST_SESSION = session
    session.tests_failed = False

    # TODO(alexmillane, 2026-08-31): [lab-render-after-rebuild-bug] Remove once the render-after-rebuild
    # bug is fixed in Lab. The shared persistent SimulationApp rebuilds the stage many times across the
    # suite, and under GPU+Fabric every build after the first has rendering artifacts, so force Fabric
    # off for every env build in the test process (read in arena_env_builder). setdefault lets a
    # developer opt back in with ISAACLAB_ARENA_DISABLE_FABRIC=0.
    os.environ.setdefault("ISAACLAB_ARENA_DISABLE_FABRIC", "1")


def pytest_runtest_logreport(report):
    # This function is called after each test is run.
    # We set the tests_failed flag to True if the test failed.
    if report.when == "call" and report.failed:
        PYTEST_SESSION.tests_failed = True
