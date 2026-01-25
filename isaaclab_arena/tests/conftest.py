# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import time
import pytest


# @pytest.fixture(autouse=True)
# def pause_between_tests():
#     """
#     Fixture that automatically adds a 10-second pause between tests.
    
#     This fixture runs automatically for every test due to autouse=True.
#     The pause occurs after each test completes (yield executes the test,
#     then the code after yield runs).
#     """
#     # Code before yield runs before the test
#     yield
#     # Code after yield runs after the test
#     time.sleep(1)

PYTEST_SESSION = None

def pytest_sessionstart(session):
    global PYTEST_SESSION
    print(f"HERE: pytest_sessionstart. Stashing the session object.")
    PYTEST_SESSION = session
    session.tests_failed = False



def pytest_runtest_logreport(report):
    print(f"HERE: pytest_runtest_logreport. Report: {report}")
    print(f"HERE: session.tests_failed: {PYTEST_SESSION.tests_failed}")
    if report.when == "call" and report.failed:
        PYTEST_SESSION.tests_failed = True
        print(f"HERE: Setting session.tests_failed to True")
