# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Holds the running pytest session so the persistent SimulationApp can read the test result on exit."""

PYTEST_SESSION = None
"""The pytest session, set by the root conftest.py. ``None`` when not running under pytest."""
