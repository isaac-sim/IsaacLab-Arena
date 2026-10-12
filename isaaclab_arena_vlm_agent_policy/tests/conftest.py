# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Preserve the shared SimulationApp failure tracking for tests outside the core package.
from isaaclab_arena.tests.conftest import pytest_runtest_logreport, pytest_sessionstart  # noqa: F401
