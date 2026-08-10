# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest skip markers for optional external dependencies."""

import pytest

try:
    import gr00t  # noqa: F401

    _HAS_GR00T = True
except ImportError:
    _HAS_GR00T = False

requires_gr00t = pytest.mark.skipif(not _HAS_GR00T, reason="gr00t not installed")
