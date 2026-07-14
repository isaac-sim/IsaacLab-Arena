# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest markers for scoping tests between the Docker and native uv environments."""

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants

requires_docker_assets = pytest.mark.skipif(
    not TestConstants.is_docker,
    reason="needs Isaac assets not yet promoted to the public Nucleus (e.g. maple_table.usda); run in Docker",
)
"""Skip natively: the test loads assets that 404 on the public Nucleus the uv env resolves against."""
