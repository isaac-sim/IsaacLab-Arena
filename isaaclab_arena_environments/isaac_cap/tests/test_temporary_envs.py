# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the temporary Isaac CAP environment demos.

Remove this directory together with ``isaaclab_arena_environments/isaac_cap``.
"""

import pytest

from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.subprocess import run_subprocess

pytestmark = [pytest.mark.isaac_cap, pytest.mark.with_subprocess]

_CABLE_DEMO_SCRIPT = f"{TestConstants.arena_environments_dir}/isaac_cap/cable_routing/cable_env_behaviour_demo.py"
_GEAR_DEMO_SCRIPT = f"{TestConstants.arena_environments_dir}/isaac_cap/gear_insertion/gear_env_behaviour_demo.py"


@pytest.mark.parametrize("variant", ("easy", "medium"))
def test_cable_routing_behaviour_demo(variant: str) -> None:
    """Run one headless demo cycle and verify that success resets the environment."""
    result = run_subprocess(
        [
            TestConstants.python_path,
            _CABLE_DEMO_SCRIPT,
            "--variant",
            variant,
            "--cycles",
            "1",
            "--pause-steps",
            "25",
            "--no-real-time",
            "--visualizer",
            "none",
        ],
        capture_output=True,
    )

    assert result is not None
    expected = "[cable-behaviour-demo] cycle 1: success reset observed"
    assert expected in result.stdout, result.stdout + result.stderr


@pytest.mark.parametrize("variant", ("easy", "medium"))
def test_gear_insertion_behaviour_demo(variant: str) -> None:
    """Run one headless demo cycle and verify that success resets every environment."""
    result = run_subprocess(
        [
            TestConstants.python_path,
            _GEAR_DEMO_SCRIPT,
            variant,
            "--cycles",
            "1",
            "--pause-steps",
            "25",
            "--no-real-time",
            "--visualizer",
            "none",
        ],
        capture_output=True,
    )

    assert result is not None
    expected = "[gear-validation] cycle 1: success reset observed in all 2 environments"
    assert expected in result.stdout, result.stdout + result.stderr
