# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for Arena's MuJoCo-Warp builder integration."""

import subprocess

from isaaclab_arena.tests.utils.constants import TestConstants

_CFG_SCRIPT = """
from isaaclab_arena.physics import ArenaMJWarpSolverCfg, NewtonArenaMJWarpManager

assert ArenaMJWarpSolverCfg().class_type is NewtonArenaMJWarpManager
assert NewtonArenaMJWarpManager.__name__.lower().startswith("newton")
"""

_BUILDER_SCRIPT = """
from newton import ModelBuilder

from isaaclab_arena.physics import NewtonArenaMJWarpManager

builder = ModelBuilder()
NewtonArenaMJWarpManager._register_builder_attributes(builder)
NewtonArenaMJWarpManager._register_builder_attributes(builder)

expected = {
    "mujoco:condim": "mjc:condim",
    "mujoco:solref": "mjc:solref",
    "mujoco:geom_solimp": "mjc:solimp",
    "mujoco:gravcomp": "mjc:gravcomp",
}
for name, usd_name in expected.items():
    assert builder.has_custom_attribute(name)
    assert builder.custom_attributes[name].usd_attribute_name == usd_name
"""


def _run_isolated(script: str) -> None:
    """Run a Newton import check without polluting later SimulationApp tests."""
    result = subprocess.run(
        [TestConstants.python_path, "-c", script],
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"isolated Newton check failed:\n{result.stdout}\n{result.stderr}"
    )


def test_arena_mjwarp_cfg_selects_usd_aware_manager():
    _run_isolated(_CFG_SCRIPT)


def test_mjwarp_builder_attributes_are_registered_idempotently():
    _run_isolated(_BUILDER_SCRIPT)
