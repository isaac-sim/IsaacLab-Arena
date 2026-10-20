# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the RoboLab table fixtures."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

ROBOLAB_TABLE_NAMES = ["table_oak_robolab", "table_black_robolab", "table_bamboo_robolab"]
ROBOLAB_TABLE_POSITION_XYZ = (0.5471954941749573, -5.750182754127309e-05, -0.3470015823841095)


def _test_robolab_table_fixtures_are_kinematic(simulation_app):
    from isaaclab_arena.assets.registries import AssetRegistry

    registry = AssetRegistry()
    for name in ROBOLAB_TABLE_NAMES:
        assert registry.is_registered(name), f"{name} is not registered"
        table = registry.get_asset_by_name(name)()
        assert table.spawn_cfg_addon["rigid_props"].kinematic_enabled is True, name
        assert tuple(table.initial_pose.position_xyz) == ROBOLAB_TABLE_POSITION_XYZ, name
    return True


def test_robolab_table_fixtures_are_kinematic():
    assert run_function_with_persistent_simulation_app(_test_robolab_table_fixtures_are_kinematic)
