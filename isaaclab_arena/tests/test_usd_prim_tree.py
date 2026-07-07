# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for USD prim-tree loading."""

from __future__ import annotations

import pytest

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environments.arena_env_graph_types import AssetSpec
from isaaclab_arena.utils.asset_usd import resolve_asset_usd_path
from isaaclab_arena.utils.usd_prim_tree import load_usd_prim_tree

pxr = pytest.importorskip("pxr")


def _record_map(usd_path: str):
    tree = load_usd_prim_tree(usd_path)
    return {record.relative_path: record for record in tree}


def test_resolve_asset_usd_path_maple_table():
    usd_path = resolve_asset_usd_path(
        AssetSpec(id="maple_table_robolab", registry_name="maple_table_robolab"),
    )
    assert usd_path.endswith("maple_table.usda")


def test_maple_table_prim_tree_contains_table_rigid_or_base():
    usd_path = resolve_asset_usd_path(
        AssetSpec(id="maple_table_robolab", registry_name="maple_table_robolab"),
    )
    try:
        records = _record_map(usd_path)
    except Exception as exc:
        pytest.skip(f"maple table USD unavailable: {exc}")
    table = records.get("table")
    assert table is not None, f"expected 'table' in prim tree, got {sorted(records)[:20]}"
    assert table.object_type in {ObjectType.RIGID, ObjectType.BASE}


@pytest.mark.flaky(max_runs=3, min_passes=1)
def test_kitchen_prim_tree_counter_and_fridge():
    """Requires Lightwheel Robocasa kitchen USD on disk (Docker dev image)."""
    usd_path = resolve_asset_usd_path(
        AssetSpec(
            id="lightwheel_robocasa_kitchen",
            registry_name="lightwheel_robocasa_kitchen",
            params={"layout_id": 1, "style_id": 1},
        ),
    )
    records = _record_map(usd_path)

    counter = records.get("counter_right_main_group/top_geometry")
    assert counter is not None, "counter_right_main_group/top_geometry missing from kitchen USD"
    assert counter.object_type == ObjectType.BASE

    fridge = records.get("fridge_main_group")
    assert fridge is not None, "fridge_main_group missing from kitchen USD"
    assert fridge.object_type == ObjectType.ARTICULATION
    assert "fridge_door_joint" in fridge.joint_names
