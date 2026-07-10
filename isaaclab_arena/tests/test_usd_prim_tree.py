# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for USD prim-tree loading."""

from __future__ import annotations

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.utils.asset_usd import resolve_asset_usd_path
from isaaclab_arena.utils.usd_prim_tree import load_usd_prim_tree


def test_kitchen_prim_tree_counter_and_fridge():
    """Requires Lightwheel Robocasa kitchen USD on disk (Docker dev image)."""
    usd_path = resolve_asset_usd_path(
        AssetSpec(
            id="lightwheel_robocasa_kitchen",
            registry_name="lightwheel_robocasa_kitchen",
            params={"layout_id": 1, "style_id": 1},
        ),
    )
    tree = load_usd_prim_tree(usd_path)
    records = {record.relative_path: record for record in tree}

    counter = records.get("counter_right_main_group/top_geometry")
    assert counter is not None, "counter_right_main_group/top_geometry missing from kitchen USD"
    assert counter.object_type == ObjectType.BASE

    fridge = records.get("fridge_main_group")
    assert fridge is not None, "fridge_main_group missing from kitchen USD"
    assert fridge.object_type == ObjectType.ARTICULATION
    assert "fridge_door_joint" in fridge.joint_names
