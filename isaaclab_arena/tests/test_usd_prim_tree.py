# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for USD prim-tree loading."""

from __future__ import annotations

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.environment_spec.arena_env_graph_types import AssetSpec
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.usd_prim_tree import load_usd_physics_roots, load_usd_prim_tree


def _test_kitchen_physics_prim_trees(_) -> bool:
    """Requires Lightwheel Robocasa kitchen USD on disk (Docker dev image)."""
    usd_path = AssetSpec(
        id="lightwheel_robocasa_kitchen",
        registry_name="lightwheel_robocasa_kitchen",
        params={"layout_id": 1, "style_id": 1},
    ).resolve_usd_path()
    tree = load_usd_prim_tree(usd_path)
    records = {record.relative_path: record for record in tree}

    counter = records.get("counter_right_main_group/top_geometry")
    assert counter is not None, "counter_right_main_group/top_geometry missing from kitchen USD"
    assert counter.object_type == ObjectType.BASE

    fridge = records.get("fridge_main_group")
    assert fridge is not None, "fridge_main_group missing from kitchen USD"
    assert fridge.object_type == ObjectType.ARTICULATION
    assert "fridge_door_joint" in fridge.joint_names

    physics_roots = load_usd_physics_roots(usd_path)
    assert len(physics_roots) == 21
    assert all(object_type == ObjectType.ARTICULATION for object_type in physics_roots.values())

    usd_path = AssetSpec(
        id="replicator_kitchen_peninsula",
        registry_name="replicator_kitchen_peninsula",
    ).resolve_usd_path()
    physics_roots = load_usd_physics_roots(usd_path)

    assert len(physics_roots) == 56
    online_visual_roots = {path: object_type for path, object_type in physics_roots.items() if "_OnlineVisual" in path}
    assert len(online_visual_roots) == 24
    assert ObjectType.ARTICULATION in online_visual_roots.values()
    assert ObjectType.RIGID in online_visual_roots.values()
    return True


def test_kitchen_physics_prim_trees():
    assert run_function_with_persistent_simulation_app(_test_kitchen_physics_prim_trees)
