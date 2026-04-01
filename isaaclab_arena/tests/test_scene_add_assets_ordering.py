# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_object_references_added_before_parents(simulation_app) -> bool:
    """ObjectReferences can be added before their parents."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.scene.scene import Scene

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.scene.scene import Scene

    # Set up a test scene
    asset_registry = AssetRegistry()
    table_background = asset_registry.get_asset_by_name("office_table")()
    table_reference = ObjectReference(
       name="table",
       prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
       parent_asset=table_background,
   )
    try:
        scene = Scene(assets=[table_reference, table_background])
    except AssertionError:
        return False

    return True


def _test_object_reference_without_parent_raises(simulation_app) -> bool:
    """Adding an ObjectReference whose parent is not in the scene should fail."""

    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.scene.scene import Scene
    asset_registry = AssetRegistry()
    table_background = asset_registry.get_asset_by_name("office_table")()
    table_reference = ObjectReference(
       name="table",
       prim_path="{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01",
       parent_asset=table_background,
   )

    try:
        Scene(assets=[table_reference])
    except AssertionError:
        return True

    return False

def test_object_references_added_before_parents():
    result = run_simulation_app_function(
        _test_object_references_added_before_parents,
        headless=HEADLESS,
    )
    assert result, "ObjectReferences can't be added before their parents"


def test_object_reference_without_parent_raises():
    result = run_simulation_app_function(
        _test_object_reference_without_parent_raises,
        headless=HEADLESS,
    )
    assert result, "ObjectReference without parent should fail"
