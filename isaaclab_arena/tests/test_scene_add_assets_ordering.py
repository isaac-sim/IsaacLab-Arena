# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from isaaclab_arena.tests.utils.subprocess import run_simulation_app_function

HEADLESS = True


def _test_object_references_added_before_parents(simulation_app) -> bool:
    """ObjectReferences can be added before their parents."""
    import torch
    from types import SimpleNamespace

    import pytest

    from isaaclab_arena.assets.object_base import ObjectType
    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.scene.scene import Scene

    # Set up a test scene
    asset_registry = AssetRegistry()
    table_background = asset_registry.get_asset_by_name("office_table")()
    prim_path = "{ENV_REGEX_NS}/office_table/Geometry/sm_tabletop_a01_01/sm_tabletop_a01_top_01"
    base_reference = ObjectReference(name="table", prim_path=prim_path, parent_asset=table_background)
    rigid_reference = ObjectReference(
        name="rigid_table",
        prim_path=prim_path,
        parent_asset=table_background,
        object_type=ObjectType.RIGID,
    )
    articulation_reference = ObjectReference(
        name="articulated_table",
        prim_path=prim_path,
        parent_asset=table_background,
        object_type=ObjectType.ARTICULATION,
    )
    try:
        scene = Scene(
            assets=[
                base_reference,
                rigid_reference,
                articulation_reference,
                table_background,
            ]
        )
    except AssertionError:
        return False

    assert scene.assets["table"] is base_reference
    scene_cfg = scene.get_scene_cfg()
    assert hasattr(scene_cfg, "office_table")
    assert not hasattr(scene_cfg, "table")
    assert hasattr(scene_cfg, "rigid_table")
    assert hasattr(scene_cfg, "articulated_table")

    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            device="cpu",
            num_envs=2,
            scene=SimpleNamespace(env_origins=env_origins),
        )
    )
    relative_pose = base_reference.get_object_pose(env)
    world_pose = base_reference.get_object_pose(env, is_relative=False)
    assert torch.allclose(relative_pose[0], relative_pose[1])
    assert torch.allclose(world_pose[:, :3], relative_pose[:, :3] + env_origins)
    with pytest.raises(AssertionError, match="cannot be moved independently"):
        base_reference.set_object_pose(env, base_reference.get_initial_pose())
    return True


def _test_object_reference_without_parent_raises(simulation_app) -> bool:
    """Adding an ObjectReference whose parent is not in the scene should fail."""

    from isaaclab_arena.assets.object_reference import ObjectReference
    from isaaclab_arena.assets.registries import AssetRegistry
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
