# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-environment spawn entries of scene objects."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

if TYPE_CHECKING:
    from isaaclab.cloner import ClonePlan
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet


def _make_cuboid_entry(size: tuple[float, float, float]) -> CuboidCfg:
    """Return a cuboid spawn entry whose rigid_props put its rigid body at the spawn root."""
    import isaaclab.sim as sim_utils

    return sim_utils.CuboidCfg(size=size, rigid_props=sim_utils.RigidBodyPropertiesCfg())


def _make_multi_asset_object(name: str, sizes: list[tuple[float, float, float]]) -> Object:
    """Return a rigid object whose PinnedMultiAssetSpawnerCfg has one cuboid spawn entry per size."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg

    spawner_cfg = PinnedMultiAssetSpawnerCfg(
        assets_cfg=[_make_cuboid_entry(size) for size in sizes],
        random_choice=False,
        activate_contact_sensors=None,
    )
    return Object(name=name, object_type=ObjectType.RIGID, spawner_cfg=spawner_cfg)


def _test_object_with_several_spawn_entries_uses_each_environment_entry(simulation_app) -> bool:
    from unittest.mock import patch

    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg
    from isaaclab_arena.assets.spawn_cfg_geometry import SpawnCfgGeometry

    sizes = [(0.1, 0.1, 0.2), (0.2, 0.2, 0.4), (0.3, 0.3, 0.1)]
    obj = _make_multi_asset_object("boxes", sizes)
    assert obj.has_multiple_assets
    # object_cfg.spawn is the object's own copy of spawner_cfg. It keeps the subclass and has no indices yet.
    assert isinstance(obj.object_cfg.spawn, PinnedMultiAssetSpawnerCfg)
    assert obj.object_cfg.spawn is not obj.spawner_cfg
    assert obj.object_cfg.spawn.asset_indices_by_env is None
    # The inherited func resolves against Isaac Lab's wrappers module, not the subclass's module.
    assert str(obj.object_cfg.spawn.func) == "isaaclab.sim.spawners.wrappers.wrappers:spawn_multi_asset"

    with patch("isaaclab_arena.assets.object.SpawnCfgGeometry", wraps=SpawnCfgGeometry) as spawn_cfg_geometry_class:
        # One box for all environments: the tallest entry, the 0.4 m cuboid.
        bounding_box = obj.get_bounding_box()
        torch.testing.assert_close(bounding_box.min_point, torch.tensor([[-0.1, -0.1, -0.2]]))
        torch.testing.assert_close(bounding_box.max_point, torch.tensor([[0.1, 0.1, 0.2]]))
        with pytest.raises(AssertionError, match="has no spawn entry indices"):
            obj.get_bounding_box_per_env(3)

        obj.object_cfg.spawn.asset_indices_by_env = (1, 0, 1)
        expected_sizes = torch.tensor([sizes[1], sizes[0], sizes[1]])
        for _ in range(2):
            torch.testing.assert_close(obj.get_bounding_box_per_env(3).size, expected_sizes)
        # Every entry has its rigid body at the spawn root.
        assert obj.get_contact_sensor_prim_path() == obj.prim_path
    # Repeated queries measure each distinct spawn entry once.
    assert spawn_cfg_geometry_class.call_count == len(sizes)
    # The indices are written to object_cfg.spawn only; the declared spawner_cfg keeps None.
    assert obj.spawner_cfg.asset_indices_by_env is None

    # A new size gives entry 0 a new geometry key, so environment 1, which spawns entry 0, gets new bounds.
    obj.object_cfg.spawn.assets_cfg[0].size = (0.5, 0.5, 0.5)
    expected_sizes = torch.tensor([sizes[1], (0.5, 0.5, 0.5), sizes[1]])
    torch.testing.assert_close(obj.get_bounding_box_per_env(3).size, expected_sizes)
    return True


def test_object_with_several_spawn_entries_uses_each_environment_entry():
    assert run_function_with_persistent_simulation_app(
        _test_object_with_several_spawn_entries_uses_each_environment_entry
    )


def _test_object_set_spawns_several_members_from_a_pinned_spawner(simulation_app) -> bool:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.pinned_multi_asset_spawner_cfg import PinnedMultiAssetSpawnerCfg

    small_box = Object(name="small_box", object_type=ObjectType.RIGID, spawner_cfg=_make_cuboid_entry((0.1, 0.1, 0.1)))
    large_box = Object(name="large_box", object_type=ObjectType.RIGID, spawner_cfg=_make_cuboid_entry((0.2, 0.2, 0.2)))

    object_set = RigidObjectSet(name="boxes", objects=[small_box, large_box])
    assert isinstance(object_set.object_cfg.spawn, PinnedMultiAssetSpawnerCfg)
    assert len(object_set.object_cfg.spawn.assets_cfg) == 2
    assert object_set.object_cfg.spawn.asset_indices_by_env is None
    assert object_set.has_multiple_assets
    return True


def test_object_set_spawns_several_members_from_a_pinned_spawner():
    assert run_function_with_persistent_simulation_app(_test_object_set_spawns_several_members_from_a_pinned_spawner)


_EXISTING_ASSET_INDICES = (2, 1, 0) * 6 + (2, 1)
"""Spawn entry indices for 20 environments, written before assignment and different from round robin."""


def _make_object_set(name: str, entry_count: int, random_choice: bool = False) -> RigidObjectSet:
    """Return a RigidObjectSet of cuboid members, where member i has edge length 0.1 * (i + 1)."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.assets.object_type import ObjectType

    members = []
    for member_index in range(entry_count):
        width = 0.1 * (member_index + 1)
        members.append(
            Object(
                name=f"{name}_{member_index}",
                object_type=ObjectType.RIGID,
                spawner_cfg=_make_cuboid_entry((width, width, width)),
            )
        )
    return RigidObjectSet(name=name, objects=members, random_choice=random_choice)


def _make_scene_cfg(objects: list[Object], num_envs: int) -> InteractiveSceneCfg:
    """Return a scene configuration with a camera, a copy of each object's configuration and a single-entry table."""
    from copy import deepcopy

    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg

    scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0)
    # Declared first, but Isaac Lab plans sensors after assets.
    scene_cfg.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", spawn=sim_utils.PinholeCameraCfg())
    for obj in objects:
        setattr(scene_cfg, obj.get_scene_key(), deepcopy(obj.object_cfg))
    scene_cfg.table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=_make_cuboid_entry((1.0, 1.0, 0.1)))
    return scene_cfg


def _make_clone_plan(scene_cfg: InteractiveSceneCfg, objects: list[Object]) -> ClonePlan:
    """Plan the scene's clones with its clone strategy, with the table between the objects and the camera last."""
    from isaaclab import cloner

    asset_cfgs = [getattr(scene_cfg, obj.get_scene_key()) for obj in objects]
    asset_cfgs.insert(1, scene_cfg.table)
    asset_cfgs.append(scene_cfg.camera)
    for asset_cfg in asset_cfgs:
        asset_cfg.prim_path = cloner.expand_env_regex_ns(asset_cfg.prim_path)
    return cloner.make_clone_plan(
        asset_cfgs,
        num_clones=scene_cfg.num_envs,
        env_spacing=scene_cfg.env_spacing,
        device="cpu",
        clone_strategy=scene_cfg.clone_cfg.clone_strategy,
    )


def _get_planned_entry_indices(plan: ClonePlan, scene_cfg: InteractiveSceneCfg, obj: Object) -> tuple[int, ...]:
    """Return the spawn entry the clone plan gives each environment for one object."""
    object_cfg = getattr(scene_cfg, obj.get_scene_key())
    entry_rows = plan.cfg_rows[id(object_cfg)]
    assert len(entry_rows) == len(object_cfg.spawn.assets_cfg)
    entry_mask = plan.clone_mask[list(entry_rows)]
    assert torch.equal(entry_mask.sum(dim=0), torch.ones(scene_cfg.num_envs, dtype=torch.long))
    return tuple(entry_mask.long().argmax(dim=0).tolist())


def _assert_plan_follows_scene_indices(plan: ClonePlan, scene_cfg: InteractiveSceneCfg, objects: list[Object]) -> None:
    """Assert that the clone plan gives each environment the spawn entry the scene configuration names for it."""
    for obj in objects:
        scene_indices = getattr(scene_cfg, obj.get_scene_key()).spawn.asset_indices_by_env
        assert _get_planned_entry_indices(plan, scene_cfg, obj) == tuple(scene_indices)


def _test_pinned_clone_plan_matches_two_sets_with_unequal_member_counts(simulation_app, three_member_set_first):
    from isaaclab_arena.scene.object_asset_assignment import (
        asset_assignment_clone_strategy,
        assign_asset_indices,
        validate_scene_asset_indices,
    )

    pickup = _make_object_set("pickup", 2)
    destination = _make_object_set("destination", 3)
    objects = [destination, pickup] if three_member_set_first else [pickup, destination]
    assert assign_asset_indices(objects, num_envs=7, seed=19) is None
    assert pickup.object_cfg.spawn.asset_indices_by_env == (0, 1, 0, 1, 0, 1, 0)
    assert destination.object_cfg.spawn.asset_indices_by_env == (0, 1, 2, 0, 1, 2, 0)

    scene_cfg = _make_scene_cfg(objects, num_envs=7)
    pinned_spawners = validate_scene_asset_indices(scene_cfg)
    assert [scene_key for scene_key, _ in pinned_spawners] == [obj.name for obj in objects]
    with asset_assignment_clone_strategy(scene_cfg):
        plan = _make_clone_plan(scene_cfg, objects)
    # The plan follows the scene configuration's copy of the indices.
    _assert_plan_follows_scene_indices(plan, scene_cfg, objects)
    # One row per spawn entry of both sets, plus the table and the camera.
    assert tuple(plan.clone_mask.shape) == (2 + 3 + 1 + 1, 7)
    # Planning writes spawn_paths into the scene configuration; its indices still pass validation.
    assert validate_scene_asset_indices(scene_cfg) == pinned_spawners

    # Isaac Lab's default sequential strategy walks the product of entry counts, so the first set
    # changes only every few environments and spawns other entries than its indices name.
    unpinned_scene_cfg = _make_scene_cfg(objects, num_envs=7)
    unpinned_plan = _make_clone_plan(unpinned_scene_cfg, objects)
    expected_unpinned_first = (0, 0, 1, 1, 2, 2, 0) if three_member_set_first else (0, 0, 0, 1, 1, 1, 0)
    assert _get_planned_entry_indices(unpinned_plan, unpinned_scene_cfg, objects[0]) == expected_unpinned_first
    return True


@pytest.mark.parametrize("three_member_set_first", [False, True])
def test_pinned_clone_plan_matches_two_sets_with_unequal_member_counts(three_member_set_first):
    assert run_function_with_persistent_simulation_app(
        _test_pinned_clone_plan_matches_two_sets_with_unequal_member_counts,
        three_member_set_first=three_member_set_first,
    )


def _make_seeded_scene_objects() -> dict[str, Object]:
    """Return, keyed by name, an ordered set, a random set with indices, a random set and a plain object."""
    existing_random_set = _make_object_set("existing", 3, random_choice=True)
    existing_random_set.object_cfg.spawn.asset_indices_by_env = _EXISTING_ASSET_INDICES
    objects = [
        _make_object_set("ordered", 2),
        existing_random_set,
        _make_object_set("random", 3, random_choice=True),
        _make_multi_asset_object("plain", [(0.1, 0.1, 0.1), (0.2, 0.2, 0.2), (0.3, 0.3, 0.3)]),
    ]
    return {obj.name: obj for obj in objects}


def _test_random_assignment_follows_the_seed_and_keeps_existing_indices(simulation_app):
    from isaaclab_arena.scene.object_asset_assignment import assign_asset_indices

    def assign_and_read_indices(seed):
        objects_by_name = _make_seeded_scene_objects()
        assign_asset_indices(objects_by_name.values(), num_envs=20, seed=seed)
        return {name: obj.object_cfg.spawn.asset_indices_by_env for name, obj in objects_by_name.items()}

    # The ordered set is the first object whose indices are drawn here (seed + 0). The existing set keeps its
    # indices and is not counted, so the random set draws with seed + 1. The plain object has no random_choice.
    expected_random_indices = tuple(
        torch.randint(0, 3, (20,), generator=torch.Generator().manual_seed(19 + 1)).tolist()
    )
    expected_indices_by_name = {
        "ordered": tuple(env_id % 2 for env_id in range(20)),
        "existing": _EXISTING_ASSET_INDICES,
        "random": expected_random_indices,
        "plain": tuple(env_id % 3 for env_id in range(20)),
    }
    assert assign_and_read_indices(19) == expected_indices_by_name
    # Fresh objects with the same seed get the same indices; another seed draws others.
    assert assign_and_read_indices(19) == expected_indices_by_name
    assert assign_and_read_indices(20)["random"] != expected_random_indices

    # The indices stay in the spawn configurations, so reusing the objects for 21 environments is rejected.
    objects_by_name = _make_seeded_scene_objects()
    assign_asset_indices(objects_by_name.values(), num_envs=20, seed=19)
    with pytest.raises(AssertionError, match="for 20 environments, not 21"):
        assign_asset_indices(objects_by_name.values(), num_envs=21, seed=19)

    # Without a seed, the random set draws from torch's global generator.
    torch.manual_seed(0)
    unseeded_indices_by_name = assign_and_read_indices(None)
    torch.manual_seed(0)
    expected_unseeded_random_indices = tuple(torch.randint(0, 3, (20,)).tolist())
    assert unseeded_indices_by_name == {**expected_indices_by_name, "random": expected_unseeded_random_indices}
    return True


def test_random_assignment_follows_the_seed_and_keeps_existing_indices():
    assert run_function_with_persistent_simulation_app(
        _test_random_assignment_follows_the_seed_and_keeps_existing_indices
    )
