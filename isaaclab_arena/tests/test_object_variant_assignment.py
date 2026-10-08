# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check placement assignments against Isaac Lab's native clone planner."""

import torch
from copy import deepcopy

import pytest
from isaaclab import cloner
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import CuboidCfg, PinholeCameraCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.assets.object_variant import ObjectVariant
from isaaclab_arena.scene.object_variant_assignment import (
    assign_object_variants,
    build_object_variant_assignments,
    object_variant_clone_strategy,
)


def _make_object(name: str, variant_count: int, random_choice: bool = False) -> Object:
    variants = []
    for variant_index in range(variant_count):
        width = 0.1 * (variant_index + 1)
        variants.append(ObjectVariant(CuboidCfg(size=(width, width, width)), ObjectType.RIGID))
    return Object(name=name, variants=variants, random_choice=random_choice)


def _make_scene(objects: list[Object], num_envs: int) -> InteractiveSceneCfg:
    scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0)
    # Isaac Lab moves sensor configs after asset configs before planning. The bridge
    # must tolerate the resulting change in the positions of homogeneous columns.
    scene_cfg.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", spawn=PinholeCameraCfg())
    for obj in objects:
        setattr(scene_cfg, obj.get_scene_key(), deepcopy(obj.object_cfg))
    scene_cfg.table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=CuboidCfg(size=(1.0, 1.0, 0.1)))
    return scene_cfg


def _make_native_plan(scene_cfg: InteractiveSceneCfg, objects: list[Object]):
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


def _assert_plan_assignments(plan, scene_cfg: InteractiveSceneCfg, objects: list[Object]) -> None:
    for obj in objects:
        scene_asset_cfg = getattr(scene_cfg, obj.get_scene_key())
        variant_rows = plan.cfg_rows[id(scene_asset_cfg)]
        object_mask = plan.clone_mask[list(variant_rows)]
        assert torch.equal(object_mask.sum(dim=0), torch.ones(scene_cfg.num_envs, dtype=torch.long))
        assert tuple(object_mask.long().argmax(dim=0).tolist()) == obj.variant_indices_by_env
        assert len(variant_rows) == len(obj.variants)


def test_native_clone_plan_matches_bounds_for_two_unequal_variant_counts():
    objects = [_make_object("pickup", 2), _make_object("destination", 3)]
    assign_object_variants(objects, num_envs=7, seed=19)
    scene_cfg = _make_scene(objects, num_envs=7)
    assignments = build_object_variant_assignments(scene_cfg, objects)
    with object_variant_clone_strategy(scene_cfg, assignments):
        plan = _make_native_plan(scene_cfg, objects)

    _assert_plan_assignments(plan, scene_cfg, objects)
    assert plan.clone_mask.shape == (2 + 3 + 1 + 1, 7)
    for obj in objects:
        bounds = obj.get_bounding_box_per_env(7)
        expected_widths = torch.tensor([0.1 * (index + 1) for index in obj.variant_indices_by_env])
        assert torch.allclose(bounds.max_point[:, 0] - bounds.min_point[:, 0], expected_widths)


def test_assignment_is_seeded_by_object_name_and_preserved_after_binding():
    objects = [_make_object("pickup", 3, True), _make_object("destination", 3, True)]
    reordered_objects = [_make_object("destination", 3, True), _make_object("pickup", 3, True)]
    other_seed_objects = [_make_object("pickup", 3, True), _make_object("destination", 3, True)]
    assign_object_variants(objects, num_envs=20, seed=19)
    assign_object_variants(reordered_objects, num_envs=20, seed=19)
    assign_object_variants(other_seed_objects, num_envs=20, seed=20)
    original_assignments = {obj.name: obj.variant_indices_by_env for obj in objects}
    assert original_assignments == {obj.name: obj.variant_indices_by_env for obj in reordered_objects}
    assert original_assignments != {obj.name: obj.variant_indices_by_env for obj in other_seed_objects}
    assert objects[0].variant_indices_by_env != objects[1].variant_indices_by_env

    assign_object_variants(objects, num_envs=20, seed=100)
    assert original_assignments == {obj.name: obj.variant_indices_by_env for obj in objects}
    with pytest.raises(AssertionError, match="already has variants assigned"):
        assign_object_variants(objects, num_envs=21, seed=19)

    # Objects have no relations; they still need the same assignment in simulation.
    assert all(not obj.get_relations() for obj in objects)
    scene_cfg = _make_scene(list(reversed(objects)), num_envs=20)
    assignments = build_object_variant_assignments(scene_cfg, objects)
    with object_variant_clone_strategy(scene_cfg, assignments):
        plan = _make_native_plan(scene_cfg, list(reversed(objects)))
    _assert_plan_assignments(plan, scene_cfg, objects)


def test_native_plan_keeps_unselected_variants_without_expanding_per_environment():
    objects = [_make_object("pickup", 4), _make_object("destination", 3)]
    assign_object_variants(objects, num_envs=2)
    scene_cfg = _make_scene(objects, num_envs=2)
    assignments = build_object_variant_assignments(scene_cfg, objects)
    with object_variant_clone_strategy(scene_cfg, assignments):
        plan = _make_native_plan(scene_cfg, objects)
    _assert_plan_assignments(plan, scene_cfg, objects)
    assert plan.clone_mask.shape == (4 + 3 + 1 + 1, 2)
    assert scene_cfg.pickup.spawn.spawn_paths[2:] == [None, None]


def test_single_variant_object_keeps_native_scene_strategy():
    obj = _make_object("pickup", 1)
    scene_cfg = _make_scene([obj], num_envs=3)
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    assign_object_variants([obj], num_envs=3)
    assignments = build_object_variant_assignments(scene_cfg, [obj])
    assert assignments == {}
    with object_variant_clone_strategy(scene_cfg, assignments):
        assert scene_cfg.clone_cfg.clone_strategy is original_strategy
    assert scene_cfg.clone_cfg.clone_strategy is original_strategy


@pytest.mark.parametrize("override", ["num_envs", "variant_count", "variant_order", "missing_object"])
def test_final_scene_overrides_cannot_change_bound_variants(override):
    obj = _make_object("pickup", 2)
    assign_object_variants([obj], num_envs=3)
    scene_cfg = _make_scene([obj], num_envs=3)
    if override == "num_envs":
        scene_cfg.num_envs = 4
    elif override == "variant_count":
        scene_cfg.pickup.spawn.assets_cfg.append(CuboidCfg(size=(2.0, 2.0, 2.0)))
    elif override == "variant_order":
        scene_cfg.pickup.spawn.assets_cfg.reverse()
    else:
        scene_cfg.pickup = None
    with pytest.raises(AssertionError):
        build_object_variant_assignments(scene_cfg, [obj])


def test_clone_strategy_rejects_unexpected_plan_structure():
    obj = _make_object("pickup", 2)
    assign_object_variants([obj], num_envs=3)
    scene_cfg = _make_scene([obj], num_envs=3)
    assignments = build_object_variant_assignments(scene_cfg, [obj])
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    with object_variant_clone_strategy(scene_cfg, assignments):
        strategy = scene_cfg.clone_cfg.clone_strategy
        with pytest.raises(AssertionError, match="variant count"):
            strategy(torch.tensor([[0], [1], [2]]), 3, "cpu")
        with pytest.raises(AssertionError, match="environment count"):
            strategy(torch.tensor([[0], [1]]), 4, "cpu")
        with pytest.raises(AssertionError, match="every asset"):
            strategy(torch.tensor([[-1], [0], [1]]), 3, "cpu")
        with pytest.raises(AssertionError, match="nonempty"):
            strategy(torch.empty((0, 1), dtype=torch.long), 3, "cpu")
    assert scene_cfg.clone_cfg.clone_strategy is original_strategy


def _roundtrip_environment_config(scene_cfg, objects):
    from omegaconf import OmegaConf

    from isaaclab_arena.utils.configclass import make_configclass

    assignment_cfg = build_object_variant_assignments(scene_cfg, objects)
    # Exercise native configclass serialization without importing the Kit-dependent Arena runtime.
    environment_cfg_type = make_configclass(
        "VariantEnvironmentCfg",
        [("scene", InteractiveSceneCfg, scene_cfg), ("object_variant_assignments", dict, assignment_cfg)],
    )
    env_cfg = environment_cfg_type()
    serialized = OmegaConf.to_container(OmegaConf.create(env_cfg.to_dict()), resolve=True)
    restored = deepcopy(env_cfg)
    restored.from_dict(serialized)
    return restored, serialized


def test_clone_strategy_survives_hydra_config_roundtrip():
    objects = [_make_object("pickup", 2, True), _make_object("destination", 3, True)]
    assign_object_variants(objects, num_envs=7, seed=19)
    restored, serialized = _roundtrip_environment_config(_make_scene(objects, num_envs=7), objects)
    original_strategy = restored.scene.clone_cfg.clone_strategy
    assert serialized["scene"]["clone_cfg"]["clone_strategy"].endswith(":sequential")
    for obj in objects:
        assert restored.object_variant_assignments[obj.name].variant_indices == obj.variant_indices_by_env

    with object_variant_clone_strategy(restored.scene, restored.object_variant_assignments):
        plan = _make_native_plan(restored.scene, objects)
    _assert_plan_assignments(plan, restored.scene, objects)
    assert restored.scene.clone_cfg.clone_strategy is original_strategy
    assert (
        restored.to_dict()["scene"]["clone_cfg"]["clone_strategy"] == serialized["scene"]["clone_cfg"]["clone_strategy"]
    )


@pytest.mark.parametrize(
    "override", ["num_envs", "variant_count", "variant_order", "variant_geometry", "missing_object"]
)
def test_runtime_rejects_hydra_overrides_that_change_placement_geometry(override):
    obj = _make_object("pickup", 2, True)
    assign_object_variants([obj], num_envs=7, seed=19)
    restored, _ = _roundtrip_environment_config(_make_scene([obj], num_envs=7), [obj])
    if override == "num_envs":
        restored.scene.num_envs = 8
    elif override == "variant_count":
        restored.scene.pickup.spawn.assets_cfg.append(CuboidCfg(size=(2.0, 2.0, 2.0)))
    elif override == "variant_order":
        restored.scene.pickup.spawn.assets_cfg.reverse()
    elif override == "variant_geometry":
        restored.scene.pickup.spawn.assets_cfg[0].size = (2.0, 2.0, 2.0)
    else:
        restored.scene.pickup = None
    original_strategy = restored.scene.clone_cfg.clone_strategy
    with pytest.raises(AssertionError):
        with object_variant_clone_strategy(restored.scene, restored.object_variant_assignments):
            pytest.fail("Invalid placement geometry reached scene construction")
    assert restored.scene.clone_cfg.clone_strategy is original_strategy


def test_runtime_clone_strategy_is_restored_when_scene_construction_fails():
    obj = _make_object("pickup", 2)
    assign_object_variants([obj], num_envs=3)
    scene_cfg = _make_scene([obj], num_envs=3)
    assignments = build_object_variant_assignments(scene_cfg, [obj])
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    with pytest.raises(RuntimeError, match="scene construction failed"):
        with object_variant_clone_strategy(scene_cfg, assignments):
            raise RuntimeError("scene construction failed")
    assert scene_cfg.clone_cfg.clone_strategy is original_strategy
