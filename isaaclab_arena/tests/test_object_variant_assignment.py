# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check pre-placement assignments against Isaac Lab's native clone planner."""

from __future__ import annotations

import torch
from copy import deepcopy
from functools import wraps
from typing import TYPE_CHECKING

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

if TYPE_CHECKING:
    from isaaclab.scene import InteractiveSceneCfg

    from isaaclab_arena.assets.object import Object


def _with_simulation_app(test_function):
    """Run a case after Kit loads USD while preserving its pytest fixture signature."""

    @wraps(test_function)
    def run_test(*args, **kwargs):
        def run_in_app(simulation_app):
            test_function(*args, **kwargs)
            return True

        assert run_function_with_persistent_simulation_app(run_in_app)

    return run_test


def _make_object(name: str, variant_count: int, random_choice: bool = False) -> Object:
    from isaaclab.sim import CuboidCfg

    from isaaclab_arena.assets.object import Object

    variants = []
    for variant_index in range(variant_count):
        width = 0.1 * (variant_index + 1)
        variants.append(CuboidCfg(size=(width, width, width)))
    return Object(name=name, variants=variants, random_choice=random_choice)


def _make_scene(objects: list[Object], num_envs: int) -> InteractiveSceneCfg:
    from isaaclab.assets import RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import CuboidCfg, PinholeCameraCfg

    scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=2.0)
    # Isaac Lab moves sensors after assets before planning; homogeneous columns may move.
    scene_cfg.camera = CameraCfg(prim_path="{ENV_REGEX_NS}/Camera", spawn=PinholeCameraCfg())
    for obj in objects:
        setattr(scene_cfg, obj.get_scene_key(), deepcopy(obj.object_cfg))
    scene_cfg.table = RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Table", spawn=CuboidCfg(size=(1.0, 1.0, 0.1)))
    return scene_cfg


def _make_native_plan(scene_cfg: InteractiveSceneCfg, objects: list[Object]):
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


def _assert_plan_assignments(plan, scene_cfg: InteractiveSceneCfg, objects: list[Object]) -> None:
    for obj in objects:
        scene_asset_cfg = getattr(scene_cfg, obj.get_scene_key())
        variant_rows = plan.cfg_rows[id(scene_asset_cfg)]
        object_mask = plan.clone_mask[list(variant_rows)]
        assert torch.equal(object_mask.sum(dim=0), torch.ones(scene_cfg.num_envs, dtype=torch.long))
        assert tuple(object_mask.long().argmax(dim=0).tolist()) == obj.variant_indices_by_env
        assert len(variant_rows) == len(obj.spawn_cfg.assets_cfg)


@_with_simulation_app
def test_native_clone_plan_matches_bounds_for_two_unequal_variant_counts():
    from isaaclab_arena.scene.object_variant_assignment import (
        assign_object_variants,
        object_variant_clone_strategy,
        validate_object_variant_assignments,
    )

    objects = [_make_object("pickup", 2), _make_object("destination", 3)]
    assignments = assign_object_variants(objects, num_envs=7, seed=19)
    scene_cfg = _make_scene(objects, num_envs=7)
    validate_object_variant_assignments(scene_cfg, assignments)
    with object_variant_clone_strategy(scene_cfg, assignments):
        plan = _make_native_plan(scene_cfg, objects)

    _assert_plan_assignments(plan, scene_cfg, objects)
    # Native planning writes spawn_paths without changing the captured asset settings.
    validate_object_variant_assignments(scene_cfg, assignments)
    assert plan.clone_mask.shape == (2 + 3 + 1 + 1, 7)
    for obj in objects:
        bounds = obj.get_bounding_box_per_env(7)
        expected_widths = torch.tensor([0.1 * (index + 1) for index in obj.variant_indices_by_env])
        assert torch.allclose(bounds.max_point[:, 0] - bounds.min_point[:, 0], expected_widths)


@_with_simulation_app
def test_assignment_is_seeded_by_object_name_and_preserved_after_binding():
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants, object_variant_clone_strategy

    objects = [_make_object("pickup", 3, True), _make_object("destination", 3, True)]
    reordered_objects = [_make_object("destination", 3, True), _make_object("pickup", 3, True)]
    other_seed_objects = [_make_object("pickup", 3, True), _make_object("destination", 3, True)]
    assignments = assign_object_variants(objects, num_envs=20, seed=19)
    assign_object_variants(reordered_objects, num_envs=20, seed=19)
    assign_object_variants(other_seed_objects, num_envs=20, seed=20)
    original_indices = {obj.name: obj.variant_indices_by_env for obj in objects}
    assert original_indices == {obj.name: obj.variant_indices_by_env for obj in reordered_objects}
    assert original_indices != {obj.name: obj.variant_indices_by_env for obj in other_seed_objects}
    assert objects[0].variant_indices_by_env != objects[1].variant_indices_by_env

    assign_object_variants(objects, num_envs=20, seed=100)
    assert original_indices == {obj.name: obj.variant_indices_by_env for obj in objects}
    with pytest.raises(AssertionError, match="already has variants assigned"):
        assign_object_variants(objects, num_envs=21, seed=19)

    # Objects without relations still need matching assignments in simulation.
    assert all(not obj.get_relations() for obj in objects)
    scene_cfg = _make_scene(list(reversed(objects)), num_envs=20)
    with object_variant_clone_strategy(scene_cfg, assignments):
        plan = _make_native_plan(scene_cfg, list(reversed(objects)))
    _assert_plan_assignments(plan, scene_cfg, objects)


@_with_simulation_app
def test_native_plan_keeps_unselected_variants_without_expanding_per_environment():
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants, object_variant_clone_strategy

    objects = [_make_object("pickup", 4), _make_object("destination", 3)]
    assignments = assign_object_variants(objects, num_envs=2)
    scene_cfg = _make_scene(objects, num_envs=2)
    with object_variant_clone_strategy(scene_cfg, assignments):
        plan = _make_native_plan(scene_cfg, objects)
    _assert_plan_assignments(plan, scene_cfg, objects)
    assert plan.clone_mask.shape == (4 + 3 + 1 + 1, 2)
    assert scene_cfg.pickup.spawn.spawn_paths[2:] == [None, None]


@_with_simulation_app
def test_single_variant_object_keeps_native_scene_strategy():
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants, object_variant_clone_strategy

    obj = _make_object("pickup", 1)
    scene_cfg = _make_scene([obj], num_envs=3)
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    assignments = assign_object_variants([obj], num_envs=3)
    assert assignments == {}
    with object_variant_clone_strategy(scene_cfg, assignments):
        assert scene_cfg.clone_cfg.clone_strategy is original_strategy
    assert scene_cfg.clone_cfg.clone_strategy is original_strategy


@pytest.mark.parametrize("mutation", ["variant_order", "variant_geometry"])
@_with_simulation_app
def test_source_changes_after_assignment_cannot_replace_the_placement_snapshot(mutation):
    from isaaclab_arena.scene.object_variant_assignment import (
        assign_object_variants,
        validate_object_variant_assignments,
    )

    obj = _make_object("pickup", 2)
    assignments = assign_object_variants([obj], num_envs=3)
    if mutation == "variant_order":
        obj.spawn_cfg.assets_cfg.reverse()
    else:
        obj.spawn_cfg.assets_cfg[0].size = (2.0, 2.0, 2.0)
    # The source object and the final cfg now agree, but both differ from placement's snapshot.
    scene_cfg = _make_scene([obj], num_envs=3)
    with pytest.raises(AssertionError, match="spawn variants changed after placement"):
        validate_object_variant_assignments(scene_cfg, assignments)


@_with_simulation_app
def test_shared_spawn_changes_after_assignment_cannot_replace_the_placement_snapshot():
    from isaaclab.sim import MassPropertiesCfg

    from isaaclab_arena.scene.object_variant_assignment import (
        assign_object_variants,
        validate_object_variant_assignments,
    )

    obj = _make_object("pickup", 2)
    assignments = assign_object_variants([obj], num_envs=3)
    obj.spawn_cfg.mass_props = MassPropertiesCfg(mass=2.0)
    scene_cfg = _make_scene([obj], num_envs=3)
    with pytest.raises(AssertionError, match="shared spawn settings changed after placement"):
        validate_object_variant_assignments(scene_cfg, assignments)


@_with_simulation_app
def test_source_edit_before_assignment_rejects_incompatible_rigid_body_paths(tmp_path):
    from isaaclab.sim import UsdFileCfg
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    for source_path, body_path in ((root_path, "/Asset"), (nested_path, "/Asset/Body")):
        stage = Usd.Stage.CreateNew(str(source_path))
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Asset").GetPrim())
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body)
        stage.GetRootLayer().Save()
    member = Object(name="member", object_type=ObjectType.RIGID, spawner_cfg=UsdFileCfg(usd_path=str(root_path)))
    obj = Object(name="pickup", variants=[member.as_variant(), member.as_variant()])
    obj.spawn_cfg.assets_cfg[1].usd_path = str(nested_path)
    with pytest.raises(AssertionError, match="incompatible rigid-body paths"):
        assign_object_variants([obj], num_envs=3)
    assert obj.variant_indices_by_env is None


@pytest.mark.parametrize("extra", ["unmanaged_spawner", "heterogeneous_collection", "optional_combinations"])
@_with_simulation_app
def test_assignment_rejects_unmanaged_native_variants(extra):
    from isaaclab import cloner
    from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
    from isaaclab.sim import CuboidCfg, MultiAssetSpawnerCfg

    from isaaclab_arena.scene.object_variant_assignment import (
        assign_object_variants,
        validate_object_variant_assignments,
    )

    obj = _make_object("pickup", 2)
    assignments = assign_object_variants([obj], num_envs=3)
    scene_cfg = _make_scene([obj], num_envs=3)
    multi_spawn_cfg = MultiAssetSpawnerCfg(
        assets_cfg=[CuboidCfg(size=(0.1, 0.1, 0.1)), CuboidCfg(size=(0.2, 0.2, 0.2))]
    )
    if extra == "unmanaged_spawner":
        scene_cfg.table.spawn = multi_spawn_cfg
    elif extra == "heterogeneous_collection":
        scene_cfg.collection = RigidObjectCollectionCfg(
            rigid_objects={"member": RigidObjectCfg(prim_path="{ENV_REGEX_NS}/Member", spawn=multi_spawn_cfg)}
        )
    else:
        scene_cfg.clone_cfg.clone_combinations = [cloner.InclusionSet(assets=["pickup"])]
    with pytest.raises(AssertionError):
        validate_object_variant_assignments(scene_cfg, assignments)


def _roundtrip_environment_config(scene_cfg, assignments):
    from isaaclab.scene import InteractiveSceneCfg
    from omegaconf import OmegaConf

    from isaaclab_arena.utils.configclass import make_configclass

    environment_cfg_type = make_configclass(
        "VariantEnvironmentCfg",
        [("scene", InteractiveSceneCfg, scene_cfg), ("object_variant_assignments", dict, assignments)],
    )
    env_cfg = environment_cfg_type()
    serialized = OmegaConf.to_container(OmegaConf.create(env_cfg.to_dict()), resolve=True)
    restored = deepcopy(env_cfg)
    restored.from_dict(serialized)
    return restored, serialized


@_with_simulation_app
def test_clone_strategy_survives_hydra_config_roundtrip():
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants, object_variant_clone_strategy

    objects = [_make_object("pickup", 2, True), _make_object("destination", 3, True)]
    assignments = assign_object_variants(objects, num_envs=7, seed=19)
    restored, serialized = _roundtrip_environment_config(_make_scene(objects, num_envs=7), assignments)
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
    "override",
    [
        "num_envs",
        "variant_count",
        "variant_order",
        "variant_geometry",
        "missing_object",
        "invalid_index",
        "shared_mass",
        "shared_contacts",
        "spawner_function",
    ],
)
@_with_simulation_app
def test_runtime_rejects_hydra_overrides_that_change_placement_geometry(override):
    from isaaclab.sim import CuboidCfg, MassPropertiesCfg

    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants, object_variant_clone_strategy

    obj = _make_object("pickup", 2, True)
    assignments = assign_object_variants([obj], num_envs=7, seed=19)
    restored, _ = _roundtrip_environment_config(_make_scene([obj], num_envs=7), assignments)
    if override == "num_envs":
        restored.scene.num_envs = 8
    elif override == "variant_count":
        restored.scene.pickup.spawn.assets_cfg.append(CuboidCfg(size=(2.0, 2.0, 2.0)))
    elif override == "variant_order":
        restored.scene.pickup.spawn.assets_cfg.reverse()
    elif override == "variant_geometry":
        restored.scene.pickup.spawn.assets_cfg[0].size = (2.0, 2.0, 2.0)
    elif override == "invalid_index":
        restored.object_variant_assignments["pickup"].variant_indices = (2,) * 7
    elif override == "shared_mass":
        restored.scene.pickup.spawn.mass_props = MassPropertiesCfg(mass=2.0)
    elif override == "shared_contacts":
        restored.scene.pickup.spawn.activate_contact_sensors = True
    elif override == "spawner_function":
        restored.scene.pickup.spawn.func = "custom:spawn_asset"
    else:
        restored.scene.pickup = None
    original_strategy = restored.scene.clone_cfg.clone_strategy
    with pytest.raises(AssertionError):
        with object_variant_clone_strategy(restored.scene, restored.object_variant_assignments):
            pytest.fail("Invalid placement geometry reached scene construction")
    assert restored.scene.clone_cfg.clone_strategy is original_strategy


@_with_simulation_app
def test_runtime_clone_strategy_is_restored_when_scene_construction_fails():
    from isaaclab_arena.scene.object_variant_assignment import assign_object_variants, object_variant_clone_strategy

    obj = _make_object("pickup", 2)
    assignments = assign_object_variants([obj], num_envs=3)
    scene_cfg = _make_scene([obj], num_envs=3)
    original_strategy = scene_cfg.clone_cfg.clone_strategy
    with pytest.raises(RuntimeError, match="scene construction failed"):
        with object_variant_clone_strategy(scene_cfg, assignments):
            raise RuntimeError("scene construction failed")
    assert scene_cfg.clone_cfg.clone_strategy is original_strategy
