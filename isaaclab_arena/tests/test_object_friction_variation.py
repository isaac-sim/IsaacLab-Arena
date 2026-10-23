# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check build-time object friction through normal material spawning and replay."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_object_friction_build_and_spawn(_simulation_app, asset_path, procedural, backend):
    from isaaclab.sim.utils import create_new_stage
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tests.test_asset_physics_config import _write_asset

    _write_asset(asset_path)
    obj = (
        Sphere(instance_name="ball")
        if procedural
        else Object(name="ball", usd_path=str(asset_path), object_type=ObjectType.BASE)
    )
    arena_env = IsaacLabArenaEnvironment(name="object_friction_test", scene=Scene(assets=[obj]))
    builder_cfg = ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, presets=backend)
    original = obj.object_cfg.spawn.physics_material
    _, disabled_kwargs = ArenaEnvBuilder(arena_env, builder_cfg).compose_manager_cfg()
    assert not obj.get_variation("friction").enabled
    assert obj.object_cfg.spawn.physics_material is original
    assert "ball.friction" not in disabled_kwargs["variation_recorder"]
    cfg, kwargs = ArenaEnvBuilder(
        arena_env,
        builder_cfg,
        hydra_overrides=[
            "ball.friction.enabled=true",
            "ball.friction.sampler_cfg.low=[0.6]",
            "ball.friction.sampler_cfg.high=[0.8]",
        ],
    ).compose_manager_cfg()
    record = kwargs["variation_recorder"]["ball.friction"]
    friction = float(record.sample_for_episode(0, 0)[0])
    assert 0.6 <= friction <= 0.8
    assert float(record.sample_for_episode(1, 10)[0]) == friction
    spawn = cfg.scene.ball.spawn
    stage = create_new_stage()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    spawn.func("/World/env_.*/Object", spawn)
    for index in range(2):
        colliders = [
            prim
            for prim in Usd.PrimRange(stage.GetPrimAtPath(f"/World/env_{index}/Object"))
            if prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        assert colliders
        for collider in colliders:
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            physics = UsdPhysics.MaterialAPI(material.GetPrim())
            assert physics.GetStaticFrictionAttr().Get() == pytest.approx(friction)
            assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(friction)
    if backend == "newton":
        import newton

        builder = newton.ModelBuilder()
        builder.add_usd(stage, root_path="/World/env_0/Object", load_visual_shapes=False)
        assert len(builder.shape_material_mu) > 0
        assert all(value == pytest.approx(friction) for value in builder.shape_material_mu)
    return True


@pytest.mark.parametrize("procedural", [False, True])
@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_object_friction_build_and_spawn(tmp_path, procedural, backend):
    assert run_function_with_persistent_simulation_app(
        _test_object_friction_build_and_spawn,
        asset_path=tmp_path / "object.usda",
        procedural=procedural,
        backend=backend,
    )


def _test_object_friction_replay_and_preservation(_simulation_app):
    from copy import deepcopy

    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab_physx.sim.spawners.materials import PhysxMaterialCfg, RigidBodyMaterialCfg

    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.variations.object_friction_variation import ObjectFrictionVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    usd = UsdPhysicsRigidBodyMaterialCfg(static_friction=0.2, dynamic_friction=0.1, restitution=0.3, density=500)
    physx = PhysxMaterialCfg(friction_combine_mode="multiply", compliant_contact_stiffness=100)
    materials = [None, usd, physx, [usd, physx], (usd, physx), RigidBodyMaterialCfg(restitution=0.3)]
    for original in materials:
        shared_spawn = deepcopy(Sphere.default_spawner_cfg)
        shared_spawn.physics_material = original
        obj = Sphere(instance_name="ball", spawner_cfg=shared_spawn)
        other = Sphere(instance_name="other", spawner_cfg=shared_spawn)
        variation = obj.get_variation("friction")
        recorded = []
        variation.add_sample_listener(lambda sample, env_ids: recorded.append(float(sample[0, 0])))
        for friction in (0.5, 0.9):
            variation.set_replay_sampler(lambda count, env_ids: [[friction]])
            variation.configure_at_build_time(seed=42, variation_key="ball.friction")
            material = obj.object_cfg.spawn.physics_material
            fragments = material if isinstance(material, list) else [material]
            friction_fragments = [
                f for f in fragments if isinstance(f, (UsdPhysicsRigidBodyMaterialCfg, RigidBodyMaterialCfg))
            ]
            assert len(friction_fragments) == 1
            assert friction_fragments[0].static_friction == pytest.approx(friction)
            assert friction_fragments[0].dynamic_friction == pytest.approx(friction)
            if original is usd or isinstance(original, (list, tuple, RigidBodyMaterialCfg)):
                assert friction_fragments[0].restitution == 0.3
            if original is usd or isinstance(original, (list, tuple)):
                assert friction_fragments[0].density == 500
            if original is physx or isinstance(original, (list, tuple)):
                backend_cfg = next(f for f in fragments if isinstance(f, PhysxMaterialCfg))
                assert backend_cfg.friction_combine_mode == "multiply"
                assert backend_cfg.compliant_contact_stiffness == 100
            assert other.object_cfg.spawn.physics_material == original
            assert shared_spawn.physics_material is original
        assert recorded == pytest.approx([0.5, 0.9])
        assert usd.static_friction == 0.2
        previous_spawn = obj.object_cfg.spawn
        for invalid in (-0.1, float("nan"), float("inf")):
            variation.set_replay_sampler(lambda count, env_ids: [[invalid]])
            with pytest.raises(AssertionError, match="finite and nonnegative"):
                variation.configure_at_build_time(seed=42, variation_key="ball.friction")
            assert obj.object_cfg.spawn is previous_spawn
        variation.set_replay_sampler(None)
        variation.apply_cfg(ObjectFrictionVariationCfg(sampler_cfg=UniformSamplerCfg(low=[0, 0], high=[1, 1])))
        with pytest.raises(AssertionError, match="one scalar"):
            variation.configure_at_build_time(seed=42, variation_key="ball.friction")
    return True


def test_object_friction_replay_and_preservation():
    assert run_function_with_persistent_simulation_app(_test_object_friction_replay_and_preservation)


def _test_object_friction_in_physics(_simulation_app):
    import torch

    import warp as wp

    from isaaclab_arena.assets.object_library import Sphere
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose
    from isaaclab_arena.variations.object_friction_variation import ObjectFrictionVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    sphere = Sphere(initial_pose=Pose(position_xyz=(0.0, 0.0, 1.0)))
    sphere.get_variation("friction").apply_cfg(
        ObjectFrictionVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[0.65], high=[0.65]))
    )
    arena_env = IsaacLabArenaEnvironment(name="object_friction_physics_test", scene=Scene(assets=[sphere]))
    env = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, presets="physx")
    ).make_registered()
    try:
        env.reset()
        base_env = env.unwrapped
        materials = wp.to_torch(base_env.scene["sphere"].root_view.get_material_properties()).clone()
        torch.testing.assert_close(materials[..., :2], torch.full_like(materials[..., :2], 0.65))
        torch.testing.assert_close(materials[0], materials[1])
        actions = torch.zeros((2, base_env.action_manager.total_action_dim), device=base_env.device)
        for _ in range(3):
            env.step(actions)
        env.reset()
        torch.testing.assert_close(wp.to_torch(base_env.scene["sphere"].root_view.get_material_properties()), materials)
        record = base_env.variation_recorder["sphere.friction"]
        assert record.sample_for_episode(1, base_env.get_episode_index(1))[0] == pytest.approx(0.65)
    finally:
        env.close()
    return True


def test_object_friction_in_physics():
    assert run_function_with_persistent_simulation_app(_test_object_friction_in_physics)


def _test_object_friction_preserves_usd_materials(_simulation_app, asset_dir, instanceable):
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.sim.utils import create_new_stage
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.tests.utils.prim_physics_configs import MassCfg
    from isaaclab_arena.variations.object_friction_variation import ObjectFrictionVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    geometry_path = str(asset_dir / "geometry.usda")
    geometry = Usd.Stage.CreateNew(geometry_path)
    geometry.SetDefaultPrim(UsdGeom.Xform.Define(geometry, "/Geometry").GetPrim())
    for index in range(2):
        group = UsdGeom.Xform.Define(geometry, f"/Geometry/part_{index}").GetPrim()
        collider = UsdGeom.Cube.Define(geometry, f"/Geometry/part_{index}/collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        material = UsdShade.Material.Define(geometry, f"/Geometry/material_{index}")
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(0.3)
        physics.CreateDynamicFrictionAttr(0.2)
        physics.CreateDensityAttr(500.0 + index * 400.0)
        physics.CreateRestitutionAttr(0.4 + index * 0.2)
        physx = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
        physx.CreateCompliantContactStiffnessAttr(100.0 + index * 100.0)
        physx.CreateCompliantContactDampingAttr(10.0 + index * 10.0)
        physx.CreateFrictionCombineModeAttr("multiply")
        physx.CreateRestitutionCombineModeAttr("min")
        # Exercise both inherited and direct material bindings.
        UsdShade.MaterialBindingAPI.Apply(group if index == 0 else collider).Bind(material, materialPurpose="physics")
    geometry.GetRootLayer().Save()

    asset_path = str(asset_dir / "object.usda")
    source = Usd.Stage.CreateNew(asset_path)
    root = UsdGeom.Xform.Define(source, "/Asset").GetPrim()
    source.SetDefaultPrim(root)
    UsdPhysics.RigidBodyAPI.Apply(root)
    geometry_ref = UsdGeom.Xform.Define(source, "/Asset/Geometry").GetPrim()
    geometry_ref.GetReferences().AddReference(geometry_path)
    geometry_ref.SetInstanceable(instanceable)
    source.GetRootLayer().Save()

    obj = Object(
        name="object",
        usd_path=asset_path,
        object_type=ObjectType.BASE,
        spawn_cfg_addon={"prim_physics": {".": MassCfg(mass=0.25)}},
    )
    original_spawn = obj.object_cfg.spawn
    variation = obj.get_variation("friction")
    for friction in (0.6, 0.9):
        variation.apply_cfg(
            ObjectFrictionVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[friction], high=[friction]))
        )
        variation.configure_at_build_time(seed=42, variation_key="object.friction")
    assert obj.object_cfg.spawn.make_uninstanceable
    assert not original_spawn.make_uninstanceable
    stage = create_new_stage()
    for env_index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{env_index}")
    spawn = obj.object_cfg.spawn
    spawn.func("/World/env_.*/Object", spawn)
    original_spawn.func("/World/Unvaried", original_spawn)
    for object_path, expected_friction in (
        ("/World/env_0/Object", 0.9),
        ("/World/env_1/Object", 0.9),
        ("/World/Unvaried", 0.3),
    ):
        assert UsdPhysics.MassAPI(stage.GetPrimAtPath(object_path)).GetMassAttr().Get() == pytest.approx(0.25)
        for index in range(2):
            collider = stage.GetPrimAtPath(f"{object_path}/Geometry/part_{index}/collider")
            if expected_friction == 0.9:
                assert not collider.IsInstanceProxy()
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            assert str(material.GetPath()).startswith(object_path + "/")
            physics = UsdPhysics.MaterialAPI(material.GetPrim())
            assert physics.GetStaticFrictionAttr().Get() == pytest.approx(expected_friction)
            assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.9 if expected_friction == 0.9 else 0.2)
            assert physics.GetDensityAttr().Get() == pytest.approx(500.0 + index * 400.0)
            assert physics.GetRestitutionAttr().Get() == pytest.approx(0.4 + index * 0.2)
            physx = PhysxSchema.PhysxMaterialAPI(material.GetPrim())
            assert physx.GetCompliantContactStiffnessAttr().Get() == pytest.approx(100.0 + index * 100.0)
            assert physx.GetCompliantContactDampingAttr().Get() == pytest.approx(10.0 + index * 10.0)
            assert physx.GetFrictionCombineModeAttr().Get() == "multiply"
            assert physx.GetRestitutionCombineModeAttr().Get() == "min"
    for index in range(2):
        material = geometry.GetPrimAtPath(f"/Geometry/material_{index}")
        assert UsdPhysics.MaterialAPI(material).GetStaticFrictionAttr().Get() == pytest.approx(0.3)

    # Switching to an explicit spawn material must not retain the previous friction override.
    obj.object_cfg.spawn.physics_material = UsdPhysicsRigidBodyMaterialCfg(density=600.0, restitution=0.1)
    variation.set_replay_sampler(lambda count, env_ids: [[1.1]])
    variation.configure_at_build_time(seed=42, variation_key="object.friction")
    spawn = obj.object_cfg.spawn
    spawn.func("/World/Configured", spawn)
    collider = stage.GetPrimAtPath("/World/Configured/Geometry/part_0/collider")
    material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
    physics = UsdPhysics.MaterialAPI(material.GetPrim())
    assert physics.GetStaticFrictionAttr().Get() == pytest.approx(1.1)
    assert physics.GetDensityAttr().Get() == pytest.approx(600.0)
    assert physics.GetRestitutionAttr().Get() == pytest.approx(0.1)
    assert UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/Configured")).GetMassAttr().Get() == pytest.approx(0.25)
    return True


@pytest.mark.parametrize("instanceable", [False, True])
def test_object_friction_preserves_usd_materials(tmp_path, instanceable):
    assert run_function_with_persistent_simulation_app(
        _test_object_friction_preserves_usd_materials, asset_dir=tmp_path, instanceable=instanceable
    )


def _test_object_friction_binding_precedence(_simulation_app, asset_path, binding_kind, configured_material):
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from isaaclab.sim.utils import create_new_stage
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.variations.object_friction_variation import ObjectFrictionVariationCfg
    from isaaclab_arena.variations.uniform_sampler import UniformSamplerCfg

    source = Usd.Stage.CreateNew(str(asset_path))
    source.SetDefaultPrim(UsdGeom.Xform.Define(source, "/Asset").GetPrim())
    original = UsdShade.Material.Define(source, "/Asset/Original")
    physics = UsdPhysics.MaterialAPI.Apply(original.GetPrim())
    physics.CreateStaticFrictionAttr(0.3)
    physics.CreateDynamicFrictionAttr(0.2)
    for name in ("free", "blocked"):
        UsdGeom.Xform.Define(source, f"/Asset/{name}")
        collider = UsdGeom.Cube.Define(source, f"/Asset/{name}/collider").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
    parent = source.GetPrimAtPath("/Asset/blocked")
    blocked = source.GetPrimAtPath("/Asset/blocked/collider")
    if binding_kind in ("local_collection", "ancestor_collection"):
        owner = blocked if binding_kind == "local_collection" else parent
        collection = Usd.CollectionAPI.Apply(owner, "contacts")
        collection.CreateIncludesRel().SetTargets([blocked.GetPath()])
        UsdShade.MaterialBindingAPI.Apply(owner).Bind(
            collection, original, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
    else:
        strength = (
            UsdShade.Tokens.weakerThanDescendants
            if binding_kind == "weak_physics"
            else UsdShade.Tokens.strongerThanDescendants
        )
        purpose = "" if binding_kind == "strong_all_purpose" else "physics"
        UsdShade.MaterialBindingAPI.Apply(parent).Bind(original, bindingStrength=strength, materialPurpose=purpose)
    source.GetRootLayer().Save()
    addon = {}
    if configured_material:
        addon["physics_material"] = UsdPhysicsRigidBodyMaterialCfg(restitution=0.4)
    obj = Object(name="object", usd_path=str(asset_path), object_type=ObjectType.BASE, spawn_cfg_addon=addon)
    variation = obj.get_variation("friction")
    variation.apply_cfg(ObjectFrictionVariationCfg(enabled=True, sampler_cfg=UniformSamplerCfg(low=[0.9], high=[0.9])))
    variation.configure_at_build_time(seed=42, variation_key="object.friction")
    stage = create_new_stage()
    for env_index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{env_index}")
    spawn = obj.object_cfg.spawn
    if binding_kind in ("strong_physics", "local_collection", "ancestor_collection"):
        with pytest.raises(AssertionError, match="Cannot vary friction") as error:
            spawn.func("/World/env_.*/Object", spawn)
        assert "/World/env_0/Object/blocked/collider" in str(error.value)
        assert "material:binding:" in str(error.value)
        if binding_kind == "local_collection":
            assert "Remove the conflicting physics collection binding" in str(error.value)
        else:
            assert "weakerThanDescendants" in str(error.value)
        # Reject before authoring any collider-local friction override or cloning the object.
        for name in ("free", "blocked"):
            assert not stage.GetPrimAtPath(f"/World/env_0/Object/{name}/collider/ArenaFrictionMaterial")
        assert not stage.GetPrimAtPath("/World/env_1/Object")
    else:
        spawn.func("/World/env_.*/Object", spawn)
        for env_index in range(2):
            for name in ("free", "blocked"):
                collider = stage.GetPrimAtPath(f"/World/env_{env_index}/Object/{name}/collider")
                material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
                physics = UsdPhysics.MaterialAPI(material.GetPrim())
                assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.9)
                assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.9)
    assert UsdPhysics.MaterialAPI(original.GetPrim()).GetStaticFrictionAttr().Get() == pytest.approx(0.3)
    return True


@pytest.mark.parametrize(
    "binding_kind",
    [
        "strong_physics",
        "local_collection",
        "ancestor_collection",
        "weak_physics",
        "strong_all_purpose",
    ],
)
@pytest.mark.parametrize("configured_material", [False, True])
def test_object_friction_binding_precedence(tmp_path, binding_kind, configured_material):
    assert run_function_with_persistent_simulation_app(
        _test_object_friction_binding_precedence,
        asset_path=tmp_path / "object.usda",
        binding_kind=binding_kind,
        configured_material=configured_material,
    )
