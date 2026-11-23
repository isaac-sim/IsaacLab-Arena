# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check gripper material selection, cloning, and physics import on the DROID USD."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_gripper_material_build_and_spawn(_simulation_app, backend):
    from isaaclab.sim.utils import create_new_stage
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.embodiments.droid.droid import (
        _DROID_GRIPPER_PAD_COLLIDER_PATHS,
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    robot = DroidAbsoluteJointPositionEmbodiment()
    arena_env = IsaacLabArenaEnvironment(name="gripper_material_test", scene=Scene(assets=[]), embodiment=robot)
    builder_cfg = ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, presets=backend)
    # Registration is opt-in, and the ordinary builder leaves disabled materials alone.
    disabled_cfg, disabled_kwargs = ArenaEnvBuilder(arena_env, builder_cfg).compose_manager_cfg()
    assert not robot.get_variation("gripper_material").enabled
    assert not getattr(disabled_cfg.scene.robot.spawn, "prim_physics", {})
    assert f"{robot.name}.gripper_material" not in disabled_kwargs["variation_recorder"]

    overrides = [
        f"{robot.name}.gripper_material.enabled=true",
        f"{robot.name}.gripper_material.material_names=[high_friction]",
        f"{robot.name}.gripper_material.presets.high_friction.static_friction=1.4",
        f"{robot.name}.gripper_material.presets.high_friction.dynamic_friction=0.83",
    ]
    cfg, kwargs = ArenaEnvBuilder(arena_env, builder_cfg, hydra_overrides=overrides).compose_manager_cfg()
    record = kwargs["variation_recorder"][f"{robot.name}.gripper_material"]
    assert record.sample_for_episode(0, 0) == record.sample_for_episode(1, 10) == "high_friction"
    assert record.cfg.presets["high_friction"]["static_friction"] == 1.4
    spawn = cfg.scene.robot.spawn
    assert set(spawn.prim_physics) == set(_DROID_GRIPPER_PAD_COLLIDER_PATHS)

    stage = create_new_stage()
    for index in range(2):
        UsdGeom.Xform.Define(stage, f"/World/env_{index}")
    spawn.func("/World/env_.*/Robot", spawn)
    for index in range(2):
        for path in _DROID_GRIPPER_PAD_COLLIDER_PATHS:
            collider = stage.GetPrimAtPath(f"/World/env_{index}/Robot/{path}")
            material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
            assert material.GetPath() == collider.GetPath().AppendChild("ArenaContactMaterial")
            physics = UsdPhysics.MaterialAPI(material.GetPrim())
            assert physics.GetStaticFrictionAttr().Get() == pytest.approx(1.4)
            assert physics.GetDynamicFrictionAttr().Get() == pytest.approx(0.83)
            assert physics.GetRestitutionAttr().Get() == pytest.approx(0.0)

    # No other collider acquires the pad material, and no source USD is edited.
    source = Usd.Stage.Open(spawn.usd_path)
    source_root = source.GetDefaultPrim()
    tuned_root = stage.GetPrimAtPath("/World/env_0/Robot")
    for prim in Usd.PrimRange(tuned_root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        relative = prim.GetPath().MakeRelativePath(tuned_root.GetPath())
        original = source.GetPrimAtPath(source_root.GetPath().AppendPath(relative))
        assert not original.GetChild("ArenaContactMaterial")
        if str(relative) not in _DROID_GRIPPER_PAD_COLLIDER_PATHS:
            assert not prim.GetChild("ArenaContactMaterial")

    if backend == "newton":
        import newton

        model_builder = newton.ModelBuilder()
        model_builder.add_usd(
            stage, root_path="/World/env_0/Robot", collapse_fixed_joints=False, load_visual_shapes=False
        )
        for path in _DROID_GRIPPER_PAD_COLLIDER_PATHS:
            indices = []
            for index, label in enumerate(model_builder.shape_label):
                if str(label).startswith(f"/World/env_0/Robot/{path}"):
                    indices.append(index)
            assert indices, f"Newton did not import fingertip collider {path}"
            # Newton's USD importer uses dynamic friction as its single Coulomb coefficient.
            for index in indices:
                assert model_builder.shape_material_mu[index] == pytest.approx(0.83)
    return True


@pytest.mark.parametrize("backend", ["physx", "newton"])
def test_gripper_material_build_and_spawn(backend):
    assert run_function_with_persistent_simulation_app(_test_gripper_material_build_and_spawn, backend=backend)


def _test_gripper_material_replay_and_validation(_simulation_app):
    from isaaclab_arena.embodiments.droid.droid import (
        _DROID_GRIPPER_PAD_COLLIDER_PATHS,
        DroidAbsoluteJointPositionEmbodiment,
    )
    from isaaclab_arena.tests.utils.prim_physics_configs import MassCfg
    from isaaclab_arena.variations.gripper_material_variation import (
        GripperMaterialVariation,
        GripperMaterialVariationCfg,
    )

    robot = DroidAbsoluteJointPositionEmbodiment()
    other_robot = DroidAbsoluteJointPositionEmbodiment()
    for paths in ((), ("pad", "pad"), ("../pad",), ("/Robot/pad",), ("pad.size",), "pad"):
        with pytest.raises(AssertionError):
            GripperMaterialVariation(robot, collider_paths=paths)
    assert robot.spawn_cfg_addon == {}
    variation = robot.get_variation("gripper_material")
    variation.apply_cfg(
        GripperMaterialVariationCfg(
            enabled=True,
            presets={
                "pad_a": dict(static_friction=0.4, dynamic_friction=0.3),
                "pad_b": dict(static_friction=0.9, dynamic_friction=0.8),
            },
        )
    )
    draws = iter((["pad_a"], ["pad_b"]))
    variation.set_replay_sampler(lambda count, env_ids: next(draws))
    robot.spawn_cfg_addon["robot"] = {"visible": False, "prim_physics": {"panda_link0": MassCfg()}}
    for expected in (0.4, 0.9):
        variation.configure_at_build_time(seed=42, variation_key=f"{robot.name}.gripper_material")
        spawn = robot.get_scene_cfg().robot.spawn
        assert spawn.visible is False
        assert spawn.prim_physics["panda_link0"].mass == 0.25
        for path in _DROID_GRIPPER_PAD_COLLIDER_PATHS:
            assert spawn.prim_physics[path].material.static_friction == expected
    assert other_robot.spawn_cfg_addon == {}
    assert "pad_a" not in other_robot.get_variation("gripper_material").cfg.presets
    assert variation.cfg.presets["pad_a"]["static_friction"] == 0.4
    variation.set_replay_sampler(None)

    invalid_configs = [
        GripperMaterialVariationCfg(presets={}),
        GripperMaterialVariationCfg(material_names=["unknown"]),
        GripperMaterialVariationCfg(material_names=["low_friction", "low_friction"]),
        GripperMaterialVariationCfg(presets={"invalid": dict(static_friction=float("nan"), dynamic_friction=0.1)}),
        GripperMaterialVariationCfg(presets={"invalid": dict(static_friction=0.5, dynamic_friction=-0.1)}),
        GripperMaterialVariationCfg(presets={"invalid": dict(static_friction=0.5, dynamic_friction=1.0)}),
        GripperMaterialVariationCfg(
            presets={"invalid": dict(static_friction=0.5, dynamic_friction=0.1, restitution=1.1)}
        ),
    ]
    for cfg in invalid_configs:
        variation.apply_cfg(cfg)
        with pytest.raises(AssertionError):
            variation.configure_at_build_time(seed=42, variation_key=f"{robot.name}.gripper_material")
        assert (
            robot.spawn_cfg_addon["robot"]["prim_physics"][
                _DROID_GRIPPER_PAD_COLLIDER_PATHS[0]
            ].material.static_friction
            == 0.9
        )

    variation.set_replay_sampler(None)
    variation.apply_cfg(GripperMaterialVariationCfg())
    robot.spawn_cfg_addon["robot"]["prim_physics"][_DROID_GRIPPER_PAD_COLLIDER_PATHS[0]] = MassCfg()
    with pytest.raises(AssertionError, match="replace another physics override"):
        variation.configure_at_build_time(seed=42, variation_key=f"{robot.name}.gripper_material")
    return True


def test_gripper_material_replay_and_validation():
    assert run_function_with_persistent_simulation_app(_test_gripper_material_replay_and_validation)


def _test_gripper_material_in_physics(_simulation_app):
    import torch

    import warp as wp

    from isaaclab_arena.embodiments.droid.droid import DroidAbsoluteJointPositionEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.variations.gripper_material_variation import GripperMaterialVariationCfg

    robot = DroidAbsoluteJointPositionEmbodiment()
    robot.get_variation("gripper_material").apply_cfg(
        GripperMaterialVariationCfg(enabled=True, material_names=["high_friction"])
    )
    arena_env = IsaacLabArenaEnvironment(name="gripper_material_physics_test", scene=Scene(assets=[]), embodiment=robot)
    env = ArenaEnvBuilder(
        arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False, presets="physx")
    ).make_registered()
    try:
        env.reset()
        base_env = env.unwrapped
        materials = wp.to_torch(base_env.scene["robot"].root_view.get_material_properties()).clone()
        pad_materials = torch.isclose(materials[..., 0], materials.new_tensor(1.2))
        # Each pad can decompose into multiple convex physics shapes.
        assert torch.all(pad_materials.sum(dim=1) >= 2)
        assert torch.all((~pad_materials).any(dim=1))
        torch.testing.assert_close(materials[0], materials[1])
        assert torch.allclose(materials[..., 1][pad_materials], torch.ones_like(materials[..., 1][pad_materials]))
        actions = torch.zeros((2, base_env.action_manager.total_action_dim), device=base_env.device)
        for _ in range(3):
            env.step(actions)
        env.reset()
        torch.testing.assert_close(wp.to_torch(base_env.scene["robot"].root_view.get_material_properties()), materials)
        record = base_env.variation_recorder[f"{robot.name}.gripper_material"]
        assert record.sample_for_episode(1, base_env.get_episode_index(1)) == "high_friction"
    finally:
        env.close()
    return True


def test_gripper_material_in_physics():
    assert run_function_with_persistent_simulation_app(_test_gripper_material_in_physics)


def _test_contact_material_preserves_properties(_simulation_app, inherited_binding):
    from isaaclab.sim.spawners.materials.physics_materials_cfg import UsdPhysicsRigidBodyMaterialCfg
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.physics_spawner import apply_prim_physics
    from isaaclab_arena.variations.gripper_material_variation import _GripperPadMaterialCfg

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
    original = UsdShade.Material.Define(stage, "/Robot/SharedMaterial")
    physics = UsdPhysics.MaterialAPI.Apply(original.GetPrim())
    physics.CreateStaticFrictionAttr(0.9)
    physics.CreateDynamicFrictionAttr(0.8)
    physics.CreateDensityAttr(1200.0)
    physx = PhysxSchema.PhysxMaterialAPI.Apply(original.GetPrim())
    physx.CreateCompliantContactStiffnessAttr(250.0)
    physx.CreateRestitutionCombineModeAttr("min")
    physx.CreateFrictionCombineModeAttr("multiply")
    for name in ("pad", "other"):
        collider = UsdGeom.Cube.Define(stage, f"/Robot/{name}").GetPrim()
        UsdPhysics.CollisionAPI.Apply(collider)
        if not inherited_binding:
            UsdShade.MaterialBindingAPI.Apply(collider).Bind(original, materialPurpose="physics")
    if inherited_binding:
        # A strong all-purpose fallback must not block a more specific physics binding.
        UsdShade.MaterialBindingAPI.Apply(root).Bind(original, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    apply_prim_physics(
        root,
        {
            "pad": _GripperPadMaterialCfg(
                material=UsdPhysicsRigidBodyMaterialCfg(static_friction=0.4, dynamic_friction=0.3)
            )
        },
    )
    pad = stage.GetPrimAtPath("/Robot/pad")
    selected, _ = UsdShade.MaterialBindingAPI(pad).ComputeBoundMaterial("physics")
    selected_physics = UsdPhysics.MaterialAPI(selected.GetPrim())
    assert selected_physics.GetStaticFrictionAttr().Get() == pytest.approx(0.4)
    assert selected_physics.GetDensityAttr().Get() == pytest.approx(1200.0)
    selected_physx = PhysxSchema.PhysxMaterialAPI(selected.GetPrim())
    assert selected_physx.GetCompliantContactStiffnessAttr().Get() == pytest.approx(250.0)
    assert selected_physx.GetRestitutionCombineModeAttr().Get() == "min"
    assert selected_physx.GetFrictionCombineModeAttr().Get() == "multiply"
    assert physics.GetStaticFrictionAttr().Get() == pytest.approx(0.9)
    other, _ = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath("/Robot/other")).ComputeBoundMaterial("physics")
    assert other.GetPath() == original.GetPath()

    clone = UsdGeom.Xform.Define(stage, "/Clone").GetPrim()
    clone.GetReferences().AddInternalReference(root.GetPath())
    cloned_pad = stage.GetPrimAtPath("/Clone/pad")
    cloned_material, _ = UsdShade.MaterialBindingAPI(cloned_pad).ComputeBoundMaterial("physics")
    assert str(cloned_material.GetPath()) == "/Clone/pad/ArenaContactMaterial"
    assert UsdPhysics.MaterialAPI(cloned_material.GetPrim()).GetDensityAttr().Get() == pytest.approx(1200.0)
    return True


@pytest.mark.parametrize("inherited_binding", [False, True])
def test_contact_material_preserves_properties(inherited_binding):
    assert run_function_with_persistent_simulation_app(
        _test_contact_material_preserves_properties, inherited_binding=inherited_binding
    )


def _test_contact_material_rejects_shadowed_bindings(_simulation_app):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.physics_spawner import apply_prim_physics
    from isaaclab_arena.variations.gripper_material_variation import _GripperPadMaterialCfg

    for collection_binding in (False, True):
        stage = Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
        material = UsdShade.Material.Define(stage, "/Robot/Material")
        UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        for name in ("valid", "blocked"):
            collider = UsdGeom.Cube.Define(stage, f"/Robot/{name}").GetPrim()
            UsdPhysics.CollisionAPI.Apply(collider)
        blocked = stage.GetPrimAtPath("/Robot/blocked")
        if collection_binding:
            collection = Usd.CollectionAPI.Apply(blocked, "contacts")
            collection.CreateIncludesRel().SetTargets([blocked.GetPath()])
            UsdShade.MaterialBindingAPI.Apply(blocked).Bind(collection, material, materialPurpose="physics")
        else:
            UsdShade.MaterialBindingAPI.Apply(root).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
            )
        with pytest.raises(AssertionError, match="binding"):
            apply_prim_physics(root, {"valid": _GripperPadMaterialCfg(), "blocked": _GripperPadMaterialCfg()})
        # Validation must fail before even the first material is authored.
        for name in ("valid", "blocked"):
            assert not stage.GetPrimAtPath(f"/Robot/{name}/ArenaContactMaterial")
    return True


def test_contact_material_rejects_shadowed_bindings():
    assert run_function_with_persistent_simulation_app(_test_contact_material_rejects_shadowed_bindings)
