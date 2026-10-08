# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _write_asset(source_path, body_path, default_path="/Asset", root_scale=1.0):
    """Write a rigid cube with an ancestor transform and a sibling material library."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = Usd.Stage.CreateNew(str(source_path))
    for prefix in Sdf.Path(body_path).GetPrefixes():
        UsdGeom.Xform.Define(stage, prefix)
    root = stage.GetPrimAtPath(default_path)
    stage.SetDefaultPrim(root)
    UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3f(root_scale))
    body = stage.GetPrimAtPath(body_path)
    UsdPhysics.RigidBodyAPI.Apply(body)
    UsdGeom.Cube.Define(stage, body_path + "/Cube").CreateSizeAttr().Set(2.0)
    if body != root:
        parent = body.GetParent()
        if parent != root:
            UsdGeom.Xformable(parent).AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.0))
    material = UsdShade.Material.Define(stage, default_path + "/Looks/Material")
    shader = UsdShade.Shader.Define(stage, default_path + "/Looks/Material/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(root).Bind(material)
    # Unreferenced top-level content must never enter the prepared asset.
    auxiliary = UsdGeom.Xform.Define(stage, "/Unrelated").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(auxiliary)
    stage.GetRootLayer().Save()
    return stage


def _make_spawn_cfg(source_path, scale=(1.0, 1.0, 1.0), **spawn_options):
    from isaaclab.sim import UsdFileCfg

    return UsdFileCfg(usd_path=str(source_path), scale=scale, **spawn_options)


def _test_compatible_layouts_keep_native_scales(simulation_app, tmp_path):
    import torch

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    source_path = tmp_path / "compatible.usda"
    _write_asset(source_path, "/Asset/Body")
    spawn_configs = [
        _make_spawn_cfg(source_path, scale=(1.0, 1.0, 1.0)),
        _make_spawn_cfg(source_path, scale=(2.0, 3.0, 4.0), semantic_tags=[("class", "large")]),
    ]
    with patch("isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir") as cache_directory:
        prepared = prepare_rigid_object_variants(spawn_configs)
        cache_directory.assert_not_called()
    assert [cfg.usd_path for cfg in prepared] == [str(source_path)] * 2
    assert [cfg.scale for cfg in prepared] == [(1.0, 1.0, 1.0), (2.0, 3.0, 4.0)]
    assert prepared[1].semantic_tags == [("class", "large")]
    prepared[1].semantic_tags.append(("color", "red"))
    assert spawn_configs[1].semantic_tags == [("class", "large")]
    assert torch.allclose(
        ObjectGeometry(spawn_configs[1], ObjectType.RIGID).get_bounding_box().size, torch.tensor([[4.0, 6.0, 8.0]])
    )
    return True


def _test_preparation_preserves_geometry_and_materials(simulation_app, tmp_path):
    import torch

    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    source_paths = [
        tmp_path / "root.usda",
        tmp_path / "nested.usda",
        tmp_path / "nested_default.usda",
    ]
    _write_asset(source_paths[0], "/Asset", root_scale=3.0)
    nested_stage = _write_asset(source_paths[1], "/Asset/Group/Body", root_scale=3.0)
    _write_asset(source_paths[2], "/Outer/Asset", default_path="/Outer/Asset", root_scale=3.0)
    geometry_path = tmp_path / "geometry.usda"
    geometry_stage = Usd.Stage.CreateNew(str(geometry_path))
    geometry_root = UsdGeom.Xform.Define(geometry_stage, "/Geometry").GetPrim()
    geometry_stage.SetDefaultPrim(geometry_root)
    UsdGeom.Cube.Define(geometry_stage, "/Geometry/ReferencedCube").CreateSizeAttr().Set(4.0)
    geometry_stage.GetRootLayer().Save()
    nested_stage.GetPrimAtPath("/Asset/Group/Body").GetReferences().AddReference("geometry.usda")
    texture_path = tmp_path / "texture.png"
    texture_path.write_bytes(b"texture-path-test")
    shader = UsdShade.Shader(nested_stage.GetPrimAtPath("/Asset/Looks/Material/Shader"))
    shader.CreateInput("texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("texture.png"))
    inherited_material = UsdShade.Material(nested_stage.GetPrimAtPath("/Asset/Looks/Material"))
    UsdShade.MaterialBindingAPI(nested_stage.GetDefaultPrim()).Bind(
        inherited_material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )
    descendant_material = UsdShade.Material.Define(nested_stage, "/Asset/Looks/DescendantMaterial")
    source_cube = nested_stage.GetPrimAtPath("/Asset/Group/Body/Cube")
    UsdShade.MaterialBindingAPI.Apply(source_cube).Bind(descendant_material)
    assert UsdShade.MaterialBindingAPI(source_cube).ComputeBoundMaterial()[0].GetPath() == inherited_material.GetPath()
    nested_stage.GetRootLayer().Save()
    spawn_configs = [_make_spawn_cfg(path, scale=(2.0, 2.0, 2.0)) for path in source_paths]
    original_content = [path.read_bytes() for path in source_paths]
    original_bounds = [ObjectGeometry(spawn_cfg, ObjectType.RIGID).get_bounding_box() for spawn_cfg in spawn_configs]
    with patch(
        "isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir",
        return_value=tmp_path,
    ):
        prepared = prepare_rigid_object_variants(spawn_configs)
    for source_path, content, spawn_cfg, original_bound in zip(
        source_paths, original_content, prepared, original_bounds
    ):
        assert source_path.read_bytes() == content
        assert spawn_cfg.scale == (2.0, 2.0, 2.0)
        prepared_geometry = ObjectGeometry(spawn_cfg, ObjectType.RIGID)
        assert prepared_geometry.get_contact_body_path() == "/rigid_body"
        prepared_bound = prepared_geometry.get_bounding_box()
        assert torch.allclose(original_bound.min_point, prepared_bound.min_point)
        assert torch.allclose(original_bound.max_point, prepared_bound.max_point)
        stage = Usd.Stage.CreateInMemory()
        holder = UsdGeom.Xform.Define(stage, "/World/Object").GetPrim()
        holder.GetReferences().AddReference(spawn_cfg.usd_path)
        bodies = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        assert [str(body.GetPath()) for body in bodies] == ["/World/Object/rigid_body"]
        material, _ = UsdShade.MaterialBindingAPI(bodies[0]).ComputeBoundMaterial()
        assert material
        connection = material.GetSurfaceOutput().GetAttr().GetConnections()[0]
        assert stage.GetPropertyAtPath(connection)
        if source_path == source_paths[1]:
            assert stage.GetPrimAtPath("/World/Object/rigid_body/ReferencedCube")
            prepared_shader = UsdShade.Shader(stage.GetPrimAtPath(connection.GetPrimPath()))
            assert prepared_shader.GetInput("texture").Get().resolvedPath == str(texture_path)
            prepared_cube = stage.GetPrimAtPath("/World/Object/rigid_body/Cube")
            assert UsdShade.MaterialBindingAPI(prepared_cube).ComputeBoundMaterial()[0].GetPath() == material.GetPath()
    return True


def _test_cache_identity_and_native_addons(simulation_app, tmp_path):
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg
    from isaaclab_arena.assets.physics_spawner import make_usd_spawn_cfg_with_prim_physics
    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    first_directory = tmp_path / "first"
    second_directory = tmp_path / "second"
    first_directory.mkdir()
    second_directory.mkdir()
    first_path = first_directory / "same_name.usda"
    second_path = second_directory / "same_name.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(first_path, "/Asset")
    _write_asset(second_path, "/Asset")
    _write_asset(nested_path, "/Asset/Group/Body")
    spawn_cfg = make_usd_spawn_cfg_with_prim_physics(
        UsdFileCfg(usd_path=str(nested_path), scale=(2.0, 2.0, 2.0)),
        {"Group/Body": UsdPrimSpawnPhysicsCfg()},
    )
    spawn_configs = [_make_spawn_cfg(first_path), _make_spawn_cfg(second_path), spawn_cfg]
    with patch(
        "isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir",
        return_value=tmp_path,
    ):
        prepared = prepare_rigid_object_variants(spawn_configs)
        repeated = prepare_rigid_object_variants(spawn_configs)
        spawn_configs[2].scale = (4.0, 4.0, 4.0)
        different_scale = prepare_rigid_object_variants(spawn_configs)
    assert prepared[0].usd_path != prepared[1].usd_path
    assert [cfg.usd_path for cfg in repeated] == [cfg.usd_path for cfg in prepared]
    assert different_scale[2].usd_path == prepared[2].usd_path
    assert different_scale[2].scale == (4.0, 4.0, 4.0)
    assert set(prepared[2].prim_physics) == {"rigid_body"}
    assert set(spawn_configs[2].prim_physics) == {"Group/Body"}
    assert prepared[2].func is spawn_cfg.func
    return True


def _test_variants_select_bounds_and_reject_multiple_bodies(simulation_app, tmp_path):
    import torch

    import pytest
    from pxr import UsdGeom, UsdPhysics

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    source_path = tmp_path / "variants.usda"
    stage = _write_asset(source_path, "/Asset")
    stage.GetPrimAtPath("/Asset/Cube").GetAttribute("size").Clear()
    variant_set = stage.GetDefaultPrim().GetVariantSets().AddVariantSet("size")
    for name, size in (("small", 1.0), ("large", 4.0)):
        variant_set.AddVariant(name)
        variant_set.SetVariantSelection(name)
        with variant_set.GetVariantEditContext():
            stage.GetPrimAtPath("/Asset/Cube").GetAttribute("size").Set(size)
    variant_set.SetVariantSelection("small")
    stage.GetRootLayer().Save()
    spawn_cfg = _make_spawn_cfg(source_path, variants={"size": "large"})
    assert torch.allclose(
        ObjectGeometry(spawn_cfg, ObjectType.RIGID).get_bounding_box().size, torch.tensor([[4.0, 4.0, 4.0]])
    )
    assert variant_set.GetVariantSelection() == "small"
    another_body = UsdGeom.Xform.Define(stage, "/Asset/AnotherBody").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(another_body)
    stage.GetRootLayer().Save()
    with pytest.raises(AssertionError, match="exactly one rigid body"):
        prepare_rigid_object_variants([_make_spawn_cfg(source_path)])
    return True


def _test_preparation_retains_native_fragment_targets(simulation_app, tmp_path):
    from isaaclab.sim.schemas import MassCfg, UsdPhysicsRigidBodyCfg
    from isaaclab.sim.utils import use_stage
    from isaaclab.utils.string import string_to_callable
    from pxr import Usd, UsdPhysics

    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset")
    nested_stage = _write_asset(nested_path, "/Asset/Group/Body")
    UsdPhysics.MassAPI.Apply(nested_stage.GetPrimAtPath("/Asset/Group/Body"))
    nested_stage.GetRootLayer().Save()
    spawn_configs = [
        _make_spawn_cfg(
            root_path, rigid_props=UsdPhysicsRigidBodyCfg(kinematic_enabled=True), mass_props=MassCfg(mass=2.0)
        ),
        _make_spawn_cfg(
            nested_path, rigid_props=UsdPhysicsRigidBodyCfg(kinematic_enabled=True), mass_props=MassCfg(mass=3.0)
        ),
    ]
    with patch("isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir", return_value=tmp_path):
        prepared = prepare_rigid_object_variants(spawn_configs)
    stage = Usd.Stage.CreateInMemory()
    with use_stage(stage):
        for index, spawn_cfg in enumerate(prepared):
            spawn_function = string_to_callable(spawn_cfg.func)
            spawn_function(f"/World/variant_{index}", spawn_cfg)
            body = stage.GetPrimAtPath(f"/World/variant_{index}/rigid_body")
            assert UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get()
            assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == index + 2.0
    return True


def _test_rigid_bounds_follow_body_pose_frame(simulation_app, tmp_path):
    import numpy as np
    import torch
    import trimesh

    from pxr import Gf, UsdGeom

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset")
    nested_stage = _write_asset(nested_path, "/Asset/Group/Body")
    UsdGeom.Xformable(nested_stage.GetPrimAtPath("/Asset/Group/Body")).AddRotateZOp().Set(90.0)
    nested_stage.RemovePrim("/Asset/Group/Body/Cube")
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh_prim = UsdGeom.Mesh.Define(nested_stage, "/Asset/Group/Body/Cube")
    mesh_prim.CreatePointsAttr().Set(cube.vertices.tolist())
    mesh_prim.CreateFaceVertexCountsAttr().Set([3] * len(cube.faces))
    mesh_prim.CreateFaceVertexIndicesAttr().Set(cube.faces.reshape(-1).tolist())
    UsdGeom.Xformable(nested_stage.GetPrimAtPath("/Asset/Group/Body/Cube")).AddScaleOp().Set(Gf.Vec3f(1.0, 2.0, 3.0))
    nested_stage.GetRootLayer().Save()
    source_spawn_cfg = _make_spawn_cfg(nested_path, scale=(2.0, 2.0, 2.0))
    with patch("isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir", return_value=tmp_path):
        compatible = prepare_rigid_object_variants([source_spawn_cfg, source_spawn_cfg])
        incompatible = prepare_rigid_object_variants([_make_spawn_cfg(root_path), source_spawn_cfg])
    for spawn_cfg in (source_spawn_cfg, compatible[0], incompatible[1]):
        prepared_geometry = ObjectGeometry(spawn_cfg, ObjectType.RIGID)
        bounds = prepared_geometry.get_bounding_box()
        assert torch.allclose(bounds.min_point, torch.tensor([[-2.0, -4.0, -6.0]]), atol=1e-6)
        assert torch.allclose(bounds.max_point, torch.tensor([[2.0, 4.0, 6.0]]), atol=1e-6)
        mesh = prepared_geometry.get_collision_mesh()
        assert np.allclose(mesh.bounds, [[-2.0, -4.0, -6.0], [2.0, 4.0, 6.0]])
        mesh.vertices[:] = 0.0
        assert np.allclose(prepared_geometry.get_collision_mesh().extents, [4.0, 8.0, 12.0])
    assert (
        ObjectGeometry(source_spawn_cfg, ObjectType.RIGID).get_collision_mesh(excluded_prim_paths=("/Asset/Group",))
        is None
    )
    return True


def _test_mesh_exclusions_inside_instances_preserve_source(simulation_app, tmp_path):
    import numpy as np
    import trimesh

    from isaaclab.sim import UsdFileCfg
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.assets.object_type import ObjectType

    geometry_path = tmp_path / "geometry.usda"
    geometry_stage = Usd.Stage.CreateNew(str(geometry_path))
    geometry_root = UsdGeom.Xform.Define(geometry_stage, "/Geometry").GetPrim()
    geometry_stage.SetDefaultPrim(geometry_root)
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    for name, position in (("Keep", 0.0), ("Exclude", 10.0)):
        mesh = UsdGeom.Mesh.Define(geometry_stage, f"/Geometry/{name}")
        mesh.CreatePointsAttr().Set(cube.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr().Set([3] * len(cube.faces))
        mesh.CreateFaceVertexIndicesAttr().Set(cube.faces.reshape(-1).tolist())
        mesh.AddTranslateOp().Set(Gf.Vec3d(position, 0.0, 0.0))
    geometry_stage.GetRootLayer().Save()

    source_path = tmp_path / "instances.usda"
    source_stage = Usd.Stage.CreateNew(str(source_path))
    source_root = UsdGeom.Xform.Define(source_stage, "/Asset").GetPrim()
    source_stage.SetDefaultPrim(source_root)
    instance = UsdGeom.Xform.Define(source_stage, "/Asset/Instance").GetPrim()
    instance.GetReferences().AddReference("geometry.usda")
    instance.SetInstanceable(True)
    source_stage.GetRootLayer().Save()
    original_content = source_path.read_bytes()

    geometry = ObjectGeometry(UsdFileCfg(usd_path=str(source_path)), ObjectType.BASE)
    excluded_mesh = geometry.get_collision_mesh(excluded_prim_paths=("/Asset/Instance/Exclude",))
    assert np.allclose(excluded_mesh.bounds, [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    assert np.allclose(geometry.get_collision_mesh().bounds, [[-1.0, -1.0, -1.0], [11.0, 1.0, 1.0]])
    assert instance.IsInstance()
    assert source_stage.GetPrimAtPath("/Asset/Instance/Exclude").IsInstanceProxy()
    assert source_path.read_bytes() == original_content
    return True


def _test_preparation_preserves_reset_transform_scaling(simulation_app, tmp_path, reset_location):
    import torch

    from pxr import Usd, UsdGeom

    from isaaclab_arena.assets.object_geometry import ObjectGeometry
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.usd.rigid_asset_preparation import prepare_rigid_object_variants

    root_path = tmp_path / "root.usda"
    nested_path = tmp_path / "nested.usda"
    _write_asset(root_path, "/Asset")
    nested_stage = _write_asset(nested_path, "/Asset/Group/Body", root_scale=3.0)
    reset_path = {"body": "/Asset/Group/Body", "ancestor": "/Asset/Group", "root": "/Asset"}[reset_location]
    UsdGeom.Xformable(nested_stage.GetPrimAtPath(reset_path)).SetResetXformStack(True)
    nested_stage.GetRootLayer().Save()
    source_contents = nested_path.read_bytes()

    spawn_cfg = _make_spawn_cfg(nested_path, scale=(2.0, 3.0, 4.0))
    source_bounds = ObjectGeometry(spawn_cfg, ObjectType.RIGID).get_bounding_box()
    with patch("isaaclab_arena.utils.usd.rigid_asset_preparation.get_arena_asset_cache_dir", return_value=tmp_path):
        prepared = prepare_rigid_object_variants([_make_spawn_cfg(root_path), spawn_cfg])
    prepared_bounds = ObjectGeometry(prepared[1], ObjectType.RIGID).get_bounding_box()
    torch.testing.assert_close(prepared_bounds.min_point, source_bounds.min_point)
    torch.testing.assert_close(prepared_bounds.max_point, source_bounds.max_point)
    expected_size = (4.0, 6.0, 8.0) if reset_location == "root" else (2.0, 2.0, 2.0)
    torch.testing.assert_close(prepared_bounds.size, torch.tensor([expected_size]))
    prepared_stage = Usd.Stage.Open(prepared[1].usd_path)
    prepared_body = UsdGeom.Xformable(prepared_stage.GetPrimAtPath("/Prepared/rigid_body"))
    assert prepared_body.GetResetXformStack() == (reset_location != "root")
    assert nested_path.read_bytes() == source_contents
    return True


def test_compatible_layouts_keep_native_scales(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_compatible_layouts_keep_native_scales, tmp_path=tmp_path)


def test_preparation_preserves_geometry_and_materials(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_preparation_preserves_geometry_and_materials, tmp_path=tmp_path
    )


def test_cache_identity_and_native_addons(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_cache_identity_and_native_addons, tmp_path=tmp_path)


def test_variants_select_bounds_and_reject_multiple_bodies(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_variants_select_bounds_and_reject_multiple_bodies, tmp_path=tmp_path
    )


def test_preparation_retains_native_fragment_targets(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_preparation_retains_native_fragment_targets, tmp_path=tmp_path
    )


def test_rigid_bounds_follow_body_pose_frame(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_rigid_bounds_follow_body_pose_frame, tmp_path=tmp_path)


def test_mesh_exclusions_inside_instances_preserve_source(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_mesh_exclusions_inside_instances_preserve_source, tmp_path=tmp_path
    )


@pytest.mark.parametrize("reset_location", ["body", "ancestor", "root"])
def test_preparation_preserves_reset_transform_scaling(tmp_path, reset_location):
    assert run_function_with_persistent_simulation_app(
        _test_preparation_preserves_reset_transform_scaling, tmp_path=tmp_path, reset_location=reset_location
    )
