# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify spawned mesh extraction uses live scene variants and pose frames."""

import numpy as np
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _define_polygon_mesh(stage, prim_path, points, face_vertex_counts=None, face_vertex_indices=None):
    """Author mesh topology without triangulating its polygons."""
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(points)
    mesh.GetFaceVertexCountsAttr().Set([len(points)] if face_vertex_counts is None else face_vertex_counts)
    polygon_indices = list(range(len(points))) if face_vertex_indices is None else face_vertex_indices
    mesh.GetFaceVertexIndicesAttr().Set(polygon_indices)
    return mesh


def _make_scene(stage, num_envs=1, entity_kind="extras", device="cpu"):
    """Provide the InteractiveScene fields used by spawned mesh extraction."""
    scene = SimpleNamespace(
        stage=stage,
        cfg=SimpleNamespace(asset=SimpleNamespace(prim_path="{ENV_REGEX_NS}/Asset")),
        env_regex_ns="/World/envs/env_.*",
        clone_plan=object(),
        num_envs=num_envs,
        device=device,
        rigid_objects={},
        articulations={},
        deformable_objects={},
        extras={},
    )
    getattr(scene, entity_kind)["asset"] = object()
    return scene


def _read_mesh_groups(scene_access, scene, source_groups):
    """Resolve representative source prims through the clone-plan query."""
    clone_sources = []
    for representative_prim, environment_ids in source_groups:
        clone_sources.append((None, None, str(representative_prim.GetPath()), environment_ids))
    with patch.object(scene_access.cloner.query, "iter_sources", return_value=iter(clone_sources)):
        return scene_access.read_spawned_meshes_in_local_frame(scene, "asset")


def _check_nested_transforms_and_polygon_topology(scene_access):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    wrapper = UsdGeom.Xform.Define(stage, "/World/Wrapper")
    translation = wrapper.AddTranslateOp()
    translation.Set(Gf.Vec3d(100.0, -50.0, 3.0))
    rotation = wrapper.AddRotateZOp()
    rotation.Set(37.0)
    wrapper.AddScaleOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
    reference = UsdGeom.Xform.Define(stage, "/World/Wrapper/Reference")
    reference.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))
    reference.AddScaleOp().Set(Gf.Vec3d(0.5, 2.0, 1.0))

    points = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 1.0], [-0.5, 0.5, 1.0]])
    mesh = _define_polygon_mesh(stage, "/World/Wrapper/Reference/Mesh", points.tolist())
    mesh.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    mesh.AddRotateZOp().Set(90.0)
    mesh.AddScaleOp().Set(Gf.Vec3d(2.0, 0.5, 1.0))
    triangle_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    triangle = _define_polygon_mesh(stage, "/World/Wrapper/Reference/Triangle", triangle_points.tolist())
    triangle.AddTranslateOp().Set(Gf.Vec3d(-2.0, 0.0, 0.0))
    ignored_cube = UsdGeom.Cube.Define(stage, "/World/Wrapper/Reference/RenderOnly")
    ignored_cube.GetPurposeAttr().Set(UsdGeom.Tokens.render)

    scene = _make_scene(stage)
    source_groups = [(reference.GetPrim(), (0,))]
    ((geometry, _environment_indices),) = _read_mesh_groups(scene_access, scene, source_groups)
    expected_quad = np.column_stack((
        1.0 - 0.5 * points[:, 1],
        6.0 * (2.0 + 2.0 * points[:, 0]),
        4.0 * (3.0 + points[:, 2]),
    ))
    expected_triangle = (triangle_points + [-2.0, 0.0, 0.0]) * [1.0, 6.0, 4.0]
    np.testing.assert_allclose(geometry.vertices, np.concatenate((expected_quad, expected_triangle)), atol=1e-6)
    np.testing.assert_array_equal(geometry.face_vertex_counts, [4, 3])
    np.testing.assert_array_equal(geometry.face_vertex_indices, [0, 1, 2, 3, 4, 5, 6])
    assert geometry.vertices.dtype == np.float64

    translation.Set(Gf.Vec3d(-20.0, 5.0, 8.0))
    rotation.Set(-23.0)
    ((moved_geometry, _environment_indices),) = _read_mesh_groups(scene_access, scene, source_groups)
    np.testing.assert_allclose(moved_geometry.vertices, geometry.vertices, atol=1e-6)


def _check_rigid_body_frame_inside_wrapper(scene_access):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    wrapper = UsdGeom.Xform.Define(stage, "/World/Wrapper")
    wrapper.AddTranslateOp().Set(Gf.Vec3d(10.0, -4.0, 7.0))
    wrapper.AddRotateZOp().Set(45.0)
    wrapper.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 2.0))
    body = UsdGeom.Xform.Define(stage, "/World/Wrapper/Body")
    body.AddTranslateOp().Set(Gf.Vec3d(3.0, 0.0, 0.0))
    body.AddRotateZOp().Set(90.0)
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    mesh = _define_polygon_mesh(stage, "/World/Wrapper/Body/Mesh", points.tolist())
    mesh.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))

    scene = _make_scene(stage, entity_kind="rigid_objects")
    ((geometry, _environment_indices),) = _read_mesh_groups(scene_access, scene, [(wrapper.GetPrim(), (0,))])
    np.testing.assert_allclose(geometry.vertices, (points + [1.0, 0.0, 0.0]) * 2.0, atol=1e-6)


def _check_inherited_geometry_purpose(scene_access):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Asset")
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    _define_polygon_mesh(stage, "/World/Asset/DefaultMesh", points)
    render_parent = UsdGeom.Xform.Define(stage, "/World/Asset/RenderGeometry")
    render_parent.GetPurposeAttr().Set(UsdGeom.Tokens.render)
    ignored_mesh = _define_polygon_mesh(stage, "/World/Asset/RenderGeometry/Mesh", (np.asarray(points) * 10).tolist())
    assert not ignored_mesh.GetPurposeAttr().HasAuthoredValueOpinion()
    assert ignored_mesh.ComputePurpose() == UsdGeom.Tokens.render

    ((geometry, _environment_indices),) = _read_mesh_groups(scene_access, _make_scene(stage), [(root.GetPrim(), (0,))])
    np.testing.assert_array_equal(geometry.vertices, points)
    np.testing.assert_array_equal(geometry.face_vertex_counts, [3])
    np.testing.assert_array_equal(geometry.face_vertex_indices, [0, 1, 2])


def _check_heterogeneous_clone_groups(scene_access):
    import torch

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    first_variant = UsdGeom.Xform.Define(stage, "/World/FirstVariant")
    first_variant.AddScaleOp().Set(Gf.Vec3d(2.0, 1.0, 1.0))
    second_variant = UsdGeom.Xform.Define(stage, "/World/SecondVariant")
    first_points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    second_points = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
    _define_polygon_mesh(stage, "/World/FirstVariant/Mesh", first_points)
    _define_polygon_mesh(stage, "/World/SecondVariant/Mesh", second_points)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scene = _make_scene(stage, num_envs=4, device=device)
    source_groups = [(first_variant.GetPrim(), (3, 1)), (second_variant.GetPrim(), (0, 2))]

    first_group, second_group = _read_mesh_groups(scene_access, scene, source_groups)
    first_geometry, first_environment_indices = first_group
    second_geometry, second_environment_indices = second_group
    np.testing.assert_allclose(first_geometry.vertices, np.asarray(first_points) * [2.0, 1.0, 1.0])
    np.testing.assert_allclose(second_geometry.vertices, second_points)
    assert first_environment_indices.tolist() == [3, 1]
    assert second_environment_indices.tolist() == [0, 2]
    for _geometry, environment_indices in (first_group, second_group):
        assert environment_indices.dtype == torch.long
        assert environment_indices.device == torch.device(device)

    invalid_source_groups = [
        [(first_variant.GetPrim(), (0, 1)), (second_variant.GetPrim(), (1, 2, 3))],
        [(first_variant.GetPrim(), (0, 1)), (second_variant.GetPrim(), (2,))],
    ]
    for source_groups in invalid_source_groups:
        with pytest.raises(AssertionError, match="cover every environment exactly once"):
            _read_mesh_groups(scene_access, scene, source_groups)
    for invalid_id in (-1, 4):
        with pytest.raises(AssertionError, match="outside"):
            _read_mesh_groups(scene_access, scene, [(first_variant.GetPrim(), (invalid_id,))])
    with pytest.raises(AssertionError, match="at least one environment"):
        _read_mesh_groups(scene_access, scene, [(first_variant.GetPrim(), ())])


def _check_instance_proxy_geometry(scene_access):
    from pxr import Gf, Usd, UsdGeom

    source_stage = Usd.Stage.CreateInMemory()
    source_root = UsdGeom.Xform.Define(source_stage, "/Source")
    source_stage.SetDefaultPrim(source_root.GetPrim())
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    _define_polygon_mesh(source_stage, "/Source/Mesh", points)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Asset")
    instance = UsdGeom.Xform.Define(stage, "/World/Asset/Instance")
    instance.GetPrim().GetReferences().AddReference(source_stage.GetRootLayer().identifier)
    instance.GetPrim().SetInstanceable(True)
    instance.AddTranslateOp().Set(Gf.Vec3d(2.0, 0.0, 0.0))
    assert stage.GetPrimAtPath("/World/Asset/Instance/Mesh").IsInstanceProxy()

    ((geometry, _environment_indices),) = _read_mesh_groups(scene_access, _make_scene(stage), [(root.GetPrim(), (0,))])
    np.testing.assert_allclose(geometry.vertices, np.asarray(points) + [2.0, 0.0, 0.0])


def _check_point_instanced_geometry(scene_access):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Asset")
    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    face_vertex_counts = [3, 3, 3, 3]
    face_vertex_indices = [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    _define_polygon_mesh(stage, "/World/Asset/DirectMesh", points, face_vertex_counts, face_vertex_indices)
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/Asset/Instances")
    prototype_mesh = _define_polygon_mesh(
        stage, "/World/Asset/Instances/Prototype", points, face_vertex_counts, face_vertex_indices
    )
    instancer.GetPrototypesRel().SetTargets([prototype_mesh.GetPath()])
    instancer.GetProtoIndicesAttr().Set([0, 0])
    instancer.GetPositionsAttr().Set([(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)])
    scene = _make_scene(stage)
    source_groups = [(root.GetPrim(), (0,))]

    # Traversing prototype meshes alone would silently lose both translated instances.
    with pytest.raises(AssertionError, match="point instancers.*requires per-instance transforms"):
        _read_mesh_groups(scene_access, scene, source_groups)

    for ignored_purpose in (UsdGeom.Tokens.render, UsdGeom.Tokens.proxy, UsdGeom.Tokens.guide):
        instancer.GetPurposeAttr().Set(ignored_purpose)
        ((geometry, _environment_indices),) = _read_mesh_groups(scene_access, scene, source_groups)
        np.testing.assert_array_equal(geometry.vertices, points)
        np.testing.assert_array_equal(geometry.face_vertex_counts, face_vertex_counts)
        np.testing.assert_array_equal(geometry.face_vertex_indices, face_vertex_indices)


def _check_unsupported_mesh_geometry(scene_access):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/World/Asset")
    source_groups = [(root.GetPrim(), (0,))]
    scene = _make_scene(stage)
    with pytest.raises(AssertionError, match="no default-purpose mesh geometry"):
        _read_mesh_groups(scene_access, scene, source_groups)

    points = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    mesh = _define_polygon_mesh(stage, "/World/Asset/Mesh", points)
    UsdGeom.Cube.Define(stage, "/World/Asset/Cube")
    with pytest.raises(AssertionError, match="unsupported geometry type 'Cube'"):
        _read_mesh_groups(scene_access, scene, source_groups)
    stage.RemovePrim("/World/Asset/Cube")

    for entity_kind in ("articulations", "deformable_objects"):
        unsupported_scene = _make_scene(stage, entity_kind=entity_kind)
        with pytest.raises(AssertionError, match="does not support articulations or deformables"):
            _read_mesh_groups(scene_access, unsupported_scene, source_groups)
    unregistered_scene = _make_scene(stage)
    unregistered_scene.extras.clear()
    with pytest.raises(AssertionError, match="registered in neither"):
        _read_mesh_groups(scene_access, unregistered_scene, source_groups)

    moving_body = UsdGeom.Xform.Define(stage, "/World/Asset/MovingBody")
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(moving_body.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr(True)
    _define_polygon_mesh(stage, "/World/Asset/MovingBody/Mesh", points)
    with pytest.raises(AssertionError, match="contains moving rigid body"):
        _read_mesh_groups(scene_access, scene, source_groups)
    rigid_body.GetRigidBodyEnabledAttr().Set(False)
    ((geometry, _environment_indices),) = _read_mesh_groups(scene_access, scene, source_groups)
    assert len(geometry.vertices) == 6
    stage.RemovePrim("/World/Asset/MovingBody")

    articulation_root = UsdGeom.Xform.Define(stage, "/World/Asset/Articulation")
    UsdPhysics.ArticulationRootAPI.Apply(articulation_root.GetPrim())
    with pytest.raises(AssertionError, match="contains articulation"):
        _read_mesh_groups(scene_access, scene, source_groups)
    stage.RemovePrim("/World/Asset/Articulation")

    mesh.GetPointsAttr().Set(points, Usd.TimeCode(0.0))
    mesh.GetPointsAttr().Set((np.asarray(points) + 1.0).tolist(), Usd.TimeCode(1.0))
    with pytest.raises(AssertionError, match="fixed vertices and topology"):
        _read_mesh_groups(scene_access, scene, source_groups)
    mesh.GetPointsAttr().Clear()
    mesh.GetPointsAttr().Set(points)
    translation = mesh.AddTranslateOp()
    translation.Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0.0))
    translation.Set(Gf.Vec3d(1.0, 0.0, 0.0), Usd.TimeCode(1.0))
    with pytest.raises(AssertionError, match="fixed descendant transforms"):
        _read_mesh_groups(scene_access, scene, source_groups)
    translation.GetAttr().Clear()
    translation.Set(Gf.Vec3d(0.0, 0.0, 0.0))
    mesh.SetResetXformStack(True)
    with pytest.raises(AssertionError, match="inherit the root pose"):
        _read_mesh_groups(scene_access, scene, source_groups)


def _test_arena_world_mesh_scene_access(_simulation_app):
    import isaaclab_arena.environments.arena_world_scene_access as scene_access

    _check_nested_transforms_and_polygon_topology(scene_access)
    _check_rigid_body_frame_inside_wrapper(scene_access)
    _check_inherited_geometry_purpose(scene_access)
    _check_heterogeneous_clone_groups(scene_access)
    _check_instance_proxy_geometry(scene_access)
    _check_point_instanced_geometry(scene_access)
    _check_unsupported_mesh_geometry(scene_access)
    return True


def test_arena_world_mesh_scene_access():
    assert run_function_with_persistent_simulation_app(_test_arena_world_mesh_scene_access)
