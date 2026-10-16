# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for USD scale handling in extract_trimesh_from_usd and compute_local_bounding_box_from_usd.

Verifies that spawn-scale is applied in the local frame (R·(S·v)+t) rather than world
frame (S·(R·v+t)), which matters for translated/rotated child prims under non-uniform scale.
"""

import numpy as np
import tempfile

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

HEADLESS = True


def _test_extract_trimesh_translated_child_nonuniform_scale(simulation_app):
    """extract_trimesh_from_usd must scale in local frame, not world frame.

    Setup: unit cube under a child Xform translated +1.0 in X, scale=(2,1,1).
    Correct (local scale): verts ±0.5 → ±1.0 in local X, then translate +1 → world X [0.0, 2.0].
    Bug (world scale): verts ±0.5, translate +1 → world [0.5, 1.5], then *2 → [1.0, 3.0].
    """
    import tempfile

    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)

    child_xform = UsdGeom.Xform.Define(stage, "/root/child")
    child_xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))

    mesh_prim = UsdGeom.Mesh.Define(stage, "/root/child/cube")
    points = [
        Gf.Vec3f(-0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, 0.5, 0.5),
        Gf.Vec3f(-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    face_vertex_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        5,
        4,
        2,
        3,
        7,
        6,
        0,
        3,
        7,
        4,
        1,
        2,
        6,
        5,
    ]
    mesh_prim.GetPointsAttr().Set(points)
    mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        usd_path = f.name
    stage.Export(usd_path)

    scale = (2.0, 1.0, 1.0)
    tri = extract_trimesh_from_usd(usd_path, scale=scale)
    verts = tri.vertices

    # Local scale: ±0.5*2=±1.0 then +1 translate → world X [0.0, 2.0]
    assert np.isclose(verts[:, 0].min(), 0.0, atol=1e-5), f"got {verts[:, 0].min():.4f}"
    assert np.isclose(verts[:, 0].max(), 2.0, atol=1e-5), f"got {verts[:, 0].max():.4f}"

    assert np.isclose(verts[:, 1].min(), -0.5, atol=1e-5)
    assert np.isclose(verts[:, 1].max(), 0.5, atol=1e-5)
    assert np.isclose(verts[:, 2].min(), -0.5, atol=1e-5)
    assert np.isclose(verts[:, 2].max(), 0.5, atol=1e-5)

    return True


def _test_extract_trimesh_from_prim_scales_in_root_frame(simulation_app):
    """extract_trimesh_from_prim applies parent scale in the referenced prim frame."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim

    stage = Usd.Stage.CreateInMemory()
    root_xform = UsdGeom.Xform.Define(stage, "/root")
    root_xform.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    root_xform.AddRotateZOp().Set(90.0)
    root = root_xform.GetPrim()
    stage.SetDefaultPrim(root)

    child_xform = UsdGeom.Xform.Define(stage, "/root/child")
    child_xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))

    mesh_prim = UsdGeom.Mesh.Define(stage, "/root/child/cube")
    points = [
        Gf.Vec3f(-0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, 0.5, 0.5),
        Gf.Vec3f(-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    face_vertex_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        5,
        4,
        2,
        3,
        7,
        6,
        0,
        3,
        7,
        4,
        1,
        2,
        6,
        5,
    ]
    mesh_prim.GetPointsAttr().Set(points)
    mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    tri = extract_trimesh_from_prim(stage, "/root", scale=(2.0, 1.0, 1.0))
    verts = tri.vertices

    # Root-frame scale: child center x=1 scales to x=2, so cube x bounds become [1, 3].
    assert np.isclose(verts[:, 0].min(), 1.0, atol=1e-5), f"got {verts[:, 0].min():.4f}"
    assert np.isclose(verts[:, 0].max(), 3.0, atol=1e-5), f"got {verts[:, 0].max():.4f}"
    assert np.isclose(verts[:, 1].min(), -0.5, atol=1e-5)
    assert np.isclose(verts[:, 1].max(), 0.5, atol=1e-5)

    return True


def _test_extract_trimesh_nested_transforms_keep_caller_frames(simulation_app):
    """File extraction retains root transforms; prim extraction removes the complete root transform."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim, extract_trimesh_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/root")
    stage.SetDefaultPrim(root.GetPrim())
    root.AddTranslateOp().Set(Gf.Vec3d(10.0, -4.0, 7.0))
    root.AddRotateZOp().Set(90.0)
    root.AddScaleOp().Set(Gf.Vec3d(2.0, 3.0, 4.0))
    child = UsdGeom.Xform.Define(stage, "/root/child")
    child.AddTranslateOp().Set(Gf.Vec3d(1.0, 2.0, 3.0))
    child.AddRotateZOp().Set(90.0)
    child.AddScaleOp().Set(Gf.Vec3d(0.5, 2.0, 1.0))
    mesh = UsdGeom.Mesh.Define(stage, "/root/child/mesh")
    mesh.AddTranslateOp().Set(Gf.Vec3d(2.0, -1.0, 0.5))
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    mesh.GetPointsAttr().Set(points.tolist())
    mesh.GetFaceVertexCountsAttr().Set([4])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    configured_scale = (3.0, 2.0, 0.5)

    with tempfile.TemporaryDirectory() as temporary_directory:
        usd_path = f"{temporary_directory}/nested_transforms.usda"
        stage.Export(usd_path)
        world_mesh = extract_trimesh_from_usd(usd_path, scale=configured_scale)
    root_mesh = extract_trimesh_from_prim(stage, "/root", scale=configured_scale)

    # File extraction scales each mesh's points before applying any authored transforms.
    expected_world_vertices = np.column_stack((
        1.0 - 4.5 * points[:, 0],
        2.0 - 8.0 * points[:, 1],
        21.0 + 2.0 * points[:, 2],
    ))
    # Prim extraction removes the authored root scale, then scales child offsets in the root frame.
    expected_root_vertices = np.column_stack((
        9.0 - 6.0 * points[:, 1],
        6.0 + points[:, 0],
        1.75 + 0.5 * points[:, 2],
    ))
    for extracted_mesh, expected_vertices in (
        (world_mesh, expected_world_vertices),
        (root_mesh, expected_root_vertices),
    ):
        # Trimesh may reorder vertices; round rotation noise before sorting coordinate triples.
        actual_vertices = sorted(np.round(extracted_mesh.vertices, decimals=6).tolist())
        np.testing.assert_allclose(actual_vertices, sorted(expected_vertices.tolist()), atol=1e-6)
    return True


def _test_extract_trimesh_combines_polygon_meshes(simulation_app):
    """Triangles from each polygon must reference that polygon's transformed vertices."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim, extract_trimesh_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)
    first_mesh = UsdGeom.Mesh.Define(stage, "/root/first_mesh")
    first_mesh.GetPointsAttr().Set([(0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)])
    first_mesh.GetFaceVertexCountsAttr().Set([4])
    first_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
    second_mesh = UsdGeom.Mesh.Define(stage, "/root/second_mesh")
    second_mesh.AddTranslateOp().Set(Gf.Vec3d(10.0, 0.0, 0.0))
    second_mesh.GetPointsAttr().Set([
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 2),
        (2, 0, 2),
        (2, 1, 2),
        (0, 1, 2),
    ])
    second_mesh.GetFaceVertexCountsAttr().Set([3, 4])
    second_mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3, 4, 5, 6])
    incomplete_mesh = UsdGeom.Mesh.Define(stage, "/root/incomplete_mesh")
    incomplete_mesh.GetPointsAttr().Set([(100, 0, 0), (101, 0, 0), (100, 1, 0)])

    with tempfile.TemporaryDirectory() as temporary_directory:
        usd_path = f"{temporary_directory}/polygon_meshes.usda"
        stage.Export(usd_path)
        world_mesh = extract_trimesh_from_usd(usd_path)
    root_mesh = extract_trimesh_from_prim(stage, "/root")

    expected_triangles = np.array([
        [(0, 0, 0), (2, 0, 0), (2, 1, 0)],
        [(0, 0, 0), (2, 1, 0), (0, 1, 0)],
        [(10, 0, 0), (11, 0, 0), (10, 1, 0)],
        [(10, 0, 2), (12, 0, 2), (12, 1, 2)],
        [(10, 0, 2), (12, 1, 2), (10, 1, 2)],
    ])
    for extracted_mesh in (world_mesh, root_mesh):
        assert len(extracted_mesh.vertices) == 11
        np.testing.assert_allclose(
            sorted(extracted_mesh.triangles.reshape(-1, 9).tolist()),
            sorted(expected_triangles.reshape(-1, 9).tolist()),
        )
    return True


def _test_extract_trimesh_from_prim_keeps_mesh_with_unsupported_geometry(simulation_app):
    """extract_trimesh_from_prim keeps extracted meshes when analytic geometry is present."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_prim

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)

    mesh_prim = UsdGeom.Mesh.Define(stage, "/root/mesh")
    mesh_prim.GetPointsAttr().Set([
        Gf.Vec3f(-0.5, -0.5, 0.0),
        Gf.Vec3f(0.5, -0.5, 0.0),
        Gf.Vec3f(0.0, 0.5, 0.0),
    ])
    mesh_prim.GetFaceVertexCountsAttr().Set([3])
    mesh_prim.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    UsdGeom.Cube.Define(stage, "/root/analytic_cube")

    tri = extract_trimesh_from_prim(stage, "/root")
    assert len(tri.vertices) == 3

    return True


def _test_extract_trimesh_from_prim_rejects_analytic_only_geometry(simulation_app):
    """extract_trimesh_from_prim rejects geometry with no mesh subset."""
    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import UnsupportedCollisionGeometryError, extract_trimesh_from_prim

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.Cube.Define(stage, "/root/analytic_cube")

    with pytest.raises(UnsupportedCollisionGeometryError):
        extract_trimesh_from_prim(stage, "/root")

    return True


def _test_extract_trimesh_from_usd_keeps_mesh_with_unsupported_geometry(simulation_app):
    """extract_trimesh_from_usd keeps extracted meshes when analytic geometry is present."""
    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import extract_trimesh_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)

    mesh_prim = UsdGeom.Mesh.Define(stage, "/root/mesh")
    mesh_prim.GetPointsAttr().Set([
        Gf.Vec3f(-0.5, -0.5, 0.0),
        Gf.Vec3f(0.5, -0.5, 0.0),
        Gf.Vec3f(0.0, 0.5, 0.0),
    ])
    mesh_prim.GetFaceVertexCountsAttr().Set([3])
    mesh_prim.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    UsdGeom.Cube.Define(stage, "/root/analytic_cube")

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        usd_path = f.name
    stage.Export(usd_path)

    tri = extract_trimesh_from_usd(usd_path)
    assert len(tri.vertices) == 3

    return True


def _test_extract_trimesh_from_usd_rejects_analytic_only_geometry(simulation_app):
    """extract_trimesh_from_usd rejects geometry with no mesh subset."""
    import tempfile

    from pxr import Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import UnsupportedCollisionGeometryError, extract_trimesh_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)
    UsdGeom.Cube.Define(stage, "/root/analytic_cube")

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        usd_path = f.name
    stage.Export(usd_path)

    with pytest.raises(UnsupportedCollisionGeometryError):
        extract_trimesh_from_usd(usd_path)

    return True


def _test_bbox_translated_child_nonuniform_scale(simulation_app):
    """AABB is conservative vs mesh path for translated children under non-uniform scale."""
    import tempfile

    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)

    child_xform = UsdGeom.Xform.Define(stage, "/root/child")
    child_xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, 0.0))

    mesh_prim = UsdGeom.Mesh.Define(stage, "/root/child/cube")
    points = [
        Gf.Vec3f(-0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, 0.5, 0.5),
        Gf.Vec3f(-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    face_vertex_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        5,
        4,
        2,
        3,
        7,
        6,
        0,
        3,
        7,
        4,
        1,
        2,
        6,
        5,
    ]
    mesh_prim.GetPointsAttr().Set(points)
    mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        usd_path = f.name
    stage.Export(usd_path)

    scale = (2.0, 1.0, 1.0)
    bbox = compute_local_bounding_box_from_usd(usd_path, scale=scale)

    # ComputeLocalBound gives [0.5,1.5] * scale_x=2 → [1.0, 3.0]
    min_pt = bbox.min_point[0]  # (3,) tensor
    max_pt = bbox.max_point[0]  # (3,) tensor
    assert np.isclose(min_pt[0].item(), 1.0, atol=1e-5), f"got {min_pt[0].item():.4f}"
    assert np.isclose(max_pt[0].item(), 3.0, atol=1e-5), f"got {max_pt[0].item():.4f}"
    assert np.isclose(min_pt[1].item(), -0.5, atol=1e-5)
    assert np.isclose(max_pt[1].item(), 0.5, atol=1e-5)
    assert np.isclose(min_pt[2].item(), -0.5, atol=1e-5)
    assert np.isclose(max_pt[2].item(), 0.5, atol=1e-5)

    return True


def _test_both_paths_agree_origin_prim(simulation_app):
    import tempfile

    from pxr import Gf, Usd, UsdGeom

    from isaaclab_arena.utils.usd.helpers import compute_local_bounding_box_from_usd, extract_trimesh_from_usd

    stage = Usd.Stage.CreateInMemory()
    root = stage.DefinePrim("/root", "Xform")
    stage.SetDefaultPrim(root)

    mesh_prim = UsdGeom.Mesh.Define(stage, "/root/cube")
    points = [
        Gf.Vec3f(-0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, -0.5, -0.5),
        Gf.Vec3f(0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, 0.5, -0.5),
        Gf.Vec3f(-0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, -0.5, 0.5),
        Gf.Vec3f(0.5, 0.5, 0.5),
        Gf.Vec3f(-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    face_vertex_indices = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        5,
        4,
        2,
        3,
        7,
        6,
        0,
        3,
        7,
        4,
        1,
        2,
        6,
        5,
    ]
    mesh_prim.GetPointsAttr().Set(points)
    mesh_prim.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh_prim.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    with tempfile.NamedTemporaryFile(suffix=".usda", delete=False) as f:
        usd_path = f.name
    stage.Export(usd_path)

    scale = (2.0, 3.0, 0.5)
    tri = extract_trimesh_from_usd(usd_path, scale=scale)
    bbox = compute_local_bounding_box_from_usd(usd_path, scale=scale)

    verts = tri.vertices
    # ±0.5 * (2, 3, 0.5) = (±1.0, ±1.5, ±0.25)
    assert np.isclose(verts[:, 0].min(), -1.0, atol=1e-5)
    assert np.isclose(verts[:, 0].max(), 1.0, atol=1e-5)
    assert np.isclose(verts[:, 1].min(), -1.5, atol=1e-5)
    assert np.isclose(verts[:, 1].max(), 1.5, atol=1e-5)
    assert np.isclose(verts[:, 2].min(), -0.25, atol=1e-5)
    assert np.isclose(verts[:, 2].max(), 0.25, atol=1e-5)

    # BBox must match mesh extents exactly for origin-centered single prim.
    min_pt = bbox.min_point[0]  # (3,) tensor
    max_pt = bbox.max_point[0]  # (3,) tensor
    assert np.isclose(min_pt[0].item(), -1.0, atol=1e-5)
    assert np.isclose(max_pt[0].item(), 1.0, atol=1e-5)
    assert np.isclose(min_pt[1].item(), -1.5, atol=1e-5)
    assert np.isclose(max_pt[1].item(), 1.5, atol=1e-5)
    assert np.isclose(min_pt[2].item(), -0.25, atol=1e-5)
    assert np.isclose(max_pt[2].item(), 0.25, atol=1e-5)

    return True


def test_extract_trimesh_translated_child_nonuniform_scale():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_translated_child_nonuniform_scale, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_from_prim_scales_in_root_frame():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_from_prim_scales_in_root_frame, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_nested_transforms_keep_caller_frames():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_nested_transforms_keep_caller_frames, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_combines_polygon_meshes():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_combines_polygon_meshes, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_from_prim_keeps_mesh_with_unsupported_geometry():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_from_prim_keeps_mesh_with_unsupported_geometry, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_from_prim_rejects_analytic_only_geometry():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_from_prim_rejects_analytic_only_geometry, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_from_usd_keeps_mesh_with_unsupported_geometry():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_from_usd_keeps_mesh_with_unsupported_geometry, headless=HEADLESS
    )
    assert result


def test_extract_trimesh_from_usd_rejects_analytic_only_geometry():
    result = run_function_with_persistent_simulation_app(
        _test_extract_trimesh_from_usd_rejects_analytic_only_geometry, headless=HEADLESS
    )
    assert result


def test_bbox_translated_child_nonuniform_scale():
    result = run_function_with_persistent_simulation_app(
        _test_bbox_translated_child_nonuniform_scale, headless=HEADLESS
    )
    assert result


def test_both_paths_agree_origin_prim():
    result = run_function_with_persistent_simulation_app(_test_both_paths_agree_origin_prim, headless=HEADLESS)
    assert result
