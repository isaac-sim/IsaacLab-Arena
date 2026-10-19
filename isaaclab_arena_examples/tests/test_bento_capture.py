# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check that face materials survive triangulation and shared mesh vertices."""


def _test_face_material_boundaries(simulation_app):
    import numpy as np

    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from isaaclab_arena_examples.bento.capture import _mesh, _vertex_colors

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4, 4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 1, 4, 5, 2])
    mesh.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set([(0.2, 0.2, 0.2)] * 6)
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    for face_id, color in enumerate(((0.8, 0.8, 0.8), (0, 0.337163, 0.723055))):
        material = UsdShade.Material.Define(stage, f"/Material{face_id}")
        shader = UsdShade.Shader.Define(stage, f"/Material{face_id}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        subset = binding.CreateMaterialBindSubset(f"Region{face_id}", [face_id])
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)

    triangles = _mesh(mesh.GetPrim())
    original_positions = triangles.vertices[triangles.faces].copy()
    colors = _vertex_colors(mesh.GetPrim(), triangles, {})
    # Quad triangulation must retain both colors at vertices shared by the two regions.
    np.testing.assert_allclose(colors[triangles.faces[:2]], np.tile((0.8, 0.8, 0.8), (2, 3, 1)))
    np.testing.assert_allclose(colors[triangles.faces[2:]], np.tile((0, 0.337163, 0.723055), (2, 3, 1)))
    np.testing.assert_allclose(triangles.vertices[triangles.faces], original_positions)
    assert len(triangles.vertices) == 12
    assert np.isfinite(triangles.vertex_normals).all()
    return True


def test_face_material_boundaries():
    from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

    assert run_function_with_persistent_simulation_app(_test_face_material_boundaries)
