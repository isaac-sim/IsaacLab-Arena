# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reject missing or collapsed image-texture coordinates before publishing assets."""

import pytest


@pytest.fixture
def textured_mesh():
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    material = UsdShade.Material.Define(stage, "/Material")
    material.CreateInput("uv_name", Sdf.ValueTypeNames.Token).Set("st")
    reader = UsdShade.Shader.Define(stage, "/Material/UVReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).ConnectToSource(material.ConnectableAPI(), "uv_name")
    reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    texture = UsdShade.Shader.Define(stage, "/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set("test.png")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    shader = UsdShade.Shader.Define(stage, "/Material/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    yield mesh


def _uvs(mesh, values, interpolation="faceVarying", indices=None):
    from pxr import Sdf, UsdGeom

    primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, interpolation)
    primvar.Set(values)
    if indices is not None:
        primvar.SetIndices(indices)


def test_missing_uvs_fail_for_image_material(textured_mesh):
    from isaaclab_arena_environments.return_to_service.asset_source.validate_exports import (
        _validate_texture_coordinates,
    )

    with pytest.raises(AssertionError, match="lacks st UVs"):
        _validate_texture_coordinates(textured_mesh)


def test_disconnected_texture_does_not_pass_with_an_unused_reader(textured_mesh):
    from pxr import UsdShade

    from isaaclab_arena_environments.return_to_service.asset_source.validate_exports import (
        _validate_texture_coordinates,
    )

    _uvs(textured_mesh, [(0, 0), (1, 0), (1, 1), (0, 1)])
    texture = UsdShade.Shader(textured_mesh.GetPrim().GetStage().GetPrimAtPath("/Material/Texture"))
    texture.GetInput("st").DisconnectSource()
    with pytest.raises(AssertionError, match="Image texture lacks UV input"):
        _validate_texture_coordinates(textured_mesh)


@pytest.mark.parametrize("degenerate_geometry", (False, True))
def test_valid_face_does_not_hide_another_faces_collapsed_uvs(textured_mesh, degenerate_geometry):
    from isaaclab_arena_environments.return_to_service.asset_source.validate_exports import (
        _validate_texture_coordinates,
    )

    textured_mesh.CreateFaceVertexCountsAttr([4, 4])
    second_face = [0, 0, 0, 0] if degenerate_geometry else [0, 1, 2, 3]
    textured_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, *second_face])
    _uvs(textured_mesh, [(0, 0), (1, 0), (1, 1), (0, 1), *([(0, 0)] * 4)])
    if degenerate_geometry:
        assert _validate_texture_coordinates(textured_mesh) == (True, 1, 0.0)
    else:
        with pytest.raises(AssertionError, match="Collapsed st UVs on face 1"):
            _validate_texture_coordinates(textured_mesh)


def test_float32_scale_slivers_are_reported(textured_mesh):
    from isaaclab_arena_environments.return_to_service.asset_source.validate_exports import (
        _validate_texture_coordinates,
    )

    points = list(textured_mesh.GetPointsAttr().Get()) + [(0, 1e-8, 0), (1, 1e-8, 0)]
    textured_mesh.CreatePointsAttr(points)
    textured_mesh.CreateFaceVertexCountsAttr([4, 4])
    textured_mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 0, 1, 5, 4])
    _uvs(textured_mesh, [(0, 0), (1, 0), (1, 1), (0, 1), *([(0, 0)] * 4)])
    textured, excluded_count, excluded_area = _validate_texture_coordinates(textured_mesh)
    assert textured and excluded_count == 1
    assert excluded_area == pytest.approx(1e-8)


def test_surface_tolerance_scales_with_squared_mesh_size():
    from isaaclab_arena_environments.return_to_service.asset_source.uv import surface_area_tolerance

    unit = surface_area_tolerance(((0, 0, 0), (1, 2, 3)))
    assert surface_area_tolerance(((0, 0, 0), (10, 20, 30))) == pytest.approx(100 * unit)


@pytest.mark.parametrize(
    "values, message",
    [
        ([(0, 0)] * 4, "Collapsed st UVs"),
        ([(0, 0), (0.25, 0.25), (0.5, 0.5), (1, 1)], "Collapsed st UVs"),
        ([(0, 0), (1, 0), (1, float("nan")), (0, 1)], "Invalid st UV coordinate"),
        ([(0, 0), (1, 0), (1, 1)], "UV corner count mismatch"),
    ],
)
def test_invalid_uvs_fail_before_export_publication(textured_mesh, values, message):
    from isaaclab_arena_environments.return_to_service.asset_source.validate_exports import (
        _validate_texture_coordinates,
    )

    _uvs(textured_mesh, values)
    with pytest.raises(AssertionError, match=message):
        _validate_texture_coordinates(textured_mesh)


@pytest.mark.parametrize("interpolation", ("faceVarying", "vertex", "varying"))
def test_indexed_uvs_and_material_interface_reader_are_supported(textured_mesh, interpolation):
    from isaaclab_arena_environments.return_to_service.asset_source.validate_exports import (
        _validate_texture_coordinates,
    )

    _uvs(textured_mesh, [(1, 1), (0, 1), (0, 0), (1, 0)], interpolation, [2, 3, 0, 1])
    assert _validate_texture_coordinates(textured_mesh) == (True, 0, 0.0)
