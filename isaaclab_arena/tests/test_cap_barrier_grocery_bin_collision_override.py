# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

import isaaclab_arena.integrations.cap_barrier.grocery_bin_collision_override as bin_override_module
from isaaclab_arena.integrations.cap_barrier.grocery_bin_collision_override import (
    _BIN_DEFAULT_PRIM_PATH,
    _BIN_MATERIAL_SUBPATH,
    _BIN_PROXY_BOX_SPECS,
    _BIN_PROXY_CONTACT_OFFSET_M,
    _BIN_PROXY_REST_OFFSET_M,
    _BIN_ROOT_SCALE,
    _BIN_SOURCE_COLLISION_SUBPATH,
    _BIN_SOURCE_LOWER_RAW,
    _BIN_SOURCE_UPPER_RAW,
    _CERTIFIED_MIN_TRIANGLE_AABB_GAP_M,
    _CERTIFIED_MIN_TRIANGLE_AABB_GAP_RAW,
    _EXPECTED_BIN_SOURCE_URI_SUFFIX,
    GroceryBinCollisionOverride,
    _require_proxy_cavity_partition,
    _require_source_cavity_certificate,
    apply_grocery_bin_collision_override,
    validate_live_grocery_bin_collision_contract,
)


def _source_path(tmp_path: Path) -> Path:
    return tmp_path / _EXPECTED_BIN_SOURCE_URI_SUFFIX.lstrip("/")


def _valid_mesh_geometry(
    *,
    right_inner_x: float = 19.70049476623535,
) -> tuple[list[Gf.Vec3f], list[int], list[int]]:
    lower_x, lower_y, lower_z = _BIN_SOURCE_LOWER_RAW
    upper_x, upper_y, upper_z = _BIN_SOURCE_UPPER_RAW
    left_inner_x = -19.70049476623535
    negative_inner_y = -14.20049476623535
    positive_inner_y = 14.20049476623535
    points = [
        Gf.Vec3f(lower_x, lower_y, lower_z),
        Gf.Vec3f(upper_x, lower_y, lower_z),
        Gf.Vec3f(upper_x, upper_y, lower_z),
        Gf.Vec3f(lower_x, upper_y, lower_z),
        Gf.Vec3f(right_inner_x, lower_y, lower_z),
        Gf.Vec3f(right_inner_x, upper_y, lower_z),
        Gf.Vec3f(right_inner_x, upper_y, upper_z),
        Gf.Vec3f(right_inner_x, lower_y, upper_z),
        Gf.Vec3f(left_inner_x, lower_y, lower_z),
        Gf.Vec3f(left_inner_x, lower_y, upper_z),
        Gf.Vec3f(left_inner_x, upper_y, upper_z),
        Gf.Vec3f(left_inner_x, upper_y, lower_z),
        Gf.Vec3f(lower_x, positive_inner_y, lower_z),
        Gf.Vec3f(upper_x, positive_inner_y, lower_z),
        Gf.Vec3f(upper_x, positive_inner_y, upper_z),
        Gf.Vec3f(lower_x, positive_inner_y, upper_z),
        Gf.Vec3f(lower_x, negative_inner_y, lower_z),
        Gf.Vec3f(lower_x, negative_inner_y, upper_z),
        Gf.Vec3f(upper_x, negative_inner_y, upper_z),
        Gf.Vec3f(upper_x, negative_inner_y, lower_z),
    ]
    face_counts = [4, 4, 4, 4, 4]
    face_indices = list(range(len(points)))
    return points, face_counts, face_indices


def _write_source(
    path: Path,
    *,
    default_prim_path: str = _BIN_DEFAULT_PRIM_PATH,
    root_scale: tuple[float, float, float] = _BIN_ROOT_SCALE,
    rigid_api: bool = True,
    rigid_enabled: bool = True,
    kinematic_enabled: bool = False,
    material_binding: bool = True,
    material_type: str = "Material",
    collision_type: str = "Mesh",
    collision_api: bool = True,
    collision_enabled: bool = True,
    mesh_collision_api: bool = True,
    approximation: str = "convexDecomposition",
    visibility: str = "inherited",
    collision_translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    collision_reset_xform_stack: bool = False,
    collision_scale: tuple[float, float, float] | None = None,
    collision_orient_wxyz: tuple[float, float, float, float] | None = None,
    right_inner_x: float = 19.70049476623535,
    points: list[Gf.Vec3f] | None = None,
    face_counts: list[int] | None = None,
    face_indices: list[int] | None = None,
    authored_extent: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, _BIN_DEFAULT_PRIM_PATH).GetPrim()
    root_xform = UsdGeom.Xformable(root)
    root_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )
    root_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*root_scale)
    )
    if rigid_api:
        rigid = UsdPhysics.RigidBodyAPI.Apply(root)
        rigid.CreateRigidBodyEnabledAttr(rigid_enabled)
        rigid.CreateKinematicEnabledAttr(kinematic_enabled)

    material_path = f"{_BIN_DEFAULT_PRIM_PATH}/{_BIN_MATERIAL_SUBPATH}"
    if material_type == "Material":
        material = UsdShade.Material.Define(stage, material_path)
    else:
        material = stage.DefinePrim(material_path, material_type)
    if material_binding:
        UsdShade.MaterialBindingAPI.Apply(root).Bind(UsdShade.Material(material))

    collision_path = f"{_BIN_DEFAULT_PRIM_PATH}/{_BIN_SOURCE_COLLISION_SUBPATH}"
    if collision_type == "Mesh":
        mesh = UsdGeom.Mesh.Define(stage, collision_path)
        if points is None or face_counts is None or face_indices is None:
            valid_points, valid_counts, valid_indices = _valid_mesh_geometry(
                right_inner_x=right_inner_x
            )
            points = valid_points if points is None else points
            face_counts = valid_counts if face_counts is None else face_counts
            face_indices = valid_indices if face_indices is None else face_indices
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        extent = authored_extent or (
            _BIN_SOURCE_LOWER_RAW,
            _BIN_SOURCE_UPPER_RAW,
        )
        mesh.CreateExtentAttr(
            [
                Gf.Vec3f(*extent[0]),
                Gf.Vec3f(*extent[1]),
            ]
        )
        imageable = UsdGeom.Imageable(mesh)
        imageable.CreateVisibilityAttr(visibility)
        collision_prim = mesh.GetPrim()
    else:
        collision_prim = stage.DefinePrim(collision_path, collision_type)
    collision_xform = UsdGeom.Xformable(collision_prim)
    collision_xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*collision_translate)
    )
    if collision_scale is not None:
        collision_xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(*collision_scale)
        )
    if collision_orient_wxyz is not None:
        collision_xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(
                collision_orient_wxyz[0],
                Gf.Vec3d(*collision_orient_wxyz[1:]),
            )
        )
    collision_xform.SetResetXformStack(collision_reset_xform_stack)
    if collision_api:
        UsdPhysics.CollisionAPI.Apply(collision_prim).CreateCollisionEnabledAttr(
            collision_enabled
        )
    if mesh_collision_api:
        UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr(
            approximation
        )

    if default_prim_path == _BIN_DEFAULT_PRIM_PATH:
        stage.SetDefaultPrim(root)
    else:
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, default_prim_path).GetPrim())
    stage.GetRootLayer().Save()


def _write_valid_source(path: Path) -> None:
    _write_source(path)


def _grocery_bin(
    source_path: Path,
    *,
    object_config: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        usd_path=str(source_path),
        bounding_box=("source", "bounds"),
        object_cfg=object_config
        or SimpleNamespace(
            spawn=SimpleNamespace(
                usd_path=str(source_path),
                unrelated_setting="preserved",
            )
        ),
    )


def _compose_live_stage(
    path: Path,
    *,
    override_path: str,
    bin_prim_path: str,
) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    grocery_bin = stage.DefinePrim(bin_prim_path)
    grocery_bin.GetReferences().AddReference(
        assetPath=override_path,
        primPath=Sdf.Path(_BIN_DEFAULT_PRIM_PATH),
    )
    return stage


def test_source_certificate_uses_raw_units_and_metric_root_scale(
    tmp_path: Path,
) -> None:
    source_path = _source_path(tmp_path)
    _write_valid_source(source_path)
    stage = Usd.Stage.Open(str(source_path))
    mesh = UsdGeom.Mesh(
        stage.GetPrimAtPath(f"{_BIN_DEFAULT_PRIM_PATH}/{_BIN_SOURCE_COLLISION_SUBPATH}")
    )

    raw_gap = _require_source_cavity_certificate(mesh)

    assert math.isclose(
        raw_gap,
        _CERTIFIED_MIN_TRIANGLE_AABB_GAP_RAW,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        raw_gap * _BIN_ROOT_SCALE[0],
        _CERTIFIED_MIN_TRIANGLE_AABB_GAP_M,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert _CERTIFIED_MIN_TRIANGLE_AABB_GAP_M > 0.0049


def test_proxy_boxes_exactly_partition_outer_prism_around_cavity() -> None:
    _require_proxy_cavity_partition()

    metric_cavity_lower = tuple(
        value * _BIN_ROOT_SCALE[index]
        for index, value in enumerate((-19.0, -13.5, 1.0))
    )
    metric_cavity_upper = tuple(
        value * _BIN_ROOT_SCALE[index] for index, value in enumerate((19.0, 13.5, 15.0))
    )
    assert metric_cavity_lower == pytest.approx((-0.133, -0.0945, 0.007))
    assert metric_cavity_upper == pytest.approx((0.133, 0.0945, 0.105))


def test_override_is_instance_local_exact_and_owned_until_close(
    tmp_path: Path,
) -> None:
    source_path = _source_path(tmp_path)
    _write_valid_source(source_path)
    shared_object_config = SimpleNamespace(
        spawn=SimpleNamespace(
            usd_path=str(source_path),
            unrelated_setting="preserved",
        )
    )
    grocery_bin = _grocery_bin(
        source_path,
        object_config=shared_object_config,
    )
    peer = _grocery_bin(
        source_path,
        object_config=shared_object_config,
    )

    override = apply_grocery_bin_collision_override(grocery_bin)
    override_path = Path(override.override_usd_path)
    try:
        assert grocery_bin.object_cfg is not shared_object_config
        assert grocery_bin.object_cfg.spawn is not shared_object_config.spawn
        assert grocery_bin.usd_path == str(override_path)
        assert grocery_bin.object_cfg.spawn.usd_path == str(override_path)
        assert grocery_bin.object_cfg.spawn.unrelated_setting == "preserved"
        assert grocery_bin.bounding_box is None
        assert peer.usd_path == str(source_path)
        assert peer.object_cfg is shared_object_config
        assert peer.object_cfg.spawn.usd_path == str(source_path)
        assert peer.bounding_box == ("source", "bounds")
        assert override_path.is_file()

        source_stage = Usd.Stage.Open(str(source_path))
        override_stage = Usd.Stage.Open(str(override_path))
        assert str(override_stage.GetDefaultPrim().GetPath()) == _BIN_DEFAULT_PRIM_PATH
        root_spec = override_stage.GetRootLayer().GetPrimAtPath(_BIN_DEFAULT_PRIM_PATH)
        references = tuple(root_spec.GetInfo("references").GetAppliedItems())
        assert len(references) == 1
        assert references[0].assetPath == str(source_path)

        source_collision_path = (
            f"{_BIN_DEFAULT_PRIM_PATH}/{_BIN_SOURCE_COLLISION_SUBPATH}"
        )
        source_collision = source_stage.GetPrimAtPath(source_collision_path)
        override_collision = override_stage.GetPrimAtPath(source_collision_path)
        assert (
            UsdPhysics.CollisionAPI(source_collision).GetCollisionEnabledAttr().Get()
            is True
        )
        assert (
            UsdPhysics.CollisionAPI(override_collision).GetCollisionEnabledAttr().Get()
            is False
        )
        assert override_collision.GetTypeName() == "Mesh"
        assert (
            UsdGeom.Imageable(override_collision).ComputeVisibility()
            == UsdGeom.Tokens.inherited
        )
        assert override_stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI)
        material_path = f"{_BIN_DEFAULT_PRIM_PATH}/{_BIN_MATERIAL_SUBPATH}"
        assert override_stage.GetPrimAtPath(material_path).GetTypeName() == "Material"
        assert tuple(
            str(path)
            for path in override_stage.GetDefaultPrim()
            .GetRelationship("material:binding")
            .GetTargets()
        ) == (material_path,)

        assert validate_live_grocery_bin_collision_contract(
            override_stage,
            bin_prim_path=_BIN_DEFAULT_PRIM_PATH,
        ) == (1, 5)
        for spec in _BIN_PROXY_BOX_SPECS:
            proxy = override_stage.GetPrimAtPath(
                f"{_BIN_DEFAULT_PRIM_PATH}/{spec.subpath}"
            )
            assert proxy.GetTypeName() == "Cube"
            assert (
                UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Get() is True
            )
            assert not proxy.HasAPI(UsdPhysics.MeshCollisionAPI)
            assert (
                UsdGeom.Imageable(proxy).ComputeVisibility() == UsdGeom.Tokens.invisible
            )
            operations = UsdGeom.Xformable(proxy).GetOrderedXformOps()
            assert tuple(op.GetOpName() for op in operations) == (
                "xformOp:translate",
                "xformOp:scale",
            )
            assert UsdGeom.Xformable(proxy).GetResetXformStack() is False
            assert tuple(float(value) for value in operations[0].Get()) == (
                spec.center_raw
            )
            assert tuple(float(value) for value in operations[1].Get()) == (
                spec.size_raw
            )
            assert math.isclose(
                proxy.GetAttribute("physxCollision:contactOffset").Get(),
                _BIN_PROXY_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            assert math.isclose(
                proxy.GetAttribute("physxCollision:restOffset").Get(),
                _BIN_PROXY_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
    finally:
        override.close()

    assert not override_path.exists()
    override.close()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            {"default_prim_path": "/wrong_bin"},
            "default prim must be /grey_bin",
        ),
        ({"root_scale": (0.07, 0.07, 0.07)}, "root scale drifted"),
        ({"rigid_api": False}, "rigid root contract drifted"),
        ({"rigid_enabled": False}, "rigid root contract drifted"),
        ({"kinematic_enabled": True}, "rigid root contract drifted"),
        ({"material_binding": False}, "visual material binding drifted"),
        ({"material_type": "Scope"}, "visual material binding drifted"),
        (
            {"collision_type": "Xform"},
            "source collision/visual contract drifted",
        ),
        (
            {"collision_api": False},
            "source collision/visual contract drifted",
        ),
        (
            {"collision_enabled": False},
            "source collision/visual contract drifted",
        ),
        (
            {"mesh_collision_api": False},
            "source collision/visual contract drifted",
        ),
        (
            {"approximation": "convexHull"},
            "source collision/visual contract drifted",
        ),
        (
            {"visibility": "invisible"},
            "source collision/visual contract drifted",
        ),
        (
            {"collision_translate": (100.0, 0.0, 0.0)},
            "source collider transform drifted",
        ),
        (
            {"collision_reset_xform_stack": True},
            "source collider transform drifted",
        ),
        (
            {"collision_scale": (1.0, 1.0, 1.0)},
            "source collider transform drifted",
        ),
        (
            {
                "collision_orient_wxyz": (
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
            },
            "source collider transform drifted",
        ),
        (
            {"right_inner_x": 18.5},
            "source triangles overlap the certified cavity",
        ),
        (
            {"right_inner_x": 19.5},
            "source cavity clearance drifted below",
        ),
        (
            {
                "authored_extent": (
                    _BIN_SOURCE_LOWER_RAW,
                    (31.0, *_BIN_SOURCE_UPPER_RAW[1:]),
                )
            },
            "authored upper extent drifted",
        ),
    ],
)
def test_override_rejects_source_contract_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    match: str,
) -> None:
    source_path = _source_path(tmp_path)
    _write_source(source_path, **mutation)

    with pytest.raises(RuntimeError, match=match):
        GroceryBinCollisionOverride(str(source_path))


@pytest.mark.parametrize(
    ("face_counts", "face_indices", "match"),
    [
        ([], [], "source mesh has no faces"),
        ([2], [0, 1], "degenerate face"),
        ([4], [0, 1, 2], "face topology is inconsistent"),
        ([3], [0, 1, 999], "invalid point index"),
    ],
)
def test_override_rejects_source_topology_drift(
    tmp_path: Path,
    face_counts: list[int],
    face_indices: list[int],
    match: str,
) -> None:
    source_path = _source_path(tmp_path)
    points, _, _ = _valid_mesh_geometry()
    _write_source(
        source_path,
        points=points,
        face_counts=face_counts,
        face_indices=face_indices,
    )

    with pytest.raises(RuntimeError, match=match):
        GroceryBinCollisionOverride(str(source_path))


def test_override_rejects_unrecognized_source_identity(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "grey_bin.usd"
    _write_valid_source(source_path)

    with pytest.raises(RuntimeError, match="requires the pinned grey-bin USD"):
        GroceryBinCollisionOverride(str(source_path))


def test_constructor_cleans_temporary_layer_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = _source_path(tmp_path)
    _write_valid_source(source_path)
    captured_directory: list[Path] = []

    def fail_write(source_usd_path: str, override_path: Path) -> None:
        assert source_usd_path == str(source_path)
        captured_directory.append(override_path.parent)
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(
        bin_override_module,
        "_write_bin_override",
        fail_write,
    )

    with pytest.raises(RuntimeError, match="injected write failure"):
        GroceryBinCollisionOverride(str(source_path))

    assert len(captured_directory) == 1
    assert not captured_directory[0].exists()


@pytest.mark.parametrize(
    "drift",
    [
        "source_enabled",
        "source_invisible",
        "proxy_disabled",
        "proxy_mesh_collision",
        "proxy_visible",
        "proxy_center",
        "proxy_size",
        "proxy_cube_size",
        "proxy_reset_xform_stack",
        "contact_offset",
        "rest_offset",
        "root_scale",
        "material",
        "extra_collider",
    ],
)
def test_live_validator_rejects_composed_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    source_path = _source_path(tmp_path)
    _write_valid_source(source_path)
    override = GroceryBinCollisionOverride(str(source_path))
    bin_prim_path = "/World/envs/env_0/grey_bin"
    try:
        stage = _compose_live_stage(
            tmp_path / "live.usda",
            override_path=override.override_usd_path,
            bin_prim_path=bin_prim_path,
        )
        assert validate_live_grocery_bin_collision_contract(
            stage,
            bin_prim_path=bin_prim_path,
        ) == (1, 5)

        source_collision = stage.GetPrimAtPath(
            f"{bin_prim_path}/{_BIN_SOURCE_COLLISION_SUBPATH}"
        )
        proxy = stage.GetPrimAtPath(
            f"{bin_prim_path}/{_BIN_PROXY_BOX_SPECS[0].subpath}"
        )
        if drift == "source_enabled":
            UsdPhysics.CollisionAPI(source_collision).GetCollisionEnabledAttr().Set(
                True
            )
        elif drift == "source_invisible":
            UsdGeom.Imageable(source_collision).GetVisibilityAttr().Set(
                UsdGeom.Tokens.invisible
            )
        elif drift == "proxy_disabled":
            UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Set(False)
        elif drift == "proxy_mesh_collision":
            UsdPhysics.MeshCollisionAPI.Apply(proxy)
        elif drift == "proxy_visible":
            UsdGeom.Imageable(proxy).GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        elif drift == "proxy_center":
            UsdGeom.Xformable(proxy).GetOrderedXformOps()[0].Set(
                Gf.Vec3d(0.0, 0.0, 0.6)
            )
        elif drift == "proxy_size":
            UsdGeom.Xformable(proxy).GetOrderedXformOps()[1].Set(
                Gf.Vec3d(60.0, 40.0, 2.0)
            )
        elif drift == "proxy_cube_size":
            UsdGeom.Cube(proxy).GetSizeAttr().Set(2.0)
        elif drift == "proxy_reset_xform_stack":
            UsdGeom.Xformable(proxy).SetResetXformStack(True)
        elif drift == "contact_offset":
            proxy.GetAttribute("physxCollision:contactOffset").Set(0.01)
        elif drift == "rest_offset":
            proxy.GetAttribute("physxCollision:restOffset").Set(-0.01)
        elif drift == "root_scale":
            stage.GetPrimAtPath(bin_prim_path).GetAttribute("xformOp:scale").Set(
                Gf.Vec3d(0.01, 0.01, 0.01)
            )
        elif drift == "material":
            stage.GetPrimAtPath(bin_prim_path).GetRelationship(
                "material:binding"
            ).SetTargets([Sdf.Path(f"{bin_prim_path}/Looks/wrong")])
        elif drift == "extra_collider":
            extra = UsdGeom.Cube.Define(
                stage,
                f"{bin_prim_path}/unexpected_collision",
            ).GetPrim()
            UsdPhysics.CollisionAPI.Apply(extra).CreateCollisionEnabledAttr(True)
        else:
            raise AssertionError(f"unhandled drift case {drift}")

        with pytest.raises(RuntimeError, match="live CAP grocery bin"):
            validate_live_grocery_bin_collision_contract(
                stage,
                bin_prim_path=bin_prim_path,
            )
    finally:
        override.close()
