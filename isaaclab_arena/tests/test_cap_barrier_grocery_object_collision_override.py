# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from isaaclab_arena.integrations.cap_barrier.grocery_object_collision_override import (
    _ANALYTIC_CYLINDER_SETTING_PATH,
    _CAN_DEFAULT_PRIM_PATH,
    _CAN_EXPECTED_MASS_KG,
    _CAN_PHYSICS_MATERIAL_PATH,
    _CAN_PROXY_CONTACT_OFFSET_M,
    _CAN_PROXY_HEIGHT_M,
    _CAN_PROXY_RADIUS_M,
    _CAN_PROXY_REST_OFFSET_M,
    _CAN_PROXY_SUBPATH,
    _CAN_SOURCE_COLLISION_SUBPATH,
    _EXPECTED_CAN_SOURCE_URI_SUFFIX,
    AnalyticCylinderCollisionSettingOverride,
    GroceryCanCollisionOverride,
    apply_grocery_can_collision_override,
    configure_analytic_cylinder_collisions,
    validate_analytic_cylinder_collision_setting,
    validate_live_grocery_can_collision_contract,
)


def _source_path(tmp_path: Path) -> Path:
    return tmp_path / _EXPECTED_CAN_SOURCE_URI_SUFFIX.lstrip("/")


def _source_points(
    *, radius_delta: float = 0.0, height_delta: float = 0.0
) -> list[Gf.Vec3f]:
    radius = _CAN_PROXY_RADIUS_M + radius_delta
    half_height = 0.5 * _CAN_PROXY_HEIGHT_M + height_delta
    return [
        Gf.Vec3f(radius, 0.0, 0.0),
        Gf.Vec3f(-radius, 0.0, 0.0),
        Gf.Vec3f(0.0, radius, 0.0),
        Gf.Vec3f(0.0, -radius, 0.0),
        Gf.Vec3f(0.0, 0.0, half_height),
        Gf.Vec3f(0.0, 0.0, -half_height),
    ]


_VALID_FACE_COUNTS = (3,) * 8
_VALID_FACE_INDICES = (
    0,
    2,
    4,
    2,
    1,
    4,
    1,
    3,
    4,
    3,
    0,
    4,
    2,
    0,
    5,
    1,
    2,
    5,
    3,
    1,
    5,
    0,
    3,
    5,
)


def _write_source(
    path: Path,
    *,
    default_prim_path: str = _CAN_DEFAULT_PRIM_PATH,
    root_scale: tuple[float, float, float] | None = None,
    mass_kg: float | None = _CAN_EXPECTED_MASS_KG,
    collision_type: str = "Mesh",
    collision_enabled: bool = True,
    collision_api: bool = True,
    mesh_collision_api: bool = True,
    approximation: str = "convexDecomposition",
    shrink_wrap: bool | None = True,
    points: list[Gf.Vec3f] | None = None,
    face_counts: tuple[int, ...] | list[int] | None = _VALID_FACE_COUNTS,
    face_indices: tuple[int, ...] | list[int] | None = _VALID_FACE_INDICES,
    collision_translate: tuple[float, float, float] | None = None,
    extra_enabled_collider: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, _CAN_DEFAULT_PRIM_PATH).GetPrim()
    if root_scale is not None:
        UsdGeom.Xformable(root).AddScaleOp().Set(Gf.Vec3d(*root_scale))
    UsdPhysics.RigidBodyAPI.Apply(root)
    if mass_kg is not None:
        UsdPhysics.MassAPI.Apply(root).CreateMassAttr(mass_kg)
    if default_prim_path == _CAN_DEFAULT_PRIM_PATH:
        stage.SetDefaultPrim(root)
    else:
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, default_prim_path).GetPrim())

    collision_path = f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_SOURCE_COLLISION_SUBPATH}"
    if collision_type == "Mesh":
        collision = UsdGeom.Mesh.Define(stage, collision_path)
        if points is not None:
            collision.CreatePointsAttr(points)
        if face_counts is not None:
            collision.CreateFaceVertexCountsAttr(face_counts)
        if face_indices is not None:
            collision.CreateFaceVertexIndicesAttr(face_indices)
        collision_prim = collision.GetPrim()
    else:
        collision_prim = stage.DefinePrim(collision_path, collision_type)
    if collision_api:
        UsdPhysics.CollisionAPI.Apply(collision_prim).CreateCollisionEnabledAttr(
            collision_enabled
        )
    if mesh_collision_api:
        UsdPhysics.MeshCollisionAPI.Apply(collision_prim).CreateApproximationAttr(
            approximation
        )
    if shrink_wrap is not None:
        collision_prim.CreateAttribute(
            "physxConvexDecompositionCollision:shrinkWrap",
            Sdf.ValueTypeNames.Bool,
        ).Set(shrink_wrap)
    if collision_translate is not None:
        UsdGeom.Xformable(collision_prim).AddTranslateOp().Set(
            Gf.Vec3d(*collision_translate)
        )
    if extra_enabled_collider:
        extra = UsdGeom.Cube.Define(
            stage,
            f"{_CAN_DEFAULT_PRIM_PATH}/unexpected_collision",
        ).GetPrim()
        UsdPhysics.CollisionAPI.Apply(extra).CreateCollisionEnabledAttr(True)
    UsdPhysics.MaterialAPI.Apply(stage.DefinePrim(_CAN_PHYSICS_MATERIAL_PATH))
    stage.GetRootLayer().Save()


def _write_valid_source(path: Path) -> None:
    _write_source(path, points=_source_points())


def _grocery(
    source_path: Path,
    *,
    object_config: SimpleNamespace | None = None,
) -> SimpleNamespace:
    if object_config is None:
        object_config = SimpleNamespace(
            spawn=SimpleNamespace(
                usd_path=str(source_path),
                unrelated_setting="preserved",
            )
        )
    return SimpleNamespace(
        usd_path=str(source_path),
        bounding_box=("source", "bounds"),
        object_cfg=object_config,
    )


def _install_fake_physx_settings(
    monkeypatch: pytest.MonkeyPatch,
    *,
    setting_path: str = _ANALYTIC_CYLINDER_SETTING_PATH,
) -> SimpleNamespace:
    settings = SimpleNamespace(
        value=True,
        calls=[],
    )

    def set_bool(path: str, value: bool) -> None:
        settings.calls.append((path, value))
        settings.value = value

    settings.set_bool = set_bool
    settings.get_as_bool = lambda path: settings.value
    carb = ModuleType("carb")
    carb.settings = SimpleNamespace(get_settings=lambda: settings)
    bindings = ModuleType("omni.physx.bindings")
    bindings._physx = SimpleNamespace(
        SETTING_COLLISION_APPROXIMATE_CYLINDERS=setting_path
    )
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "omni.physx.bindings", bindings)
    return settings


def _compose_live_stage(
    path: Path,
    *,
    override_path: str,
    can_prim_path: str,
) -> Usd.Stage:
    stage = Usd.Stage.CreateNew(str(path))
    can = stage.DefinePrim(can_prim_path)
    can.GetReferences().AddReference(
        assetPath=override_path,
        primPath=Sdf.Path(_CAN_DEFAULT_PRIM_PATH),
    )
    return stage


def test_analytic_cylinder_setting_is_pinned_before_scene_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _install_fake_physx_settings(monkeypatch)

    assert configure_analytic_cylinder_collisions() == _ANALYTIC_CYLINDER_SETTING_PATH
    assert settings.calls == [(_ANALYTIC_CYLINDER_SETTING_PATH, False)]
    assert settings.value is False
    assert (
        validate_analytic_cylinder_collision_setting()
        == _ANALYTIC_CYLINDER_SETTING_PATH
    )


def test_analytic_cylinder_setting_path_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _install_fake_physx_settings(
        monkeypatch,
        setting_path="/physics/silentReplacement",
    )

    with pytest.raises(RuntimeError, match="setting path drifted"):
        configure_analytic_cylinder_collisions()

    assert settings.calls == []


def test_analytic_cylinder_setting_change_during_load_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _install_fake_physx_settings(monkeypatch)
    configure_analytic_cylinder_collisions()
    settings.value = True

    with pytest.raises(RuntimeError, match="changed during scene load"):
        validate_analytic_cylinder_collision_setting()


@pytest.mark.parametrize("prior_value", [False, True])
def test_analytic_cylinder_setting_override_restores_owned_prior_value(
    monkeypatch: pytest.MonkeyPatch,
    prior_value: bool,
) -> None:
    settings = _install_fake_physx_settings(monkeypatch)
    settings.value = prior_value

    override = AnalyticCylinderCollisionSettingOverride()
    assert override.setting_path == _ANALYTIC_CYLINDER_SETTING_PATH
    assert settings.value is False

    override.close()
    assert settings.value is prior_value
    override.close()
    assert settings.value is prior_value


def test_override_is_instance_local_exact_and_owned_until_close(tmp_path: Path) -> None:
    source_path = _source_path(tmp_path)
    _write_valid_source(source_path)
    shared_object_config = SimpleNamespace(
        spawn=SimpleNamespace(
            usd_path=str(source_path),
            unrelated_setting="preserved",
        )
    )
    grocery = _grocery(source_path, object_config=shared_object_config)
    peer = _grocery(source_path, object_config=shared_object_config)

    override = apply_grocery_can_collision_override(grocery)
    override_path = Path(override.override_usd_path)
    try:
        assert grocery.object_cfg is not shared_object_config
        assert grocery.object_cfg.spawn is not shared_object_config.spawn
        assert grocery.usd_path == str(override_path)
        assert grocery.object_cfg.spawn.usd_path == str(override_path)
        assert grocery.object_cfg.spawn.unrelated_setting == "preserved"
        assert grocery.bounding_box is None
        assert peer.usd_path == str(source_path)
        assert peer.object_cfg is shared_object_config
        assert peer.object_cfg.spawn.usd_path == str(source_path)
        assert peer.bounding_box == ("source", "bounds")
        assert override_path.is_file()

        source_stage = Usd.Stage.Open(str(source_path))
        override_stage = Usd.Stage.Open(str(override_path))
        assert str(override_stage.GetDefaultPrim().GetPath()) == _CAN_DEFAULT_PRIM_PATH
        reference_spec = override_stage.GetRootLayer().GetPrimAtPath(
            _CAN_DEFAULT_PRIM_PATH
        )
        assert [
            reference.assetPath
            for reference in reference_spec.GetInfo("references").GetAppliedItems()
        ] == [str(source_path)]

        source_collision = source_stage.GetPrimAtPath(
            f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_SOURCE_COLLISION_SUBPATH}"
        )
        override_collision = override_stage.GetPrimAtPath(
            f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_SOURCE_COLLISION_SUBPATH}"
        )
        assert (
            UsdPhysics.CollisionAPI(source_collision).GetCollisionEnabledAttr().Get()
            is True
        )
        assert (
            UsdPhysics.CollisionAPI(override_collision).GetCollisionEnabledAttr().Get()
            is False
        )

        proxy = override_stage.GetPrimAtPath(
            f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_PROXY_SUBPATH}"
        )
        cylinder = UsdGeom.Cylinder(proxy)
        assert proxy.GetTypeName() == "Cylinder"
        assert proxy.HasAPI(UsdPhysics.CollisionAPI)
        assert not proxy.HasAPI(UsdPhysics.MeshCollisionAPI)
        assert UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Get() is True
        assert cylinder.GetAxisAttr().Get() == UsdGeom.Tokens.z
        assert math.isclose(
            cylinder.GetRadiusAttr().Get(),
            _CAN_PROXY_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            cylinder.GetHeightAttr().Get(),
            _CAN_PROXY_HEIGHT_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert proxy.GetAttribute("visibility").Get() == UsdGeom.Tokens.invisible
        assert "PhysxCollisionAPI" in tuple(
            proxy.GetMetadata("apiSchemas").GetAppliedItems()
        )
        assert math.isclose(
            proxy.GetAttribute("physxCollision:contactOffset").Get(),
            _CAN_PROXY_CONTACT_OFFSET_M,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        assert math.isclose(
            proxy.GetAttribute("physxCollision:restOffset").Get(),
            _CAN_PROXY_REST_OFFSET_M,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        assert tuple(
            str(target)
            for target in proxy.GetRelationship("material:binding:physics").GetTargets()
        ) == (_CAN_PHYSICS_MATERIAL_PATH,)
    finally:
        override.close()

    assert not override_path.exists()
    override.close()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            {"default_prim_path": "/wrong_can"},
            "default prim must be /alphabet_soup_can",
        ),
        ({"mass_kg": None}, "source mass drifted"),
        ({"mass_kg": 0.6}, "source mass drifted"),
        ({"root_scale": (1.0, 1.0, 1.01)}, "metric transform drifted"),
        ({"collision_type": "Xform"}, "source collision mesh contract drifted"),
        ({"collision_api": False}, "source collision mesh contract drifted"),
        ({"collision_enabled": False}, "source collision mesh contract drifted"),
        ({"mesh_collision_api": False}, "source collision mesh contract drifted"),
        ({"approximation": "convexHull"}, "convex-decomposition contract drifted"),
        ({"shrink_wrap": False}, "convex-decomposition contract drifted"),
        ({"shrink_wrap": None}, "convex-decomposition contract drifted"),
        (
            {"collision_translate": (0.001, 0.0, 0.0)},
            "source collision transform drifted",
        ),
        (
            {"extra_enabled_collider": True},
            "source enabled collision set drifted",
        ),
        ({"points": []}, "source mesh has no points"),
        (
            {"points": _source_points(radius_delta=2e-6)},
            "no longer fits its analytic proxy",
        ),
        (
            {"points": _source_points(radius_delta=-2e-6)},
            "no longer fits its analytic proxy",
        ),
        (
            {"points": _source_points(height_delta=2e-6)},
            "no longer fits its analytic proxy",
        ),
        (
            {"points": _source_points(height_delta=-2e-6)},
            "no longer fits its analytic proxy",
        ),
    ],
)
def test_override_rejects_source_contract_drift(
    tmp_path: Path,
    mutation: dict[str, object],
    match: str,
) -> None:
    source_path = _source_path(tmp_path)
    mutation.setdefault("points", _source_points())
    _write_source(source_path, **mutation)

    with pytest.raises(RuntimeError, match=match):
        GroceryCanCollisionOverride(str(source_path))


@pytest.mark.parametrize(
    ("points", "face_counts", "face_indices", "match"),
    [
        (
            [
                Gf.Vec3f(float("nan"), 0.0, 0.0),
                *_source_points()[1:],
            ],
            _VALID_FACE_COUNTS,
            _VALID_FACE_INDICES,
            "non-finite geometry",
        ),
        (
            _source_points(),
            [],
            [],
            "source mesh has no faces",
        ),
        (
            _source_points(),
            [2],
            [0, 1],
            "degenerate face",
        ),
        (
            _source_points(),
            [3],
            [0, 1],
            "face topology is inconsistent",
        ),
        (
            _source_points(),
            [3],
            [0, 1, 999],
            "invalid point index",
        ),
        (
            _source_points(),
            [3],
            [0, 0, 1],
            "degenerate triangle",
        ),
    ],
)
def test_override_rejects_nonfinite_or_invalid_source_topology(
    tmp_path: Path,
    points: list[Gf.Vec3f],
    face_counts: tuple[int, ...] | list[int],
    face_indices: tuple[int, ...] | list[int],
    match: str,
) -> None:
    source_path = _source_path(tmp_path)
    _write_source(
        source_path,
        points=points,
        face_counts=face_counts,
        face_indices=face_indices,
    )

    with pytest.raises(RuntimeError, match=match):
        GroceryCanCollisionOverride(str(source_path))


def test_override_rejects_unrecognized_source_identity(tmp_path: Path) -> None:
    source_path = tmp_path / "alphabet_soup_can.usd"
    _write_valid_source(source_path)

    with pytest.raises(RuntimeError, match="requires the pinned soup-can USD"):
        GroceryCanCollisionOverride(str(source_path))


@pytest.mark.parametrize(
    "drift",
    [
        "source_enabled",
        "proxy_disabled",
        "mesh_collision_api",
        "radius",
        "height",
        "local_transform",
        "visibility",
        "contact_offset",
        "rest_offset",
        "material",
        "source_transform",
        "root_scale",
        "extra_enabled_collider",
    ],
)
def test_live_validator_detects_composed_proxy_drift(
    tmp_path: Path, drift: str
) -> None:
    source_path = _source_path(tmp_path)
    _write_valid_source(source_path)
    override = GroceryCanCollisionOverride(str(source_path))
    can_prim_path = "/World/envs/env_0/objects/alphabet_soup_can"
    try:
        stage = _compose_live_stage(
            tmp_path / "live.usda",
            override_path=override.override_usd_path,
            can_prim_path=can_prim_path,
        )
        validate_live_grocery_can_collision_contract(stage, can_prim_path=can_prim_path)

        source_collision = stage.GetPrimAtPath(
            f"{can_prim_path}/{_CAN_SOURCE_COLLISION_SUBPATH}"
        )
        proxy = stage.GetPrimAtPath(f"{can_prim_path}/{_CAN_PROXY_SUBPATH}")
        if drift == "source_enabled":
            UsdPhysics.CollisionAPI(source_collision).GetCollisionEnabledAttr().Set(
                True
            )
        elif drift == "proxy_disabled":
            UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Set(False)
        elif drift == "mesh_collision_api":
            UsdPhysics.MeshCollisionAPI.Apply(proxy)
        elif drift == "radius":
            UsdGeom.Cylinder(proxy).GetRadiusAttr().Set(_CAN_PROXY_RADIUS_M + 1e-3)
        elif drift == "height":
            UsdGeom.Cylinder(proxy).GetHeightAttr().Set(_CAN_PROXY_HEIGHT_M + 1e-3)
        elif drift == "local_transform":
            UsdGeom.Xformable(proxy).AddTranslateOp().Set(Gf.Vec3d(0.001, 0.0, 0.0))
        elif drift == "visibility":
            imageable = UsdGeom.Imageable(proxy)
            imageable.GetVisibilityAttr().Set(UsdGeom.Tokens.inherited)
        elif drift == "contact_offset":
            proxy.GetAttribute("physxCollision:contactOffset").Set(0.01)
        elif drift == "rest_offset":
            proxy.GetAttribute("physxCollision:restOffset").Set(-0.01)
        elif drift == "material":
            proxy.GetRelationship("material:binding:physics").SetTargets(
                [Sdf.Path(f"{can_prim_path}/wrong_material")]
            )
        elif drift == "source_transform":
            UsdGeom.Xformable(source_collision).AddTranslateOp().Set(
                Gf.Vec3d(0.001, 0.0, 0.0)
            )
        elif drift == "root_scale":
            UsdGeom.Xformable(stage.GetPrimAtPath(can_prim_path)).AddScaleOp().Set(
                Gf.Vec3d(1.01, 1.0, 1.0)
            )
        elif drift == "extra_enabled_collider":
            extra = UsdGeom.Cube.Define(
                stage,
                f"{can_prim_path}/unexpected_collision",
            ).GetPrim()
            UsdPhysics.CollisionAPI.Apply(extra).CreateCollisionEnabledAttr(True)
        else:
            raise AssertionError(f"unhandled drift case {drift}")

        with pytest.raises(RuntimeError, match="live CAP grocery can"):
            validate_live_grocery_can_collision_contract(
                stage, can_prim_path=can_prim_path
            )
    finally:
        override.close()
