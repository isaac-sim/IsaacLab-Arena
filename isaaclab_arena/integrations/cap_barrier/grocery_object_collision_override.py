# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Analytic collision proxy for the CAP grocery soup can."""

from __future__ import annotations

import copy
import math
import tempfile
from pathlib import Path
from typing import Any

_EXPECTED_CAN_SOURCE_URI_SUFFIX = (
    "/Arena/assets/object_library/srl_robolab_assets/objects/hope/alphabet_soup_can.usd"
)
_CAN_DEFAULT_PRIM_PATH = "/alphabet_soup_can"
_CAN_SOURCE_COLLISION_SUBPATH = "obj_000001_Mesh"
_CAN_PROXY_SUBPATH = "cap_collision_proxy"
_CAN_PROXY_RADIUS_M = 0.035780169069767
_CAN_PROXY_HEIGHT_M = 0.08355461061000824
_CAN_PROXY_CONTACT_OFFSET_M = 0.00025
_CAN_PROXY_REST_OFFSET_M = 0.0
_CAN_PHYSICS_MATERIAL_PATH = "/alphabet_soup_can/physics_material"
_CAN_EXPECTED_MASS_KG = 0.5
_ANALYTIC_CYLINDER_SETTING_PATH = "/physics/collisionApproximateCylinders"
_METRIC_TRANSFORM_TOLERANCE = 1e-9


def _require_can_source_identity(source_usd_path: str) -> None:
    normalized_path = source_usd_path.replace("\\", "/")
    if not normalized_path.endswith(_EXPECTED_CAN_SOURCE_URI_SUFFIX):
        raise RuntimeError(
            "CAP grocery can collision override requires the pinned soup-can USD; "
            f"got {source_usd_path!r}"
        )


def configure_analytic_cylinder_collisions() -> str:
    """Pin native PhysX cylinder collision before the scene is parsed."""
    import carb
    from omni.physx.bindings import _physx

    setting_path = _physx.SETTING_COLLISION_APPROXIMATE_CYLINDERS
    if setting_path != _ANALYTIC_CYLINDER_SETTING_PATH:
        raise RuntimeError(
            f"PhysX analytic-cylinder setting path drifted: {setting_path!r}"
        )
    settings = carb.settings.get_settings()
    settings.set_bool(setting_path, False)
    if settings.get_as_bool(setting_path) is not False:
        raise RuntimeError("PhysX refused the CAP analytic-cylinder collision setting")
    return setting_path


class AnalyticCylinderCollisionSettingOverride:
    """Own and restore the process-global native-cylinder PhysX setting."""

    def __init__(self) -> None:
        import carb
        from omni.physx.bindings import _physx

        setting_path = _physx.SETTING_COLLISION_APPROXIMATE_CYLINDERS
        if setting_path != _ANALYTIC_CYLINDER_SETTING_PATH:
            raise RuntimeError(
                f"PhysX analytic-cylinder setting path drifted: {setting_path!r}"
            )
        settings = carb.settings.get_settings()
        previous_value = settings.get_as_bool(setting_path)
        if not isinstance(previous_value, bool):
            raise RuntimeError("PhysX analytic-cylinder setting did not yield a bool")
        self.setting_path = setting_path
        self._settings = settings
        self._previous_value = previous_value
        self._closed = False
        try:
            settings.set_bool(setting_path, False)
            if settings.get_as_bool(setting_path) is not False:
                raise RuntimeError(
                    "PhysX refused the CAP analytic-cylinder collision setting"
                )
        except BaseException:
            settings.set_bool(setting_path, previous_value)
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._settings.set_bool(self.setting_path, self._previous_value)
        if self._settings.get_as_bool(self.setting_path) is not self._previous_value:
            raise RuntimeError("PhysX refused to restore the analytic-cylinder setting")
        self._closed = True


def validate_analytic_cylinder_collision_setting() -> str:
    """Reject a setting change that occurred while PhysX parsed the scene."""
    import carb
    from omni.physx.bindings import _physx

    setting_path = _physx.SETTING_COLLISION_APPROXIMATE_CYLINDERS
    if setting_path != _ANALYTIC_CYLINDER_SETTING_PATH:
        raise RuntimeError(
            f"PhysX analytic-cylinder setting path drifted: {setting_path!r}"
        )
    if carb.settings.get_settings().get_as_bool(setting_path) is not False:
        raise RuntimeError(
            "PhysX analytic-cylinder collision setting changed during scene load"
        )
    return setting_path


def _require_can_default_prim(stage: Any, *, label: str) -> Any:
    default_prim = stage.GetDefaultPrim()
    if (
        not default_prim.IsValid()
        or str(default_prim.GetPath()) != _CAN_DEFAULT_PRIM_PATH
    ):
        raise RuntimeError(f"{label} default prim must be {_CAN_DEFAULT_PRIM_PATH}")
    return default_prim


def _require_unit_metric_transform(
    stage: Any,
    *,
    prim_path: str,
    label: str,
) -> None:
    """Reject inherited scale or shear that the pose-only guard cannot represent."""
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid() or not prim.IsDefined():
        raise RuntimeError(f"{label} metric prim is missing: {prim_path}")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)
    axes = tuple(
        tuple(float(value) for value in matrix.TransformDir(direction))
        for direction in (
            Gf.Vec3d(1.0, 0.0, 0.0),
            Gf.Vec3d(0.0, 1.0, 0.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
    )
    squared_norms = tuple(
        sum(component * component for component in axis) for axis in axes
    )
    pairwise_dots = (
        sum(a * b for a, b in zip(axes[0], axes[1], strict=True)),
        sum(a * b for a, b in zip(axes[0], axes[2], strict=True)),
        sum(a * b for a, b in zip(axes[1], axes[2], strict=True)),
    )
    cross_xy = (
        axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
        axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
        axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0],
    )
    handed_volume = sum(
        component * axis_component
        for component, axis_component in zip(cross_xy, axes[2], strict=True)
    )
    if (
        any(
            not math.isclose(
                value,
                1.0,
                rel_tol=0.0,
                abs_tol=_METRIC_TRANSFORM_TOLERANCE,
            )
            for value in squared_norms
        )
        or any(
            not math.isclose(
                value,
                0.0,
                rel_tol=0.0,
                abs_tol=_METRIC_TRANSFORM_TOLERANCE,
            )
            for value in pairwise_dots
        )
        or not math.isclose(
            handed_volume,
            1.0,
            rel_tol=0.0,
            abs_tol=_METRIC_TRANSFORM_TOLERANCE,
        )
    ):
        raise RuntimeError(
            f"{label} metric transform drifted: axes={axes!r}, "
            f"handed_volume={handed_volume!r}"
        )


def _require_can_source_contract(stage: Any) -> None:
    from pxr import UsdGeom, UsdPhysics

    root = _require_can_default_prim(stage, label="CAP grocery can source USD")
    _require_unit_metric_transform(
        stage,
        prim_path=_CAN_DEFAULT_PRIM_PATH,
        label="CAP grocery can source USD",
    )
    if not root.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError("CAP grocery can source root is not a rigid body")
    mass = UsdPhysics.MassAPI(root).GetMassAttr().Get()
    if not isinstance(mass, float) or not math.isclose(
        mass,
        _CAN_EXPECTED_MASS_KG,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(f"CAP grocery can source mass drifted: {mass!r}")

    collision_path = f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_SOURCE_COLLISION_SUBPATH}"
    collision = stage.GetPrimAtPath(collision_path)
    if (
        not collision.IsValid()
        or collision.GetTypeName() != "Mesh"
        or not collision.HasAPI(UsdPhysics.CollisionAPI)
        or not collision.HasAPI(UsdPhysics.MeshCollisionAPI)
        or UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
        is not True
    ):
        raise RuntimeError("CAP grocery can source collision mesh contract drifted")
    collision_xform = UsdGeom.Xformable(collision)
    if collision_xform.GetResetXformStack() or collision_xform.GetOrderedXformOps():
        raise RuntimeError("CAP grocery can source collision transform drifted")
    enabled_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(root.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    }
    if enabled_collision_paths != {collision_path}:
        raise RuntimeError(
            "CAP grocery can source enabled collision set drifted: "
            f"{sorted(enabled_collision_paths)!r}"
        )
    approximation = UsdPhysics.MeshCollisionAPI(collision).GetApproximationAttr().Get()
    shrink_wrap = collision.GetAttribute(
        "physxConvexDecompositionCollision:shrinkWrap"
    ).Get()
    if approximation != "convexDecomposition" or shrink_wrap is not True:
        raise RuntimeError(
            "CAP grocery can source convex-decomposition contract drifted"
        )

    mesh = UsdGeom.Mesh(collision)
    points = mesh.GetPointsAttr().Get()
    if points is None or not points:
        raise RuntimeError("CAP grocery can source mesh has no points")
    if any(
        not math.isfinite(float(coordinate)) for point in points for coordinate in point
    ):
        raise RuntimeError("CAP grocery can source mesh contains non-finite geometry")

    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if face_counts is None or not face_counts:
        raise RuntimeError("CAP grocery can source mesh has no faces")
    if face_indices is None or sum(int(count) for count in face_counts) != len(
        face_indices
    ):
        raise RuntimeError("CAP grocery can source mesh face topology is inconsistent")
    point_count = len(points)
    offset = 0
    triangle_count = 0
    for count_value in face_counts:
        count = int(count_value)
        if count < 3:
            raise RuntimeError("CAP grocery can source mesh contains a degenerate face")
        face = tuple(int(index) for index in face_indices[offset : offset + count])
        offset += count
        if any(index < 0 or index >= point_count for index in face):
            raise RuntimeError(
                "CAP grocery can source mesh contains an invalid point index"
            )
        for vertex_index in range(1, count - 1):
            first = points[face[0]]
            second = points[face[vertex_index]]
            third = points[face[vertex_index + 1]]
            first_edge = tuple(
                float(second[axis]) - float(first[axis]) for axis in range(3)
            )
            second_edge = tuple(
                float(third[axis]) - float(first[axis]) for axis in range(3)
            )
            cross = (
                first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
                first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
                first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
            )
            if not all(math.isfinite(component) for component in cross) or not any(
                component != 0.0 for component in cross
            ):
                raise RuntimeError(
                    "CAP grocery can source mesh contains a degenerate triangle"
                )
            triangle_count += 1
    if triangle_count == 0:
        raise RuntimeError("CAP grocery can source mesh has no triangles")

    maximum_radius = max(
        math.hypot(float(point[0]), float(point[1])) for point in points
    )
    lower_z = min(float(point[2]) for point in points)
    upper_z = max(float(point[2]) for point in points)
    if (
        maximum_radius > _CAN_PROXY_RADIUS_M
        or lower_z < -0.5 * _CAN_PROXY_HEIGHT_M
        or upper_z > 0.5 * _CAN_PROXY_HEIGHT_M
        or _CAN_PROXY_RADIUS_M - maximum_radius > 1e-6
        or -0.5 * _CAN_PROXY_HEIGHT_M - lower_z > 1e-6
        or 0.5 * _CAN_PROXY_HEIGHT_M - upper_z > 1e-6
    ):
        raise RuntimeError("CAP grocery can source no longer fits its analytic proxy")


def _author_can_proxy(stage: Any) -> None:
    from pxr import Sdf, UsdGeom, UsdPhysics

    source_collision = stage.OverridePrim(
        f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_SOURCE_COLLISION_SUBPATH}"
    )
    UsdPhysics.CollisionAPI(source_collision).CreateCollisionEnabledAttr(False).Set(
        False
    )

    proxy = UsdGeom.Cylinder.Define(
        stage,
        f"{_CAN_DEFAULT_PRIM_PATH}/{_CAN_PROXY_SUBPATH}",
    )
    proxy.CreateAxisAttr(UsdGeom.Tokens.z)
    proxy.CreateRadiusAttr(_CAN_PROXY_RADIUS_M)
    proxy.CreateHeightAttr(_CAN_PROXY_HEIGHT_M)
    proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    prim = proxy.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute(
        "physxCollision:contactOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(_CAN_PROXY_CONTACT_OFFSET_M)
    prim.CreateAttribute(
        "physxCollision:restOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(_CAN_PROXY_REST_OFFSET_M)
    prim.AddAppliedSchema("MaterialBindingAPI")
    prim.CreateRelationship("material:binding:physics").SetTargets(
        [Sdf.Path(_CAN_PHYSICS_MATERIAL_PATH)]
    )


def _has_applied_schema(prim: Any, schema_name: str) -> bool:
    schemas = prim.GetMetadata("apiSchemas")
    return schemas is not None and schema_name in tuple(schemas.GetAppliedItems())


def _require_can_proxy(
    stage: Any,
    *,
    label: str,
    prim_root: str = _CAN_DEFAULT_PRIM_PATH,
) -> None:
    from pxr import UsdGeom, UsdPhysics

    source_collision = stage.GetPrimAtPath(
        f"{prim_root}/{_CAN_SOURCE_COLLISION_SUBPATH}"
    )
    if (
        not source_collision.IsValid()
        or UsdPhysics.CollisionAPI(source_collision).GetCollisionEnabledAttr().Get()
        is not False
    ):
        raise RuntimeError(f"{label} source collider is not disabled")
    source_xform = UsdGeom.Xformable(source_collision)
    if source_xform.GetResetXformStack() or source_xform.GetOrderedXformOps():
        raise RuntimeError(f"{label} source collider transform drifted")

    proxy_path = f"{prim_root}/{_CAN_PROXY_SUBPATH}"
    proxy_prim = stage.GetPrimAtPath(proxy_path)
    if (
        not proxy_prim.IsValid()
        or proxy_prim.GetTypeName() != "Cylinder"
        or not proxy_prim.HasAPI(UsdPhysics.CollisionAPI)
        or proxy_prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        or UsdPhysics.CollisionAPI(proxy_prim).GetCollisionEnabledAttr().Get()
        is not True
    ):
        raise RuntimeError(f"{label} analytic cylinder contract drifted")
    root = stage.GetPrimAtPath(prim_root)
    enabled_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(root.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    }
    if enabled_collision_paths != {proxy_path}:
        raise RuntimeError(
            f"{label} enabled collision set drifted: "
            f"{sorted(enabled_collision_paths)!r}"
        )
    cylinder = UsdGeom.Cylinder(proxy_prim)
    xformable = UsdGeom.Xformable(proxy_prim)
    contact_offset = proxy_prim.GetAttribute("physxCollision:contactOffset").Get()
    rest_offset = proxy_prim.GetAttribute("physxCollision:restOffset").Get()
    material_targets = tuple(
        str(path)
        for path in proxy_prim.GetRelationship("material:binding:physics").GetTargets()
    )
    expected_material_path = (
        f"{prim_root}/physics_material"
        if prim_root != _CAN_DEFAULT_PRIM_PATH
        else _CAN_PHYSICS_MATERIAL_PATH
    )
    if not (
        cylinder.GetAxisAttr().Get() == UsdGeom.Tokens.z
        and math.isclose(
            cylinder.GetRadiusAttr().Get(),
            _CAN_PROXY_RADIUS_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and math.isclose(
            cylinder.GetHeightAttr().Get(),
            _CAN_PROXY_HEIGHT_M,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        and proxy_prim.GetAttribute("visibility").Get() == UsdGeom.Tokens.invisible
        and not xformable.GetOrderedXformOps()
        and xformable.GetResetXformStack() is False
        and _has_applied_schema(proxy_prim, "PhysxCollisionAPI")
        and math.isclose(
            contact_offset,
            _CAN_PROXY_CONTACT_OFFSET_M,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and math.isclose(
            rest_offset,
            _CAN_PROXY_REST_OFFSET_M,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        and material_targets == (expected_material_path,)
    ):
        raise RuntimeError(f"{label} analytic cylinder attributes drifted")


def _write_can_override(
    source_usd_path: str,
    override_usd_path: Path,
) -> None:
    from pxr import Usd

    stage = Usd.Stage.CreateNew(str(override_usd_path))
    if stage is None:
        raise RuntimeError(
            f"failed to create can collision override {override_usd_path}"
        )
    root = stage.DefinePrim(_CAN_DEFAULT_PRIM_PATH)
    root.GetReferences().AddReference(assetPath=source_usd_path)
    stage.SetDefaultPrim(root)
    _author_can_proxy(stage)
    stage.GetRootLayer().Save()


class GroceryCanCollisionOverride:
    """Own one verified temporary soup-can analytic collision layer."""

    def __init__(self, source_usd_path: str):
        from pxr import Usd

        if not isinstance(source_usd_path, str) or not source_usd_path:
            raise RuntimeError("CAP grocery can is missing its source USD path")
        _require_can_source_identity(source_usd_path)
        source_stage = Usd.Stage.Open(source_usd_path)
        if source_stage is None:
            raise RuntimeError(
                f"failed to open CAP grocery can USD {source_usd_path!r}"
            )
        _require_can_source_contract(source_stage)

        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="cap_grocery_can_collision_"
        )
        self.source_usd_path = source_usd_path
        self.override_usd_path = str(
            Path(self._temporary_directory.name)
            / "alphabet_soup_can.cap_collision_override.usda"
        )
        try:
            _write_can_override(
                source_usd_path,
                Path(self.override_usd_path),
            )
            override_stage = Usd.Stage.Open(self.override_usd_path)
            if override_stage is None:
                raise RuntimeError(
                    "failed to reopen CAP grocery can collision override"
                )
            _require_can_default_prim(
                override_stage,
                label="CAP grocery can collision override",
            )
            _require_unit_metric_transform(
                override_stage,
                prim_path=_CAN_DEFAULT_PRIM_PATH,
                label="CAP grocery can collision override",
            )
            _require_can_proxy(
                override_stage,
                label="CAP grocery can collision override",
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        temporary_directory = self._temporary_directory
        if temporary_directory is None:
            return
        self._temporary_directory = None
        temporary_directory.cleanup()


def apply_grocery_can_collision_override(
    grocery: Any,
) -> GroceryCanCollisionOverride:
    """Replace one grocery asset's collider without mutating registry peers."""
    try:
        source_usd_path = grocery.usd_path
        object_config = copy.deepcopy(grocery.object_cfg)
        spawn = object_config.spawn
    except AttributeError as exc:
        raise RuntimeError("CAP grocery can asset has no mutable USD spawn") from exc
    if spawn.usd_path != source_usd_path:
        raise RuntimeError("CAP grocery can asset source and spawn paths disagree")

    override = GroceryCanCollisionOverride(source_usd_path)
    try:
        grocery.usd_path = override.override_usd_path
        spawn.usd_path = override.override_usd_path
        grocery.object_cfg = object_config
        grocery.bounding_box = None
    except BaseException:
        override.close()
        raise
    return override


def validate_live_grocery_can_collision_contract(
    stage: Any,
    *,
    can_prim_path: str,
) -> None:
    """Validate the composed live can using the instance's absolute prim path."""
    _require_unit_metric_transform(
        stage,
        prim_path=can_prim_path,
        label="live CAP grocery can",
    )
    _require_can_proxy(
        stage,
        prim_root=can_prim_path,
        label="live CAP grocery can",
    )
