# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Verified analytic collision proxy for the CAP grocery grey bin."""

from __future__ import annotations

import copy
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPECTED_BIN_SOURCE_URI_SUFFIX = (
    "/Arena/assets/object_library/srl_robolab_assets/fixtures/grey_bin.usd"
)
_BIN_DEFAULT_PRIM_PATH = "/grey_bin"
_BIN_SOURCE_COLLISION_SUBPATH = "Tote_B09_01"
_BIN_MATERIAL_SUBPATH = "Looks/Plastic_Bin"
_BIN_ROOT_SCALE = (0.007, 0.007, 0.007)
_BIN_SOURCE_LOWER_RAW = (
    -30.0,
    -20.000001907348633,
    -5.960464477539063e-08,
)
_BIN_SOURCE_UPPER_RAW = (
    30.000003814697266,
    20.000001907348633,
    15.000004768371582,
)
_BIN_SOURCE_BOUNDS_TOLERANCE_RAW = 1e-9
_BIN_PROXY_OUTER_LOWER_RAW = (-30.0, -20.0, 0.0)
_BIN_PROXY_OUTER_UPPER_RAW = (30.0, 20.0, 15.0)

# This cavity is certified in the source mesh's raw coordinate system. The
# authored root scale converts one raw unit to 0.007 metres at runtime.
_CERTIFIED_CAVITY_LOWER_RAW = (-19.0, -13.5, 1.0)
_CERTIFIED_CAVITY_UPPER_RAW = (19.0, 13.5, 15.0)
_CERTIFIED_MIN_TRIANGLE_AABB_GAP_RAW = 0.7004947662353516
_CERTIFIED_MIN_TRIANGLE_AABB_GAP_M = (
    _CERTIFIED_MIN_TRIANGLE_AABB_GAP_RAW * _BIN_ROOT_SCALE[0]
)

# PhysX collision offsets are metric properties and are not multiplied by the
# USD root transform scale.
_BIN_PROXY_CONTACT_OFFSET_M = 0.001
_BIN_PROXY_REST_OFFSET_M = 0.0


@dataclass(frozen=True)
class _BinCollisionBoxSpec:
    name: str
    center_raw: tuple[float, float, float]
    half_extent_raw: tuple[float, float, float]

    @property
    def subpath(self) -> str:
        return f"cap_collision_proxy_{self.name}"

    @property
    def size_raw(self) -> tuple[float, float, float]:
        return tuple(2.0 * value for value in self.half_extent_raw)

    @property
    def lower_raw(self) -> tuple[float, float, float]:
        return tuple(
            center - half_extent
            for center, half_extent in zip(
                self.center_raw,
                self.half_extent_raw,
                strict=True,
            )
        )

    @property
    def upper_raw(self) -> tuple[float, float, float]:
        return tuple(
            center + half_extent
            for center, half_extent in zip(
                self.center_raw,
                self.half_extent_raw,
                strict=True,
            )
        )


_BIN_PROXY_BOX_SPECS = (
    _BinCollisionBoxSpec(
        name="bottom",
        center_raw=(0.0, 0.0, 0.5),
        half_extent_raw=(30.0, 20.0, 0.5),
    ),
    _BinCollisionBoxSpec(
        name="x_negative_wall",
        center_raw=(-24.5, 0.0, 8.0),
        half_extent_raw=(5.5, 20.0, 7.0),
    ),
    _BinCollisionBoxSpec(
        name="x_positive_wall",
        center_raw=(24.5, 0.0, 8.0),
        half_extent_raw=(5.5, 20.0, 7.0),
    ),
    _BinCollisionBoxSpec(
        name="y_negative_wall",
        center_raw=(0.0, -16.75, 8.0),
        half_extent_raw=(19.0, 3.25, 7.0),
    ),
    _BinCollisionBoxSpec(
        name="y_positive_wall",
        center_raw=(0.0, 16.75, 8.0),
        half_extent_raw=(19.0, 3.25, 7.0),
    ),
)


def _require_proxy_cavity_partition() -> None:
    """Prove the five boxes exactly bound the certified open cavity."""
    specs = {spec.name: spec for spec in _BIN_PROXY_BOX_SPECS}
    if set(specs) != {
        "bottom",
        "x_negative_wall",
        "x_positive_wall",
        "y_negative_wall",
        "y_positive_wall",
    }:
        raise RuntimeError("CAP grocery bin proxy roster drifted")

    # The analytic partition uses clean authored coordinates. The source mesh's
    # float32 bounds differ by only micrometres in raw units and are certified
    # separately; do not silently fold those mesh-rounding artifacts into the
    # physical cavity.
    outer_lower = _BIN_PROXY_OUTER_LOWER_RAW
    outer_upper = _BIN_PROXY_OUTER_UPPER_RAW
    cavity_lower = _CERTIFIED_CAVITY_LOWER_RAW
    cavity_upper = _CERTIFIED_CAVITY_UPPER_RAW
    expected_bounds = {
        "bottom": (
            outer_lower,
            (outer_upper[0], outer_upper[1], cavity_lower[2]),
        ),
        "x_negative_wall": (
            (outer_lower[0], outer_lower[1], cavity_lower[2]),
            (cavity_lower[0], outer_upper[1], cavity_upper[2]),
        ),
        "x_positive_wall": (
            (cavity_upper[0], outer_lower[1], cavity_lower[2]),
            (outer_upper[0], outer_upper[1], cavity_upper[2]),
        ),
        "y_negative_wall": (
            (cavity_lower[0], outer_lower[1], cavity_lower[2]),
            (cavity_upper[0], cavity_lower[1], cavity_upper[2]),
        ),
        "y_positive_wall": (
            (cavity_lower[0], cavity_upper[1], cavity_lower[2]),
            (cavity_upper[0], outer_upper[1], cavity_upper[2]),
        ),
    }
    for name, (expected_lower, expected_upper) in expected_bounds.items():
        spec = specs[name]
        _require_close_tuple(
            spec.lower_raw,
            expected_lower,
            absolute_tolerance=1e-12,
            label=f"CAP grocery bin {name} lower bound",
        )
        _require_close_tuple(
            spec.upper_raw,
            expected_upper,
            absolute_tolerance=1e-12,
            label=f"CAP grocery bin {name} upper bound",
        )

    def volume(
        lower: tuple[float, float, float],
        upper: tuple[float, float, float],
    ) -> float:
        return math.prod(
            upper_value - lower_value
            for lower_value, upper_value in zip(lower, upper, strict=True)
        )

    proxy_volume = sum(
        volume(spec.lower_raw, spec.upper_raw) for spec in _BIN_PROXY_BOX_SPECS
    )
    expected_proxy_volume = volume(outer_lower, outer_upper) - volume(
        cavity_lower,
        cavity_upper,
    )
    if not math.isclose(
        proxy_volume,
        expected_proxy_volume,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError(
            "CAP grocery bin proxy volume does not partition the outer prism "
            "around its certified cavity"
        )


def _require_bin_source_identity(source_usd_path: str) -> None:
    normalized_path = source_usd_path.replace("\\", "/")
    if not normalized_path.endswith(_EXPECTED_BIN_SOURCE_URI_SUFFIX):
        raise RuntimeError(
            "CAP grocery bin collision override requires the pinned grey-bin USD; "
            f"got {source_usd_path!r}"
        )


def _require_default_prim(stage: Any, *, label: str) -> Any:
    default_prim = stage.GetDefaultPrim()
    if (
        not default_prim.IsValid()
        or str(default_prim.GetPath()) != _BIN_DEFAULT_PRIM_PATH
    ):
        raise RuntimeError(f"{label} default prim must be {_BIN_DEFAULT_PRIM_PATH}")
    return default_prim


def _require_close_tuple(
    actual: tuple[float, float, float],
    expected: tuple[float, float, float],
    *,
    absolute_tolerance: float,
    label: str,
) -> None:
    if any(
        not math.isclose(
            actual_value,
            expected_value,
            rel_tol=0.0,
            abs_tol=absolute_tolerance,
        )
        for actual_value, expected_value in zip(
            actual,
            expected,
            strict=True,
        )
    ):
        raise RuntimeError(f"{label} drifted: expected {expected!r}, got {actual!r}")


def _require_root_contract(
    stage: Any,
    *,
    prim_root: str,
    label: str,
) -> None:
    from pxr import UsdPhysics

    root = stage.GetPrimAtPath(prim_root)
    if (
        not root.IsValid()
        or root.GetTypeName() != "Xform"
        or not root.HasAPI(UsdPhysics.RigidBodyAPI)
        or UsdPhysics.RigidBodyAPI(root).GetRigidBodyEnabledAttr().Get() is not True
        or UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get() is not False
    ):
        raise RuntimeError(f"{label} rigid root contract drifted")

    scale_attribute = root.GetAttribute("xformOp:scale")
    scale = scale_attribute.Get() if scale_attribute.IsValid() else None
    if scale is None:
        raise RuntimeError(f"{label} root scale is missing")
    _require_close_tuple(
        tuple(float(value) for value in scale),
        _BIN_ROOT_SCALE,
        absolute_tolerance=1e-12,
        label=f"{label} root scale",
    )

    material_path = f"{prim_root}/{_BIN_MATERIAL_SUBPATH}"
    material = stage.GetPrimAtPath(material_path)
    material_targets = tuple(
        str(path) for path in root.GetRelationship("material:binding").GetTargets()
    )
    if (
        not material.IsValid()
        or material.GetTypeName() != "Material"
        or material_targets != (material_path,)
    ):
        raise RuntimeError(f"{label} visual material binding drifted")


def _require_source_collider(
    stage: Any,
    *,
    prim_root: str,
    enabled: bool,
    label: str,
) -> Any:
    from pxr import UsdGeom, UsdPhysics

    collision_path = f"{prim_root}/{_BIN_SOURCE_COLLISION_SUBPATH}"
    collision = stage.GetPrimAtPath(collision_path)
    if (
        not collision.IsValid()
        or collision.GetTypeName() != "Mesh"
        or not collision.HasAPI(UsdPhysics.CollisionAPI)
        or not collision.HasAPI(UsdPhysics.MeshCollisionAPI)
        or UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
        is not enabled
        or UsdPhysics.MeshCollisionAPI(collision).GetApproximationAttr().Get()
        != "convexDecomposition"
        or UsdGeom.Imageable(collision).ComputeVisibility() == UsdGeom.Tokens.invisible
    ):
        raise RuntimeError(f"{label} source collision/visual contract drifted")
    xform = UsdGeom.Xformable(collision)
    operations = xform.GetOrderedXformOps()
    operation_names = tuple(operation.GetOpName() for operation in operations)
    if (
        xform.GetResetXformStack()
        or operation_names != ("xformOp:translate",)
        or tuple(float(value) for value in operations[0].Get()) != (0.0, 0.0, 0.0)
    ):
        raise RuntimeError(f"{label} source collider transform drifted")
    return collision


def _triangles_from_mesh(mesh: Any) -> list[tuple[Any, Any, Any]]:
    points = mesh.GetPointsAttr().Get()
    face_counts = mesh.GetFaceVertexCountsAttr().Get()
    face_indices = mesh.GetFaceVertexIndicesAttr().Get()
    if points is None or not points:
        raise RuntimeError("CAP grocery bin source mesh has no points")
    if face_counts is None or not face_counts:
        raise RuntimeError("CAP grocery bin source mesh has no faces")
    if face_indices is None or sum(int(count) for count in face_counts) != len(
        face_indices
    ):
        raise RuntimeError("CAP grocery bin source mesh face topology is inconsistent")

    point_count = len(points)
    triangles: list[tuple[Any, Any, Any]] = []
    offset = 0
    for count_value in face_counts:
        count = int(count_value)
        if count < 3:
            raise RuntimeError("CAP grocery bin source mesh contains a degenerate face")
        face = tuple(int(index) for index in face_indices[offset : offset + count])
        offset += count
        if any(index < 0 or index >= point_count for index in face):
            raise RuntimeError(
                "CAP grocery bin source mesh contains an invalid point index"
            )
        for vertex_index in range(1, count - 1):
            triangle = (
                points[face[0]],
                points[face[vertex_index]],
                points[face[vertex_index + 1]],
            )
            if any(
                not math.isfinite(float(coordinate))
                for point in triangle
                for coordinate in point
            ):
                raise RuntimeError(
                    "CAP grocery bin source mesh contains non-finite geometry"
                )
            triangles.append(triangle)
    if not triangles:
        raise RuntimeError("CAP grocery bin source mesh has no triangles")
    return triangles


def _triangle_aabb_separation_raw(
    triangle: tuple[Any, Any, Any],
) -> float:
    triangle_lower = tuple(
        min(float(point[axis]) for point in triangle) for axis in range(3)
    )
    triangle_upper = tuple(
        max(float(point[axis]) for point in triangle) for axis in range(3)
    )
    component_gaps = tuple(
        max(
            cavity_lower - triangle_upper_value,
            triangle_lower_value - cavity_upper,
            0.0,
        )
        for triangle_lower_value, triangle_upper_value, cavity_lower, cavity_upper in zip(
            triangle_lower,
            triangle_upper,
            _CERTIFIED_CAVITY_LOWER_RAW,
            _CERTIFIED_CAVITY_UPPER_RAW,
            strict=True,
        )
    )
    return math.sqrt(sum(gap * gap for gap in component_gaps))


def _require_source_cavity_certificate(mesh: Any) -> float:
    from pxr import UsdGeom

    points = mesh.GetPointsAttr().Get()
    if points is None or not points:
        raise RuntimeError("CAP grocery bin source mesh has no points")
    source_lower = tuple(
        min(float(point[axis]) for point in points) for axis in range(3)
    )
    source_upper = tuple(
        max(float(point[axis]) for point in points) for axis in range(3)
    )
    _require_close_tuple(
        source_lower,
        _BIN_SOURCE_LOWER_RAW,
        absolute_tolerance=_BIN_SOURCE_BOUNDS_TOLERANCE_RAW,
        label="CAP grocery bin source lower bound",
    )
    _require_close_tuple(
        source_upper,
        _BIN_SOURCE_UPPER_RAW,
        absolute_tolerance=_BIN_SOURCE_BOUNDS_TOLERANCE_RAW,
        label="CAP grocery bin source upper bound",
    )

    authored_extent = UsdGeom.Mesh(mesh.GetPrim()).GetExtentAttr().Get()
    if authored_extent is None or len(authored_extent) != 2:
        raise RuntimeError("CAP grocery bin source mesh extent is missing")
    _require_close_tuple(
        tuple(float(value) for value in authored_extent[0]),
        _BIN_SOURCE_LOWER_RAW,
        absolute_tolerance=_BIN_SOURCE_BOUNDS_TOLERANCE_RAW,
        label="CAP grocery bin authored lower extent",
    )
    _require_close_tuple(
        tuple(float(value) for value in authored_extent[1]),
        _BIN_SOURCE_UPPER_RAW,
        absolute_tolerance=_BIN_SOURCE_BOUNDS_TOLERANCE_RAW,
        label="CAP grocery bin authored upper extent",
    )

    minimum_gap = min(
        _triangle_aabb_separation_raw(triangle)
        for triangle in _triangles_from_mesh(mesh)
    )
    if minimum_gap <= 0.0:
        raise RuntimeError(
            "CAP grocery bin source triangles overlap the certified cavity"
        )
    if minimum_gap + 1e-12 < _CERTIFIED_MIN_TRIANGLE_AABB_GAP_RAW:
        raise RuntimeError(
            "CAP grocery bin source cavity clearance drifted below its "
            f"{_CERTIFIED_MIN_TRIANGLE_AABB_GAP_RAW!r}-raw-unit certificate: "
            f"{minimum_gap!r}"
        )
    return minimum_gap


def _open_and_validate_source(source_usd_path: str) -> Any:
    from pxr import Usd, UsdGeom

    _require_proxy_cavity_partition()
    _require_bin_source_identity(source_usd_path)
    try:
        stage = Usd.Stage.Open(source_usd_path)
    except Exception as exc:
        raise RuntimeError(
            f"failed to open CAP grocery bin USD {source_usd_path!r}"
        ) from exc
    if stage is None:
        raise RuntimeError(f"failed to open CAP grocery bin USD {source_usd_path!r}")
    _require_default_prim(stage, label="CAP grocery bin source USD")
    _require_root_contract(
        stage,
        prim_root=_BIN_DEFAULT_PRIM_PATH,
        label="CAP grocery bin source USD",
    )
    collision = _require_source_collider(
        stage,
        prim_root=_BIN_DEFAULT_PRIM_PATH,
        enabled=True,
        label="CAP grocery bin source USD",
    )
    _require_source_cavity_certificate(UsdGeom.Mesh(collision))
    return stage


def _author_physx_offsets(prim: Any) -> None:
    from pxr import Sdf

    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute(
        "physxCollision:contactOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(_BIN_PROXY_CONTACT_OFFSET_M)
    prim.CreateAttribute(
        "physxCollision:restOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(_BIN_PROXY_REST_OFFSET_M)


def _has_applied_schema(prim: Any, schema_name: str) -> bool:
    schemas = prim.GetMetadata("apiSchemas")
    return schemas is not None and schema_name in tuple(schemas.GetAppliedItems())


def _author_proxy_box(
    stage: Any,
    *,
    prim_root: str,
    spec: _BinCollisionBoxSpec,
) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    proxy = UsdGeom.Cube.Define(
        stage,
        f"{prim_root}/{spec.subpath}",
    )
    proxy.CreateSizeAttr(1.0)
    proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    xform = UsdGeom.Xformable(proxy)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*spec.center_raw)
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*spec.size_raw)
    )
    prim = proxy.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    _author_physx_offsets(prim)


def _require_proxy_boxes(
    stage: Any,
    *,
    prim_root: str,
    label: str,
) -> None:
    from pxr import UsdGeom, UsdPhysics

    root = stage.GetPrimAtPath(prim_root)
    expected_paths = {f"{prim_root}/{spec.subpath}" for spec in _BIN_PROXY_BOX_SPECS}
    enabled_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(root.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    }
    if enabled_collision_paths != expected_paths:
        raise RuntimeError(
            f"{label} enabled collision set drifted: "
            f"expected={sorted(expected_paths)!r}, "
            f"actual={sorted(enabled_collision_paths)!r}"
        )

    errors: list[str] = []
    for spec in _BIN_PROXY_BOX_SPECS:
        proxy_path = f"{prim_root}/{spec.subpath}"
        prim = stage.GetPrimAtPath(proxy_path)
        if not prim.IsValid() or prim.GetTypeName() != "Cube":
            errors.append(f"{proxy_path}=missing_or_not_Cube")
            continue
        if (
            not prim.HasAPI(UsdPhysics.CollisionAPI)
            or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True
        ):
            errors.append(f"{proxy_path}=collision_disabled")
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            errors.append(f"{proxy_path}=unexpected_PhysicsMeshCollisionAPI")
        if not _has_applied_schema(prim, "PhysxCollisionAPI"):
            errors.append(f"{proxy_path}=missing_PhysxCollisionAPI")
        if UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible:
            errors.append(f"{proxy_path}=visible")

        cube_size = UsdGeom.Cube(prim).GetSizeAttr().Get()
        xform = UsdGeom.Xformable(prim)
        operations = xform.GetOrderedXformOps()
        operation_names = tuple(operation.GetOpName() for operation in operations)
        if xform.GetResetXformStack():
            errors.append(f"{proxy_path}=reset_xform_stack")
        if operation_names != ("xformOp:translate", "xformOp:scale"):
            errors.append(f"{proxy_path}=xform_order({operation_names!r})")
        else:
            actual_center = tuple(float(value) for value in operations[0].Get())
            actual_size = tuple(float(value) for value in operations[1].Get())
            try:
                _require_close_tuple(
                    actual_center,
                    spec.center_raw,
                    absolute_tolerance=1e-12,
                    label=f"{proxy_path} center",
                )
                _require_close_tuple(
                    actual_size,
                    spec.size_raw,
                    absolute_tolerance=1e-12,
                    label=f"{proxy_path} size",
                )
            except RuntimeError as exc:
                errors.append(str(exc))
        if not isinstance(cube_size, float) or not math.isclose(
            cube_size,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            errors.append(f"{proxy_path}=cube_size({cube_size!r})")

        contact_offset = prim.GetAttribute("physxCollision:contactOffset").Get()
        rest_offset = prim.GetAttribute("physxCollision:restOffset").Get()
        if not (
            isinstance(contact_offset, float)
            and math.isclose(
                contact_offset,
                _BIN_PROXY_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and isinstance(rest_offset, float)
            and math.isclose(
                rest_offset,
                _BIN_PROXY_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append(f"{proxy_path}=offsets({contact_offset!r},{rest_offset!r})")
    if errors:
        raise RuntimeError(f"{label} analytic box contract drifted: {errors}")


def _write_bin_override(
    source_usd_path: str,
    override_usd_path: Path,
) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateNew(str(override_usd_path))
    if stage is None:
        raise RuntimeError(f"failed to create grocery bin override {override_usd_path}")
    root = stage.DefinePrim(_BIN_DEFAULT_PRIM_PATH)
    root.GetReferences().AddReference(assetPath=source_usd_path)
    stage.SetDefaultPrim(root)
    source_collision = stage.OverridePrim(
        f"{_BIN_DEFAULT_PRIM_PATH}/{_BIN_SOURCE_COLLISION_SUBPATH}"
    )
    UsdPhysics.CollisionAPI(source_collision).CreateCollisionEnabledAttr(False).Set(
        False
    )
    for spec in _BIN_PROXY_BOX_SPECS:
        _author_proxy_box(
            stage,
            prim_root=_BIN_DEFAULT_PRIM_PATH,
            spec=spec,
        )
    stage.GetRootLayer().Save()


def _validate_override_layer(
    override_usd_path: Path,
    *,
    source_usd_path: str,
) -> None:
    from pxr import Usd

    try:
        stage = Usd.Stage.Open(str(override_usd_path))
    except Exception as exc:
        raise RuntimeError(
            f"failed to reopen CAP grocery bin override {override_usd_path}"
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"failed to reopen CAP grocery bin override {override_usd_path}"
        )
    root = _require_default_prim(
        stage,
        label="CAP grocery bin collision override",
    )
    root_spec = stage.GetRootLayer().GetPrimAtPath(root.GetPath())
    if root_spec is None or not root_spec.HasInfo("references"):
        raise RuntimeError(
            "CAP grocery bin override root layer has no source reference"
        )
    references = tuple(root_spec.GetInfo("references").GetAppliedItems())
    if len(references) != 1 or references[0].assetPath != source_usd_path:
        raise RuntimeError(
            "CAP grocery bin override must contain exactly one reference to "
            f"{source_usd_path!r}; got {references!r}"
        )
    _require_root_contract(
        stage,
        prim_root=_BIN_DEFAULT_PRIM_PATH,
        label="CAP grocery bin collision override",
    )
    _require_source_collider(
        stage,
        prim_root=_BIN_DEFAULT_PRIM_PATH,
        enabled=False,
        label="CAP grocery bin collision override",
    )
    _require_proxy_boxes(
        stage,
        prim_root=_BIN_DEFAULT_PRIM_PATH,
        label="CAP grocery bin collision override",
    )


class GroceryBinCollisionOverride:
    """Own one verified temporary grey-bin analytic collision layer."""

    def __init__(self, source_usd_path: str):
        if not isinstance(source_usd_path, str) or not source_usd_path:
            raise RuntimeError("CAP grocery bin is missing its source USD path")
        _open_and_validate_source(source_usd_path)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="cap_grocery_bin_collision_"
        )
        self.source_usd_path = source_usd_path
        self.override_usd_path = str(
            Path(self._temporary_directory.name)
            / "grey_bin.cap_collision_override.usda"
        )
        try:
            _write_bin_override(
                source_usd_path,
                Path(self.override_usd_path),
            )
            _validate_override_layer(
                Path(self.override_usd_path),
                source_usd_path=source_usd_path,
            )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Remove the temporary layer after the environment has closed."""
        temporary_directory = self._temporary_directory
        if temporary_directory is None:
            return
        self._temporary_directory = None
        temporary_directory.cleanup()


def apply_grocery_bin_collision_override(
    grocery_bin: Any,
) -> GroceryBinCollisionOverride:
    """Replace one bin collider without mutating a shared spawn config."""
    try:
        source_usd_path = grocery_bin.usd_path
        object_config = copy.deepcopy(grocery_bin.object_cfg)
        spawn = object_config.spawn
    except AttributeError as exc:
        raise RuntimeError("CAP grocery bin asset has no mutable USD spawn") from exc
    if spawn.usd_path != source_usd_path:
        raise RuntimeError("CAP grocery bin asset source and spawn paths disagree")

    override = GroceryBinCollisionOverride(source_usd_path)
    try:
        spawn.usd_path = override.override_usd_path
        grocery_bin.usd_path = override.override_usd_path
        grocery_bin.object_cfg = object_config
        grocery_bin.bounding_box = None
    except BaseException:
        override.close()
        raise
    return override


def validate_live_grocery_bin_collision_contract(
    stage: Any,
    *,
    bin_prim_path: str,
) -> tuple[int, int]:
    """Validate the composed live bin and return original/proxy counts."""
    _require_root_contract(
        stage,
        prim_root=bin_prim_path,
        label="live CAP grocery bin",
    )
    _require_source_collider(
        stage,
        prim_root=bin_prim_path,
        enabled=False,
        label="live CAP grocery bin",
    )
    _require_proxy_boxes(
        stage,
        prim_root=bin_prim_path,
        label="live CAP grocery bin",
    )
    return 1, len(_BIN_PROXY_BOX_SPECS)
