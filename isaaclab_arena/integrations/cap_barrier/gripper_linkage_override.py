# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pre-parse collision correction for the CAP grocery DROID gripper."""

from __future__ import annotations

import copy
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_EXPECTED_SOURCE_URI_SUFFIX = (
    "/Arena/assets/robot_library/droid/franka_robotiq_2f_85_flattened.usd"
)
_ROBOT_DEFAULT_PRIM_PATH = "/panda"
_LINKAGE_COLLISION_SUBPATHS = (
    "Gripper/Robotiq_2F_85/left_outer_knuckle/Defeatured_2F_85_PAD_OPEN_Finger1step_01",
    "Gripper/Robotiq_2F_85/left_outer_finger/Defeatured_2F_85_PAD_OPEN_finger2step_01",
    "Gripper/Robotiq_2F_85/left_inner_knuckle/Defeatured_2F_85_PAD_OPEN_finger3step_01",
    "Gripper/Robotiq_2F_85/right_outer_knuckle/Defeatured_2F_85_PAD_OPEN_Finger1step_01",
    "Gripper/Robotiq_2F_85/right_outer_finger/Defeatured_2F_85_PAD_OPEN_finger2step_01",
    "Gripper/Robotiq_2F_85/right_inner_knuckle/Defeatured_2F_85_PAD_OPEN_finger3step_01",
)
_RETAINED_COLLISION_SUBPATHS = (
    "Gripper/Robotiq_2F_85/base_link/Defeatured_2F_85_PAD_OPEN_basestep_01",
    "Gripper/Robotiq_2F_85/left_inner_finger/Defeatured_2F_85_PAD_OPEN_finger4step_01",
    "Gripper/Robotiq_2F_85/left_inner_finger/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01",
    "Gripper/Robotiq_2F_85/right_inner_finger/Defeatured_2F_85_PAD_OPEN_finger4step_01",
    "Gripper/Robotiq_2F_85/right_inner_finger/Defeatured_2F_85_PAD_OPEN_fingertipsstep_01",
)
_ALL_ORIGINAL_COLLISION_SUBPATHS = (
    *_LINKAGE_COLLISION_SUBPATHS,
    *_RETAINED_COLLISION_SUBPATHS,
)
_PROXY_CONTACT_OFFSET_M = 0.00025
_PROXY_REST_OFFSET_M = 0.0
_MAX_PROXY_PADDING_M = 1e-7
_METRIC_TRANSFORM_TOLERANCE = 1e-9


@dataclass(frozen=True)
class _CollisionBoxSpec:
    body_subpath: str
    name: str
    source_collision_subpath: str
    lower_m: tuple[float, float, float]
    upper_m: tuple[float, float, float]

    @property
    def proxy_subpath(self) -> str:
        return f"{self.body_subpath}/{self.name}"

    @property
    def center_m(self) -> tuple[float, float, float]:
        return tuple(
            (lower + upper) * 0.5
            for lower, upper in zip(self.lower_m, self.upper_m, strict=True)
        )

    @property
    def size_m(self) -> tuple[float, float, float]:
        return tuple(
            upper - lower
            for lower, upper in zip(self.lower_m, self.upper_m, strict=True)
        )


# These bounds are directed-outward float32 envelopes of the five source collision
# meshes in their owning rigid-body frames. The generated cubes are conservative
# analytic collision proxies; they are not claimed to be shape-equivalent.
_PROXY_BOX_SPECS = (
    _CollisionBoxSpec(
        body_subpath="Gripper/Robotiq_2F_85/base_link",
        name="cap_collision_proxy_palm",
        source_collision_subpath=_RETAINED_COLLISION_SUBPATHS[0],
        lower_m=(
            -0.003598434152081609,
            -0.04244335740804672,
            -0.03750000149011612,
        ),
        upper_m=(
            0.08975811302661896,
            0.042443349957466125,
            0.03750000521540642,
        ),
    ),
    _CollisionBoxSpec(
        body_subpath="Gripper/Robotiq_2F_85/left_inner_finger",
        name="cap_collision_proxy_finger4",
        source_collision_subpath=_RETAINED_COLLISION_SUBPATHS[1],
        lower_m=(
            0.09211990237236023,
            0.04385676607489586,
            -0.013500000350177288,
        ),
        upper_m=(
            0.12010053545236588,
            0.07375796139240265,
            0.013500001281499863,
        ),
    ),
    _CollisionBoxSpec(
        body_subpath="Gripper/Robotiq_2F_85/left_inner_finger",
        name="cap_collision_proxy_fingertip",
        source_collision_subpath=_RETAINED_COLLISION_SUBPATHS[2],
        lower_m=(
            0.1110997200012207,
            0.04249757155776024,
            -0.010999999940395355,
        ),
        upper_m=(
            0.1490999311208725,
            0.06165679171681404,
            0.01100000087171793,
        ),
    ),
    _CollisionBoxSpec(
        body_subpath="Gripper/Robotiq_2F_85/right_inner_finger",
        name="cap_collision_proxy_finger4",
        source_collision_subpath=_RETAINED_COLLISION_SUBPATHS[3],
        lower_m=(
            0.09211990237236023,
            -0.07375796139240265,
            -0.013500001281499863,
        ),
        upper_m=(
            0.12010053545236588,
            -0.04385676607489586,
            0.013500000350177288,
        ),
    ),
    _CollisionBoxSpec(
        body_subpath="Gripper/Robotiq_2F_85/right_inner_finger",
        name="cap_collision_proxy_fingertip",
        source_collision_subpath=_RETAINED_COLLISION_SUBPATHS[4],
        lower_m=(
            0.1110997200012207,
            -0.06165679171681404,
            -0.01100000087171793,
        ),
        upper_m=(
            0.1490999311208725,
            -0.04249757155776024,
            0.010999999940395355,
        ),
    ),
)
_PROXY_BODY_SUBPATHS = tuple(
    dict.fromkeys(spec.body_subpath for spec in _PROXY_BOX_SPECS)
)


def _require_source_identity(source_usd_path: str) -> None:
    normalized_path = source_usd_path.replace("\\", "/")
    if not normalized_path.endswith(_EXPECTED_SOURCE_URI_SUFFIX):
        raise RuntimeError(
            "CAP grocery linkage override requires the pinned DROID robot USD; "
            f"got {source_usd_path!r}"
        )


def _require_default_prim(stage: Any, *, label: str) -> Any:
    default_prim = stage.GetDefaultPrim()
    if not default_prim.IsValid():
        raise RuntimeError(f"{label} has no valid default prim")
    actual_path = str(default_prim.GetPath())
    if actual_path != _ROBOT_DEFAULT_PRIM_PATH:
        raise RuntimeError(
            f"{label} default prim must be {_ROBOT_DEFAULT_PRIM_PATH}, got {actual_path}"
        )
    return default_prim


def _require_unit_metric_transforms(
    stage: Any,
    *,
    prim_paths: tuple[str, ...],
    label: str,
) -> None:
    """Reject inherited scale or shear that the pose-only guard cannot represent."""
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    errors: list[str] = []
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsDefined():
            errors.append(f"{prim_path}=missing")
            continue
        matrix = cache.GetLocalToWorldTransform(prim)
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
        cross_xy = (
            axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1],
            axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2],
            axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0],
        )
        handed_volume = sum(
            component * axis_component
            for component, axis_component in zip(cross_xy, axes[2], strict=True)
        )
        pairwise_dots = (
            sum(a * b for a, b in zip(axes[0], axes[1], strict=True)),
            sum(a * b for a, b in zip(axes[0], axes[2], strict=True)),
            sum(a * b for a, b in zip(axes[1], axes[2], strict=True)),
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
            errors.append(f"{prim_path}=axes{axes!r}, handed_volume={handed_volume!r}")
    if errors:
        raise RuntimeError(f"{label} metric transform drifted: {errors}")


def _require_collision_state(
    stage: Any,
    *,
    prim_root: str,
    subpaths: tuple[str, ...],
    enabled: bool,
    label: str,
) -> None:
    from pxr import UsdPhysics

    missing: list[str] = []
    missing_api: list[str] = []
    wrong_state: list[str] = []
    for subpath in subpaths:
        prim_path = f"{prim_root}/{subpath}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or not prim.IsDefined():
            missing.append(prim_path)
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            missing_api.append(prim_path)
            continue
        actual_enabled = UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        if actual_enabled is not enabled:
            wrong_state.append(f"{prim_path}={actual_enabled!r}")
    if missing or missing_api or wrong_state:
        raise RuntimeError(
            f"{label} linkage collision contract mismatch: "
            f"missing={missing}, missing_PhysicsCollisionAPI={missing_api}, "
            f"wrong_collisionEnabled={wrong_state}"
        )


def _require_linkage_collision_state(
    stage: Any,
    *,
    prim_root: str = _ROBOT_DEFAULT_PRIM_PATH,
    enabled: bool,
    label: str,
) -> None:
    _require_collision_state(
        stage,
        prim_root=prim_root,
        subpaths=_LINKAGE_COLLISION_SUBPATHS,
        enabled=enabled,
        label=label,
    )


def _require_retained_collision_state(
    stage: Any,
    *,
    prim_root: str = _ROBOT_DEFAULT_PRIM_PATH,
    label: str,
) -> None:
    _require_collision_state(
        stage,
        prim_root=prim_root,
        subpaths=_RETAINED_COLLISION_SUBPATHS,
        enabled=True,
        label=label,
    )


def _require_source_proxy_envelopes(stage: Any) -> None:
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=False,
    )
    errors: list[str] = []
    for spec in _PROXY_BOX_SPECS:
        source_path = f"{_ROBOT_DEFAULT_PRIM_PATH}/{spec.source_collision_subpath}"
        source_prim = stage.GetPrimAtPath(source_path)
        if not source_prim.IsValid():
            errors.append(f"{source_path}=missing")
            continue
        source_box = cache.ComputeLocalBound(source_prim).ComputeAlignedBox()
        source_lower = tuple(float(value) for value in source_box.GetMin())
        source_upper = tuple(float(value) for value in source_box.GetMax())
        if any(
            source < proxy
            for source, proxy in zip(source_lower, spec.lower_m, strict=True)
        ) or any(
            source > proxy
            for source, proxy in zip(source_upper, spec.upper_m, strict=True)
        ):
            errors.append(
                f"{source_path}=outside proxy {source_lower!r}..{source_upper!r}"
            )
            continue
        maximum_padding = max(
            *(
                source - proxy
                for source, proxy in zip(source_lower, spec.lower_m, strict=True)
            ),
            *(
                proxy - source
                for source, proxy in zip(source_upper, spec.upper_m, strict=True)
            ),
        )
        if maximum_padding > _MAX_PROXY_PADDING_M:
            errors.append(f"{source_path}=unexpected proxy padding {maximum_padding!r}")
    if errors:
        raise RuntimeError(f"CAP grocery source proxy envelope mismatch: {errors}")


def _author_physx_offsets(prim: Any) -> None:
    from pxr import Sdf

    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute(
        "physxCollision:contactOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(_PROXY_CONTACT_OFFSET_M)
    prim.CreateAttribute(
        "physxCollision:restOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(_PROXY_REST_OFFSET_M)


def _has_applied_schema(prim: Any, schema_name: str) -> bool:
    schemas = prim.GetMetadata("apiSchemas")
    return schemas is not None and schema_name in tuple(schemas.GetAppliedItems())


def _author_proxy_box(stage: Any, *, prim_root: str, spec: _CollisionBoxSpec) -> None:
    from pxr import Gf, UsdGeom, UsdPhysics

    proxy = UsdGeom.Cube.Define(stage, f"{prim_root}/{spec.proxy_subpath}")
    proxy.CreateSizeAttr(1.0)
    proxy.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    xform = UsdGeom.Xformable(proxy)
    xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*spec.center_m)
    )
    xform.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*spec.size_m)
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
    from pxr import Usd, UsdGeom, UsdPhysics

    gripper_root = stage.GetPrimAtPath(f"{prim_root}/Gripper/Robotiq_2F_85")
    expected_collision_paths = {
        f"{prim_root}/{spec.proxy_subpath}" for spec in _PROXY_BOX_SPECS
    }
    enabled_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetPath().HasPrefix(gripper_root.GetPath())
        and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is True
    }
    if enabled_collision_paths != expected_collision_paths:
        raise RuntimeError(
            f"{label} enabled gripper collision set mismatch: "
            f"expected={sorted(expected_collision_paths)!r}, "
            f"actual={sorted(enabled_collision_paths)!r}"
        )

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        ["default", "render", "proxy"],
        useExtentsHint=False,
    )
    errors: list[str] = []
    for spec in _PROXY_BOX_SPECS:
        proxy_path = f"{prim_root}/{spec.proxy_subpath}"
        prim = stage.GetPrimAtPath(proxy_path)
        if not prim.IsValid() or prim.GetTypeName() != "Cube":
            errors.append(f"{proxy_path}=missing_or_not_Cube")
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            errors.append(f"{proxy_path}=missing_PhysicsCollisionAPI")
            continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True:
            errors.append(f"{proxy_path}=collision_disabled")
        if prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            errors.append(f"{proxy_path}=unexpected_PhysicsMeshCollisionAPI")
        if not _has_applied_schema(prim, "PhysxCollisionAPI"):
            errors.append(f"{proxy_path}=missing_PhysxCollisionAPI")
        contact_offset = prim.GetAttribute("physxCollision:contactOffset").Get()
        rest_offset = prim.GetAttribute("physxCollision:restOffset").Get()
        if not (
            isinstance(contact_offset, float)
            and math.isclose(
                contact_offset,
                _PROXY_CONTACT_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            and isinstance(rest_offset, float)
            and math.isclose(
                rest_offset,
                _PROXY_REST_OFFSET_M,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            errors.append(f"{proxy_path}=offsets({contact_offset!r},{rest_offset!r})")
        imageable = UsdGeom.Imageable(prim)
        if imageable.GetVisibilityAttr().Get() != UsdGeom.Tokens.invisible:
            errors.append(f"{proxy_path}=visible")
        box = cache.ComputeLocalBound(prim).ComputeAlignedBox()
        actual_lower = tuple(float(value) for value in box.GetMin())
        actual_upper = tuple(float(value) for value in box.GetMax())
        if any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                actual_lower,
                spec.lower_m,
                strict=True,
            )
        ) or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
            for actual, expected in zip(
                actual_upper,
                spec.upper_m,
                strict=True,
            )
        ):
            errors.append(f"{proxy_path}=bounds({actual_lower!r},{actual_upper!r})")
    if errors:
        raise RuntimeError(f"{label} collision proxy contract mismatch: {errors}")


def _open_and_validate_source(source_usd_path: str) -> Any:
    from pxr import Usd

    _require_source_identity(source_usd_path)
    try:
        stage = Usd.Stage.Open(source_usd_path)
    except Exception as exc:
        raise RuntimeError(
            f"failed to open CAP grocery DROID robot USD {source_usd_path!r}"
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"failed to open CAP grocery DROID robot USD {source_usd_path!r}"
        )
    _require_default_prim(stage, label="CAP grocery DROID source USD")
    _require_linkage_collision_state(
        stage,
        enabled=True,
        label="CAP grocery DROID source USD",
    )
    _require_retained_collision_state(
        stage,
        label="CAP grocery DROID source USD",
    )
    _require_source_proxy_envelopes(stage)
    _require_unit_metric_transforms(
        stage,
        prim_paths=(
            _ROBOT_DEFAULT_PRIM_PATH,
            *(
                f"{_ROBOT_DEFAULT_PRIM_PATH}/{subpath}"
                for subpath in _PROXY_BODY_SUBPATHS
            ),
        ),
        label="CAP grocery DROID source USD",
    )
    return stage


def _write_override_layer(source_usd_path: str, override_usd_path: Path) -> None:
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.CreateNew(str(override_usd_path))
    if stage is None:
        raise RuntimeError(
            f"failed to create linkage override layer {override_usd_path}"
        )
    robot_prim = stage.DefinePrim(_ROBOT_DEFAULT_PRIM_PATH)
    robot_prim.GetReferences().AddReference(assetPath=source_usd_path)
    stage.SetDefaultPrim(robot_prim)
    for subpath in _ALL_ORIGINAL_COLLISION_SUBPATHS:
        prim = stage.OverridePrim(f"{_ROBOT_DEFAULT_PRIM_PATH}/{subpath}")
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False).Set(False)
    for spec in _PROXY_BOX_SPECS:
        _author_proxy_box(
            stage,
            prim_root=_ROBOT_DEFAULT_PRIM_PATH,
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
            f"failed to reopen linkage override layer {override_usd_path}"
        ) from exc
    if stage is None:
        raise RuntimeError(
            f"failed to reopen linkage override layer {override_usd_path}"
        )
    default_prim = _require_default_prim(stage, label="CAP grocery linkage override")
    _require_unit_metric_transforms(
        stage,
        prim_paths=(
            _ROBOT_DEFAULT_PRIM_PATH,
            *(
                f"{_ROBOT_DEFAULT_PRIM_PATH}/{subpath}"
                for subpath in _PROXY_BODY_SUBPATHS
            ),
        ),
        label="CAP grocery linkage override",
    )
    default_prim_spec = stage.GetRootLayer().GetPrimAtPath(default_prim.GetPath())
    if default_prim_spec is None or not default_prim_spec.HasInfo("references"):
        raise RuntimeError(
            "CAP grocery linkage override root layer has no robot reference"
        )
    authored_references = tuple(
        default_prim_spec.GetInfo("references").GetAppliedItems()
    )
    if (
        len(authored_references) != 1
        or authored_references[0].assetPath != source_usd_path
    ):
        raise RuntimeError(
            "CAP grocery linkage override must contain exactly one reference to "
            f"{source_usd_path!r}; got {authored_references!r}"
        )
    _require_linkage_collision_state(
        stage,
        enabled=False,
        label="CAP grocery linkage override",
    )
    _require_collision_state(
        stage,
        prim_root=_ROBOT_DEFAULT_PRIM_PATH,
        subpaths=_RETAINED_COLLISION_SUBPATHS,
        enabled=False,
        label="CAP grocery linkage override",
    )
    _require_proxy_boxes(
        stage,
        prim_root=_ROBOT_DEFAULT_PRIM_PATH,
        label="CAP grocery linkage override",
    )


def validate_live_grocery_gripper_collision_contract(
    stage: Any,
    *,
    robot_prim_path: str,
) -> tuple[int, int]:
    """Prove the effective grocery robot uses only the analytic CAP proxies."""
    _require_unit_metric_transforms(
        stage,
        prim_paths=(
            robot_prim_path,
            *(f"{robot_prim_path}/{subpath}" for subpath in _PROXY_BODY_SUBPATHS),
        ),
        label="live CAP grocery DROID robot",
    )
    _require_linkage_collision_state(
        stage,
        prim_root=robot_prim_path,
        enabled=False,
        label="live CAP grocery DROID robot",
    )
    _require_collision_state(
        stage,
        prim_root=robot_prim_path,
        subpaths=_RETAINED_COLLISION_SUBPATHS,
        enabled=False,
        label="live CAP grocery DROID robot",
    )
    _require_proxy_boxes(
        stage,
        prim_root=robot_prim_path,
        label="live CAP grocery DROID robot",
    )
    return len(_ALL_ORIGINAL_COLLISION_SUBPATHS), len(_PROXY_BOX_SPECS)


class GripperLinkageCollisionOverride:
    """Own one verified temporary DROID gripper linkage override layer."""

    def __init__(self, source_usd_path: str):
        if not isinstance(source_usd_path, str) or not source_usd_path:
            raise RuntimeError("CAP grocery DROID robot spawn is missing its USD path")
        _open_and_validate_source(source_usd_path)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix="cap_grocery_gripper_linkage_"
        )
        self.source_usd_path = source_usd_path
        self.override_usd_path = str(
            Path(self._temporary_directory.name)
            / "franka_robotiq_2f_85_flattened.cap_grocery_override.usda"
        )
        try:
            _write_override_layer(source_usd_path, Path(self.override_usd_path))
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


def apply_grocery_gripper_linkage_override(
    scene_config: Any,
) -> GripperLinkageCollisionOverride:
    """Apply a verified linkage override to one grocery embodiment instance."""
    try:
        robot = copy.deepcopy(scene_config.robot)
        source_usd_path = robot.spawn.usd_path
    except AttributeError as exc:
        raise RuntimeError(
            "CAP grocery DROID scene is missing robot.spawn.usd_path"
        ) from exc

    override = GripperLinkageCollisionOverride(source_usd_path)
    try:
        robot.spawn.usd_path = override.override_usd_path
        scene_config.robot = robot
    except BaseException:
        override.close()
        raise
    return override
