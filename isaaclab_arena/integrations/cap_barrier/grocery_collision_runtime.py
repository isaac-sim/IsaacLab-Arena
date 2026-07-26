# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Runtime PhysX proof inputs for the CAP grocery close guard."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any

from .gripper_linkage_override import (
    _PROXY_CONTACT_OFFSET_M,
    _PROXY_REST_OFFSET_M,
)
from .grocery_bin_collision_override import (
    _BIN_PROXY_CONTACT_OFFSET_M,
    _BIN_PROXY_REST_OFFSET_M,
)
from .grocery_close_guard import (
    CollisionOffsets,
    GroceryCollisionOffsets,
    GroceryCloseAuthorizationError,
    Pose,
)
from .grocery_dynamics_certificate import GroceryDynamicsCertificate
from .grocery_object_collision_override import (
    _CAN_PROXY_CONTACT_OFFSET_M,
    _CAN_PROXY_REST_OFFSET_M,
)
from .grocery_scene_spec import (
    CAP_GROCERY_BIN_ASSET,
    CAP_GROCERY_GROUND_COLLISION_PRIM_PATH,
    CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
    CAP_GROCERY_GROUND_REST_OFFSET_M,
    CAP_GROCERY_OBJECT_ASSET,
    CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
    CAP_GROCERY_SUPPORT_INSTANCE,
    CAP_GROCERY_SUPPORT_POSE,
    CAP_GROCERY_SUPPORT_REST_OFFSET_M,
    CAP_GROCERY_SUPPORT_SIZE,
)

_MATERIAL_TOLERANCE = 1.0e-6
_TIMESTAMP_TOLERANCE_S = 1.0e-12


@dataclass(frozen=True)
class GroceryRuntimeGeometry:
    """One same-timestamp live geometry sample."""

    simulation_timestamp_s: float
    gripper_base_pose: Pose
    left_inner_finger_pose: Pose
    right_inner_finger_pose: Pose
    can_pose: Pose
    bin_pose: Pose
    support_pose: Pose
    collision_offsets: GroceryCollisionOffsets


@dataclass(frozen=True)
class _BodyViewSpec:
    label: str
    path: str
    shape_count: int
    material: tuple[float, float, float]
    contact_offset_m: float
    rest_offset_m: float


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise GroceryCloseAuthorizationError(
            f"{label} must be a real number, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise GroceryCloseAuthorizationError(f"{label} must be finite, got {value!r}")
    return result


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "torch"):
        value = value.torch
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _numeric_row(value: object, *, length: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)):
        raise GroceryCloseAuthorizationError(
            f"{label} must contain exactly {length} numbers"
        )
    try:
        row = tuple(
            _finite_float(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)  # type: ignore[arg-type]
        )
    except TypeError as exc:
        raise GroceryCloseAuthorizationError(
            f"{label} must contain exactly {length} numbers"
        ) from exc
    if len(row) != length:
        raise GroceryCloseAuthorizationError(
            f"{label} must contain exactly {length} numbers, got {len(row)}"
        )
    return row


def _numeric_matrix(
    value: Any,
    *,
    rows: int,
    columns: int,
    label: str,
) -> tuple[tuple[float, ...], ...]:
    value = _to_builtin(value)
    if isinstance(value, (str, bytes)):
        raise GroceryCloseAuthorizationError(
            f"{label} must have shape ({rows}, {columns})"
        )
    try:
        matrix = tuple(
            _numeric_row(row, length=columns, label=f"{label}[{index}]")
            for index, row in enumerate(value)
        )
    except TypeError as exc:
        raise GroceryCloseAuthorizationError(
            f"{label} must have shape ({rows}, {columns})"
        ) from exc
    if len(matrix) != rows:
        raise GroceryCloseAuthorizationError(
            f"{label} must have shape ({rows}, {columns}), got {len(matrix)} rows"
        )
    return matrix


def _positive_definite_inertia(
    values: tuple[float, ...],
    *,
    label: str,
) -> None:
    if len(values) != 9:
        raise GroceryCloseAuthorizationError(
            f"{label} must contain a 3x3 inertia matrix"
        )
    xx, xy, xz, yx, yy, yz, zx, zy, zz = values
    if not (
        math.isclose(xy, yx, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(xz, zx, rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(yz, zy, rel_tol=0.0, abs_tol=1.0e-6)
    ):
        raise GroceryCloseAuthorizationError(f"{label} is not symmetric")
    leading_two = xx * yy - xy * yx
    determinant = (
        xx * (yy * zz - yz * zy) - xy * (yx * zz - yz * zx) + xz * (yx * zy - yy * zx)
    )
    if xx <= 0.0 or leading_two <= 0.0 or determinant <= 0.0:
        raise GroceryCloseAuthorizationError(f"{label} is not positive definite")


def configure_grocery_ground_collision_contract(stage: Any) -> None:
    """Author the finite grocery ground offsets before PhysX construction."""
    from pxr import Sdf, UsdGeom, UsdPhysics

    ground = stage.GetPrimAtPath(CAP_GROCERY_GROUND_COLLISION_PRIM_PATH)
    if (
        not ground.IsValid()
        or ground.GetTypeName() != "Plane"
        or not ground.HasAPI(UsdPhysics.CollisionAPI)
        or UsdPhysics.CollisionAPI(ground).GetCollisionEnabledAttr().Get() is not True
        or UsdGeom.Plane(ground).GetAxisAttr().Get() != UsdGeom.Tokens.z
    ):
        raise GroceryCloseAuthorizationError(
            "composed grocery ground-plane contract drifted"
        )
    try:
        from pxr import PhysxSchema
    except ImportError:
        # CPU-only OpenUSD tests do not load Kit's PhysX schema registry.
        ground.AddAppliedSchema("PhysxCollisionAPI")
    else:
        PhysxSchema.PhysxCollisionAPI.Apply(ground)
    ground.CreateAttribute(
        "physxCollision:contactOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(CAP_GROCERY_GROUND_CONTACT_OFFSET_M)
    ground.CreateAttribute(
        "physxCollision:restOffset",
        Sdf.ValueTypeNames.Float,
    ).Set(CAP_GROCERY_GROUND_REST_OFFSET_M)


def validate_live_grocery_fixture_contract(
    stage: Any,
    *,
    support_prim_path: str,
) -> None:
    """Validate the pre-authored support and ground collision contract."""
    from pxr import Gf, UsdGeom, UsdPhysics

    if not isinstance(support_prim_path, str) or not support_prim_path.startswith("/"):
        raise GroceryCloseAuthorizationError(
            "support_prim_path must be an absolute USD prim path"
        )
    if not math.isclose(
        float(UsdGeom.GetStageMetersPerUnit(stage)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise GroceryCloseAuthorizationError(
            "CAP grocery fixture stage must use metersPerUnit=1"
        )

    support = stage.GetPrimAtPath(support_prim_path)
    if (
        not support.IsValid()
        or support.GetTypeName() != "Xform"
        or not support.HasAPI(UsdPhysics.RigidBodyAPI)
        or UsdPhysics.RigidBodyAPI(support).GetRigidBodyEnabledAttr().Get() is not True
        or UsdPhysics.RigidBodyAPI(support).GetKinematicEnabledAttr().Get() is not True
    ):
        raise GroceryCloseAuthorizationError(
            "composed grocery support rigid/collision contract drifted"
        )
    support_collision = stage.GetPrimAtPath(f"{support_prim_path}/geometry/mesh")
    if (
        not support_collision.IsValid()
        or support_collision.GetTypeName() != "Cube"
        or not support_collision.HasAPI(UsdPhysics.CollisionAPI)
        or UsdPhysics.CollisionAPI(support_collision).GetCollisionEnabledAttr().Get()
        is not True
    ):
        raise GroceryCloseAuthorizationError(
            "composed grocery support collider contract drifted"
        )

    support_translate = support.GetAttribute("xformOp:translate").Get()
    support_orientation = support.GetAttribute("xformOp:orient").Get()
    support_size = support_collision.GetAttribute("size").Get()
    support_scale = support_collision.GetAttribute("xformOp:scale").Get()
    support_xform_order = tuple(
        str(item) for item in support.GetAttribute("xformOpOrder").Get() or ()
    )
    collider_xform_order = tuple(
        str(item) for item in support_collision.GetAttribute("xformOpOrder").Get() or ()
    )
    actual_translation = (
        tuple(float(value) for value in support_translate)
        if support_translate is not None
        else ()
    )
    actual_orientation = (
        (
            *(float(value) for value in support_orientation.GetImaginary()),
            float(support_orientation.GetReal()),
        )
        if support_orientation is not None
        else ()
    )
    actual_scale = (
        tuple(float(value) for value in support_scale)
        if support_scale is not None
        else ()
    )
    expected_scale = tuple(
        value / min(CAP_GROCERY_SUPPORT_SIZE) for value in CAP_GROCERY_SUPPORT_SIZE
    )
    if not (
        support_xform_order == ("xformOp:translate", "xformOp:orient")
        and collider_xform_order == ("xformOp:scale",)
        and len(actual_translation) == 3
        and len(actual_orientation) == 4
        and len(actual_scale) == 3
        and isinstance(support_size, float)
        and math.isclose(
            support_size,
            min(CAP_GROCERY_SUPPORT_SIZE),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(
                actual_translation,
                CAP_GROCERY_SUPPORT_POSE[0],
                strict=True,
            )
        )
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(
                actual_orientation,
                CAP_GROCERY_SUPPORT_POSE[1],
                strict=True,
            )
        )
        and all(
            math.isclose(actual, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for actual, expected in zip(
                actual_scale,
                expected_scale,
                strict=True,
            )
        )
    ):
        raise GroceryCloseAuthorizationError(
            "composed grocery support pose/size drifted: "
            f"translate={actual_translation!r}, orient={actual_orientation!r}, "
            f"size={support_size!r}, scale={actual_scale!r}"
        )

    def require_offsets(
        prim: Any,
        *,
        expected_contact_m: float,
        expected_rest_m: float,
        label: str,
    ) -> None:
        contact_attribute = prim.GetAttribute("physxCollision:contactOffset")
        rest_attribute = prim.GetAttribute("physxCollision:restOffset")
        if not (
            contact_attribute.IsValid()
            and rest_attribute.IsValid()
            and contact_attribute.HasAuthoredValueOpinion()
            and rest_attribute.HasAuthoredValueOpinion()
        ):
            raise GroceryCloseAuthorizationError(
                f"composed {label} collision offsets are missing"
            )
        contact = contact_attribute.Get()
        rest = rest_attribute.Get()
        if not (
            isinstance(contact, float)
            and isinstance(rest, float)
            and math.isclose(
                contact,
                expected_contact_m,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            and math.isclose(
                rest,
                expected_rest_m,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise GroceryCloseAuthorizationError(
                f"composed {label} collision offsets drifted: ({contact!r}, {rest!r})"
            )

    require_offsets(
        support_collision,
        expected_contact_m=CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
        expected_rest_m=CAP_GROCERY_SUPPORT_REST_OFFSET_M,
        label="grocery support",
    )

    ground = stage.GetPrimAtPath(CAP_GROCERY_GROUND_COLLISION_PRIM_PATH)
    if (
        not ground.IsValid()
        or ground.GetTypeName() != "Plane"
        or not ground.HasAPI(UsdPhysics.CollisionAPI)
        or UsdPhysics.CollisionAPI(ground).GetCollisionEnabledAttr().Get() is not True
        or UsdGeom.Plane(ground).GetAxisAttr().Get() != UsdGeom.Tokens.z
    ):
        raise GroceryCloseAuthorizationError(
            "composed grocery ground-plane contract drifted"
        )
    try:
        from pxr import PhysxSchema
    except ImportError:
        # CPU-only OpenUSD tests validate the authored attributes below.
        pass
    else:
        if not ground.HasAPI(PhysxSchema.PhysxCollisionAPI):
            raise GroceryCloseAuthorizationError(
                "composed grocery ground PhysxCollisionAPI is missing"
            )
    require_offsets(
        ground,
        expected_contact_m=CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
        expected_rest_m=CAP_GROCERY_GROUND_REST_OFFSET_M,
        label="grocery ground",
    )

    ground_transform = UsdGeom.XformCache().GetLocalToWorldTransform(ground)
    ground_origin = ground_transform.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    ground_normal = ground_transform.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    normal_length = ground_normal.GetLength()
    if normal_length <= 0.0:
        raise GroceryCloseAuthorizationError(
            "composed grocery ground normal is degenerate"
        )
    normalized = ground_normal / normal_length
    if not (
        math.isclose(float(ground_origin[2]), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(normalized[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(normalized[1]), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(normalized[2]), 1.0, rel_tol=0.0, abs_tol=1.0e-12)
    ):
        raise GroceryCloseAuthorizationError(
            "composed grocery ground plane is not world z=0"
        )


class GroceryCollisionRuntimeContract:
    """Own exact PhysX views and produce same-timestamp close-proof inputs."""

    def __init__(
        self,
        environment: Any,
        *,
        robot_prim_path: str,
        can_prim_path: str,
        bin_prim_path: str,
        support_prim_path: str,
        dynamics_certificate: GroceryDynamicsCertificate | None = None,
    ) -> None:
        self._unwrapped = environment.unwrapped
        self._robot = self._unwrapped.scene["robot"]
        self._can = self._unwrapped.scene[CAP_GROCERY_OBJECT_ASSET]
        self._bin = self._unwrapped.scene[CAP_GROCERY_BIN_ASSET]
        self._support = self._unwrapped.scene[CAP_GROCERY_SUPPORT_INSTANCE]
        validate_live_grocery_fixture_contract(
            self._unwrapped.scene.stage,
            support_prim_path=support_prim_path,
        )
        if dynamics_certificate is not None and not isinstance(
            dynamics_certificate,
            GroceryDynamicsCertificate,
        ):
            raise GroceryCloseAuthorizationError(
                "grocery dynamics certificate has the wrong type"
            )
        self._dynamics_certificate = dynamics_certificate
        self._body_indices = {
            label: self._unique_body_index(body_name)
            for label, body_name in (
                ("palm", "base_link"),
                ("left", "left_inner_finger"),
                ("right", "right_inner_finger"),
            )
        }

        simulation_view = self._unwrapped.sim.physics_manager.get_physics_sim_view()
        if simulation_view is None:
            raise GroceryCloseAuthorizationError(
                "PhysX simulation view is unavailable after scene construction"
            )
        self._simulation_view = simulation_view
        self._view_specs = (
            _BodyViewSpec(
                "palm",
                f"{robot_prim_path}/Gripper/Robotiq_2F_85/base_link",
                1,
                (0.5, 0.5, 0.0),
                _PROXY_CONTACT_OFFSET_M,
                _PROXY_REST_OFFSET_M,
            ),
            _BodyViewSpec(
                "left inner finger",
                f"{robot_prim_path}/Gripper/Robotiq_2F_85/left_inner_finger",
                2,
                (2.0, 2.0, 0.0),
                _PROXY_CONTACT_OFFSET_M,
                _PROXY_REST_OFFSET_M,
            ),
            _BodyViewSpec(
                "right inner finger",
                f"{robot_prim_path}/Gripper/Robotiq_2F_85/right_inner_finger",
                2,
                (2.0, 2.0, 0.0),
                _PROXY_CONTACT_OFFSET_M,
                _PROXY_REST_OFFSET_M,
            ),
            _BodyViewSpec(
                "can",
                can_prim_path,
                1,
                (2.0, 2.0, 0.1),
                _CAN_PROXY_CONTACT_OFFSET_M,
                _CAN_PROXY_REST_OFFSET_M,
            ),
            _BodyViewSpec(
                "bin",
                bin_prim_path,
                5,
                (0.5, 0.5, 0.0),
                _BIN_PROXY_CONTACT_OFFSET_M,
                _BIN_PROXY_REST_OFFSET_M,
            ),
            _BodyViewSpec(
                "support",
                support_prim_path,
                1,
                (0.5, 0.5, 0.0),
                CAP_GROCERY_SUPPORT_CONTACT_OFFSET_M,
                CAP_GROCERY_SUPPORT_REST_OFFSET_M,
            ),
        )
        self._views = {
            spec.label: simulation_view.create_rigid_body_view(spec.path)
            for spec in self._view_specs
        }
        # Shape topology, materials, offsets, and mass properties are immutable
        # after scene construction in this owned process. Validate them once:
        # repeatedly fetching these backend tensors in the 200 Hz close loop can
        # synchronize CPU/GPU work and consume the controller's entire 5 ms tick.
        self._collision_offsets = self._require_runtime_contract()

    @property
    def dynamics_certified(self) -> bool:
        """Whether exact stock-equivalent dynamics were supplied and validated."""
        return self._dynamics_certificate is not None

    def require_dynamics_certificate(self) -> GroceryDynamicsCertificate:
        """Reject close authorization until exact stock dynamics are installed."""
        certificate = self._dynamics_certificate
        if certificate is None:
            raise GroceryCloseAuthorizationError(
                "grocery close is disabled: exact stock-equivalent mass/COM/inertia "
                "certificate is not installed"
            )
        return certificate

    def _unique_body_index(self, body_name: str) -> int:
        body_names = tuple(self._robot.body_names)
        matches = tuple(
            index
            for index, candidate in enumerate(body_names)
            if candidate == body_name
        )
        if len(matches) != 1:
            raise GroceryCloseAuthorizationError(
                f"robot body roster must contain exactly one {body_name!r}, "
                f"got {matches!r}"
            )
        return matches[0]

    def _require_simulation_view(self) -> None:
        is_valid = getattr(self._simulation_view, "is_valid", False)
        if callable(is_valid):
            is_valid = is_valid()
        if is_valid is not True:
            raise GroceryCloseAuthorizationError("PhysX simulation view is invalid")

    def _require_view_handle(
        self,
        spec: _BodyViewSpec,
    ) -> Any:
        view = self._views[spec.label]
        if getattr(view, "_backend", None) is None:
            raise GroceryCloseAuthorizationError(
                f"{spec.label} PhysX view has no active backend"
            )
        count = getattr(view, "count", None)
        max_shapes = getattr(view, "max_shapes", None)
        prim_paths = tuple(getattr(view, "prim_paths", ()))
        if isinstance(count, bool) or count != 1:
            raise GroceryCloseAuthorizationError(
                f"{spec.label} PhysX view count drifted: {count!r}"
            )
        if isinstance(max_shapes, bool) or max_shapes != spec.shape_count:
            raise GroceryCloseAuthorizationError(
                f"{spec.label} PhysX shape count drifted: "
                f"{max_shapes!r} != {spec.shape_count}"
            )
        if prim_paths != (spec.path,):
            raise GroceryCloseAuthorizationError(
                f"{spec.label} PhysX view path drifted: {prim_paths!r}"
            )
        return view

    def _require_view(
        self,
        spec: _BodyViewSpec,
    ) -> tuple[CollisionOffsets, ...]:
        view = self._require_view_handle(spec)

        contacts = _numeric_matrix(
            view.get_contact_offsets(),
            rows=1,
            columns=spec.shape_count,
            label=f"{spec.label} contact offsets",
        )[0]
        rests = _numeric_matrix(
            view.get_rest_offsets(),
            rows=1,
            columns=spec.shape_count,
            label=f"{spec.label} rest offsets",
        )[0]
        materials = _to_builtin(view.get_material_properties())
        if not isinstance(materials, (tuple, list)) or len(materials) != 1:
            raise GroceryCloseAuthorizationError(
                f"{spec.label} material properties must have one body row"
            )
        material_rows = _numeric_matrix(
            materials[0],
            rows=spec.shape_count,
            columns=3,
            label=f"{spec.label} material properties",
        )

        result: list[CollisionOffsets] = []
        for index, (contact, rest, material) in enumerate(
            zip(contacts, rests, material_rows, strict=True)
        ):
            if not (
                math.isclose(
                    contact,
                    spec.contact_offset_m,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                and math.isclose(
                    rest,
                    spec.rest_offset_m,
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
                and contact > rest
            ):
                raise GroceryCloseAuthorizationError(
                    f"{spec.label} shape {index} collision offsets drifted: "
                    f"({contact!r}, {rest!r})"
                )
            if any(
                not math.isclose(
                    actual,
                    expected,
                    rel_tol=0.0,
                    abs_tol=_MATERIAL_TOLERANCE,
                )
                for actual, expected in zip(
                    material,
                    spec.material,
                    strict=True,
                )
            ):
                raise GroceryCloseAuthorizationError(
                    f"{spec.label} shape {index} material drifted: "
                    f"{material!r} != {spec.material!r}"
                )
            result.append(CollisionOffsets(contact, rest))
        return tuple(result)

    def _require_mass_properties(self) -> None:
        robot_data = self._robot.data
        can_data = self._can.data
        bin_data = self._bin.data
        robot_masses = _to_builtin(robot_data.body_mass)
        robot_inertias = _to_builtin(robot_data.body_inertia)
        robot_com_poses = _to_builtin(robot_data.body_com_pose_b)
        can_masses = _to_builtin(can_data.body_mass)
        can_inertias = _to_builtin(can_data.body_inertia)
        can_com_poses = _to_builtin(can_data.body_com_pose_b)
        bin_masses = _to_builtin(bin_data.body_mass)
        bin_inertias = _to_builtin(bin_data.body_inertia)
        bin_com_poses = _to_builtin(bin_data.body_com_pose_b)
        try:
            records = tuple(
                (
                    label,
                    robot_masses[0][index],
                    robot_inertias[0][index],
                    robot_com_poses[0][index],
                )
                for label, index in self._body_indices.items()
            ) + (
                (
                    "can",
                    can_masses[0][0],
                    can_inertias[0][0],
                    can_com_poses[0][0],
                ),
                (
                    "bin",
                    bin_masses[0][0],
                    bin_inertias[0][0],
                    bin_com_poses[0][0],
                ),
            )
        except (IndexError, TypeError) as exc:
            raise GroceryCloseAuthorizationError(
                "runtime mass/COM/inertia tensor shape drifted"
            ) from exc

        for label, raw_mass, raw_inertia, raw_com_pose in records:
            mass = _finite_float(raw_mass, label=f"{label} mass")
            if mass <= 0.0:
                raise GroceryCloseAuthorizationError(f"{label} mass must be positive")
            inertia = _numeric_row(
                raw_inertia,
                length=9,
                label=f"{label} inertia",
            )
            _positive_definite_inertia(inertia, label=f"{label} inertia")
            com_pose = _numeric_row(
                raw_com_pose,
                length=7,
                label=f"{label} COM pose",
            )
            Pose(
                position_m=com_pose[:3],  # type: ignore[arg-type]
                orientation_xyzw=com_pose[3:],  # type: ignore[arg-type]
            )

    def _require_runtime_contract(
        self,
    ) -> GroceryCollisionOffsets:
        self._require_simulation_view()
        offsets = {spec.label: self._require_view(spec) for spec in self._view_specs}
        self._require_mass_properties()
        return GroceryCollisionOffsets(
            palm=offsets["palm"][0],
            left_finger4=offsets["left inner finger"][0],
            left_fingertip=offsets["left inner finger"][1],
            right_finger4=offsets["right inner finger"][0],
            right_fingertip=offsets["right inner finger"][1],
            can=offsets["can"][0],
            bin=offsets["bin"][0],
            support=offsets["support"][0],
            ground=CollisionOffsets(
                CAP_GROCERY_GROUND_CONTACT_OFFSET_M,
                CAP_GROCERY_GROUND_REST_OFFSET_M,
            ),
        )

    def capture(self) -> GroceryRuntimeGeometry:
        """Capture all close-proof geometry without advancing simulation."""
        self._require_simulation_view()
        for spec in self._view_specs:
            self._require_view_handle(spec)
        robot_timestamp = _finite_float(
            getattr(self._robot.data, "_sim_timestamp", None),
            label="robot simulation timestamp",
        )
        can_timestamp = _finite_float(
            getattr(self._can.data, "_sim_timestamp", None),
            label="can simulation timestamp",
        )
        bin_timestamp = _finite_float(
            getattr(self._bin.data, "_sim_timestamp", None),
            label="bin simulation timestamp",
        )
        support_timestamp = _finite_float(
            getattr(self._support.data, "_sim_timestamp", None),
            label="support simulation timestamp",
        )
        if not (
            math.isclose(
                robot_timestamp,
                can_timestamp,
                rel_tol=0.0,
                abs_tol=_TIMESTAMP_TOLERANCE_S,
            )
            and math.isclose(
                robot_timestamp,
                bin_timestamp,
                rel_tol=0.0,
                abs_tol=_TIMESTAMP_TOLERANCE_S,
            )
            and math.isclose(
                robot_timestamp,
                support_timestamp,
                rel_tol=0.0,
                abs_tol=_TIMESTAMP_TOLERANCE_S,
            )
        ):
            raise GroceryCloseAuthorizationError(
                "robot, can, bin, and support observations are not from the same "
                "simulation timestamp: "
                f"{robot_timestamp!r}, {can_timestamp!r}, {bin_timestamp!r}, "
                f"{support_timestamp!r}"
            )

        robot_poses = _to_builtin(self._robot.data.body_link_pose_w)
        can_poses = _to_builtin(self._can.data.root_link_pose_w)
        bin_poses = _to_builtin(self._bin.data.root_link_pose_w)
        support_poses = _to_builtin(self._support.data.root_link_pose_w)
        try:
            base = _numeric_row(
                robot_poses[0][self._body_indices["palm"]],
                length=7,
                label="gripper base pose",
            )
            left = _numeric_row(
                robot_poses[0][self._body_indices["left"]],
                length=7,
                label="left inner finger pose",
            )
            right = _numeric_row(
                robot_poses[0][self._body_indices["right"]],
                length=7,
                label="right inner finger pose",
            )
            can = _numeric_row(
                can_poses[0],
                length=7,
                label="can root-link pose",
            )
            bin_pose = _numeric_row(
                bin_poses[0],
                length=7,
                label="bin root-link pose",
            )
            support_pose = _numeric_row(
                support_poses[0],
                length=7,
                label="support root-link pose",
            )
        except (IndexError, TypeError) as exc:
            raise GroceryCloseAuthorizationError(
                "runtime body-pose tensor shape drifted"
            ) from exc

        def pose(values: tuple[float, ...]) -> Pose:
            return Pose(
                position_m=values[:3],  # type: ignore[arg-type]
                orientation_xyzw=values[3:],  # type: ignore[arg-type]
            )

        return GroceryRuntimeGeometry(
            simulation_timestamp_s=robot_timestamp,
            gripper_base_pose=pose(base),
            left_inner_finger_pose=pose(left),
            right_inner_finger_pose=pose(right),
            can_pose=pose(can),
            bin_pose=pose(bin_pose),
            support_pose=pose(support_pose),
            collision_offsets=self._collision_offsets,
        )
