# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load USD prim trees with physics metadata for object_reference resolution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from isaaclab_arena.assets.object_type import ObjectType

if TYPE_CHECKING:
    from pxr import Usd


@dataclass(frozen=True)
class UsdBoundingBox:
    """Axis-aligned bounds in the inspected root's frame, expressed in meters."""

    center_m: tuple[float, float, float]
    """Bounding-box center, not the body origin or pose."""
    size_m: tuple[float, float, float]


@dataclass(frozen=True)
class UsdJointInfo:
    """Authored USD joint properties with root-relative or external absolute body paths."""

    type_name: str
    enabled: bool
    body_0_path: str | None
    body_1_path: str | None
    axis: str | None = None
    lower_limit: float | None = None
    upper_limit: float | None = None
    limit_unit: str | None = None


@dataclass(frozen=True)
class UsdColliderInfo:
    """Authored collider properties with a root-relative or external absolute owner path."""

    type_name: str
    enabled: bool
    rigid_body_owner_path: str | None
    approximation: str | None
    contact_offset_m: float | None = None
    rest_offset_m: float | None = None


@dataclass(frozen=True)
class UsdRigidBodyInfo:
    """Authored USD rigid-body and mass properties."""

    enabled: bool
    kinematic: bool
    mass_kg: float | None
    density_kg_m3: float | None
    center_of_mass: tuple[float, ...] | None
    diagonal_inertia: tuple[float, ...] | None
    disable_gravity: bool | None = None
    linear_damping: float | None = None
    angular_damping: float | None = None


@dataclass(frozen=True)
class UsdInstanceInfo:
    """Composed USD instancing state."""

    instance: bool
    instance_proxy: bool
    instanceable: bool
    prototype_path: str | None


@dataclass(frozen=True)
class UsdPrimRecord:
    """One prim inside a USD asset, keyed by inspection-root-relative suffix."""

    relative_path: str
    object_type: ObjectType
    joint_names: tuple[str, ...] = ()
    bounding_box_root: UsdBoundingBox | None = None
    joint: UsdJointInfo | None = None
    collider: UsdColliderInfo | None = None
    rigid_body: UsdRigidBodyInfo | None = None
    instance: UsdInstanceInfo | None = None


def format_usd_prim_record_metadata(record: UsdPrimRecord) -> str:
    """Return short physics labels for hierarchy browsing.

    Args:
        record: Prim record whose joint and collider metadata to summarize.

    Returns:
        Space-separated labels, or an empty string when the record has no displayed metadata.
    """
    fields: list[str] = []
    if record.joint_names:
        fields.append(f"joints=[{','.join(record.joint_names)}]")
    if record.joint is not None:
        fields.append(f"joint(type={record.joint.type_name})")
    if record.collider is not None:
        fields.append(f"collider(type={record.collider.type_name},approximation={record.collider.approximation})")
    return " ".join(fields)


def format_usd_prim_tree(prim_tree: list[UsdPrimRecord], *, heading: str = "USD PRIM TREE:") -> str:
    """Format prim records as a nested text tree with physics metadata.

    Args:
        prim_tree: Prim records to sort and display by relative path.
        heading: First line of the formatted tree.

    Returns:
        Newline-separated tree text.
    """
    records = sorted(prim_tree, key=lambda record: record.relative_path)
    lines = [heading]
    stack: list[str] = []
    for record in records:
        path = record.relative_path
        while stack and not path.startswith(stack[-1] + "/"):
            stack.pop()
        parent = stack[-1] if stack else ""
        suffix = path[len(parent) + 1 :] if parent else path
        indent = "  " * len(stack)
        tag = record.object_type.value
        metadata = format_usd_prim_record_metadata(record)
        if metadata:
            tag += " " + metadata
        lines.append(f"{indent}{suffix} ({tag})")
        stack.append(path)
    return "\n".join(lines)


def usd_prim_records_to_dicts(prim_tree: list[UsdPrimRecord]) -> list[dict[str, Any]]:
    """Return complete records suitable for strict JSON serialization.

    Args:
        prim_tree: Prim records to convert.

    Returns:
        Record dictionaries with object types encoded as strings and non-finite floats encoded as
        ``"inf"``, ``"-inf"``, or ``"nan"``.
    """
    records = []
    for record in prim_tree:
        data = asdict(record)
        data["object_type"] = record.object_type.value
        records.append(_json_value(data))
    return records


def find_nested_physics_roots(root_prim: Usd.Prim) -> dict[str, ObjectType]:
    """Return absolute paths of independently resettable physics roots.

    Standalone rigid bodies and articulation roots are reset roots. Articulation
    links are owned by their articulation root, while base and collision-only prims
    are not reset roots.

    Args:
        root_prim: Root prim whose composed subtree to inspect.

    Returns:
        Physics object types keyed by absolute prim path.
    """
    from pxr import Usd

    from isaaclab_arena.utils.usd.helpers import is_articulation_root, is_rigid_body

    root_path = root_prim.GetPath()
    articulation_paths = set()
    roots: dict[str, ObjectType] = {}
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        prim_path = prim.GetPath()
        articulation_root = is_articulation_root(prim)
        if articulation_root:
            articulation_paths.add(prim_path)
        if prim == root_prim:
            continue
        if articulation_root:
            roots[str(prim_path)] = ObjectType.ARTICULATION
        elif is_rigid_body(prim):
            ancestor = prim.GetParent()
            while ancestor.IsValid() and ancestor.GetPath().HasPrefix(root_path):
                if ancestor.GetPath() in articulation_paths:
                    break
                ancestor = ancestor.GetParent()
            else:
                roots[str(prim_path)] = ObjectType.RIGID
    return roots


def exclude_referenced_physics_roots(
    physics_roots: dict[str, ObjectType],
    referenced_prim_paths: dict[str, ObjectType],
) -> dict[str, ObjectType]:
    """Remove physics roots whose reset state is owned by matching object references.

    Args:
        physics_roots: Independently resettable physics roots keyed by absolute prim path.
        referenced_prim_paths: Object references keyed by absolute prim path.

    Returns:
        Physics roots not owned by rigid or articulation object references.
    """
    reset_owned_references = {
        path: object_type
        for path, object_type in referenced_prim_paths.items()
        if object_type in (ObjectType.RIGID, ObjectType.ARTICULATION)
    }
    articulation_paths = tuple(
        path for path, object_type in physics_roots.items() if object_type == ObjectType.ARTICULATION
    )
    for path, referenced_type in reset_owned_references.items():
        physics_type = physics_roots.get(path)
        if physics_type is None:
            articulation_owner = next(
                (
                    articulation_path
                    for articulation_path in articulation_paths
                    if path.startswith(f"{articulation_path}/")
                ),
                None,
            )
            assert articulation_owner is None, (
                f"Object reference '{path}' lies inside articulation root '{articulation_owner}' and cannot "
                "independently own its reset state. Reference the articulation root instead."
            )
            assert (
                physics_type is not None
            ), f"Object reference '{path}' does not identify an independently resettable physics root"
        assert referenced_type == physics_type, (
            f"Object reference '{path}' is declared as {referenced_type.name}, "
            f"but its physics root is {physics_type.name}"
        )

    return {path: object_type for path, object_type in physics_roots.items() if path not in reset_owned_references}


def load_usd_physics_roots(
    usd_path: str,
    referenced_prim_paths: dict[str, ObjectType] | None = None,
) -> dict[str, ObjectType]:
    """Return independently resettable physics roots in a composed USD asset.

    Articulation links are owned by their articulation root and are omitted.
    Rigid bodies outside articulation subtrees remain separate records, including
    bodies connected by ordinary USD joints. Base and collision-only prims are
    omitted. Physics roots represented by matching rigid or articulation object
    references are also omitted because those scene entities own their reset events.

    Args:
        usd_path: Local filesystem path or Nucleus/HTTPS URL for the USD.
        referenced_prim_paths: Default-prim-relative paths owned by explicit
            object references, mapped to their physics object types.

    Returns:
        Physics object types keyed by sorted default-prim-relative paths.
    """
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_arena.utils.usd.helpers import open_stage, relative_path_from_default_prim

    local_usd_path = retrieve_file_path(usd_path)
    roots: dict[str, ObjectType] = {}
    with open_stage(local_usd_path) as stage:
        stage.Load()
        default_prim = stage.GetDefaultPrim()
        default_prim_path = str(default_prim.GetPath())
        referenced_absolute_paths = {
            f"{default_prim_path}/{relative_path}": object_type
            for relative_path, object_type in (referenced_prim_paths or {}).items()
        }
        physics_roots = exclude_referenced_physics_roots(
            find_nested_physics_roots(default_prim),
            referenced_absolute_paths,
        )
        for prim_path, object_type in physics_roots.items():
            relative_path = relative_path_from_default_prim(stage, prim_path)
            if not relative_path:
                continue
            roots[relative_path] = object_type

    return dict(sorted(roots.items()))


def load_stage_prim_tree(
    stage: Usd.Stage,
    root_prim: Usd.Prim,
    *,
    include_physical_root: bool = False,
) -> list[UsdPrimRecord]:
    """Return physics/collision records below a composed stage prim.

    A prim is included when it directly participates in physics or collision, or
    when any descendant does. Record paths and bounds are relative to ``root_prim``.
    Body paths outside that subtree are absolute; ``.`` identifies the root itself.
    Bounds include all geometry purposes. Bounds and collider offsets use meters.

    Args:
        stage: Open composed USD stage containing ``root_prim``.
        root_prim: Root prim of the subtree to inspect.
        include_physical_root: Include ``root_prim`` as path ``.`` when it directly
            participates in physics.

    Returns:
        Sorted list of :class:`UsdPrimRecord` entries keyed by relative_path suffix.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.helpers import (
        articulation_joint_names,
        has_physics_or_collision,
        object_type_for_prim,
    )

    records: list[UsdPrimRecord] = []
    assert root_prim and root_prim.IsValid(), "Inspection root prim must exist."
    prims = tuple(Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()))
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), UsdGeom.Imageable.GetOrderedPurposeTokens())
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    kilograms_per_unit = UsdPhysics.GetStageKilogramsPerUnit(stage)
    root_path = root_prim.GetPath()
    included_paths: set[Sdf.Path] = set()
    for prim in prims:
        if not has_physics_or_collision(prim) and not prim.IsA(UsdPhysics.Joint):
            continue
        ancestor = prim
        while ancestor and ancestor.GetPath().HasPrefix(root_path):
            path = ancestor.GetPath()
            if path in included_paths:
                break
            included_paths.add(path)
            if ancestor == root_prim:
                break
            ancestor = ancestor.GetParent()

    for prim in prims:
        prim_path = prim.GetPath()
        if prim_path not in included_paths:
            continue
        if prim == root_prim and (
            not include_physical_root or (not has_physics_or_collision(prim) and not prim.IsA(UsdPhysics.Joint))
        ):
            continue
        relative_path = str(prim_path.MakeRelativePath(root_path))
        object_type = object_type_for_prim(prim)
        joint_names: tuple[str, ...] = ()
        if object_type == ObjectType.ARTICULATION:
            joint_names = articulation_joint_names(prim)
        directly_physical = has_physics_or_collision(prim)
        prototype = prim.GetPrototype() if prim.IsInstance() else prim.GetPrimInPrototype()
        records.append(
            UsdPrimRecord(
                relative_path=relative_path,
                object_type=object_type,
                joint_names=joint_names,
                bounding_box_root=(
                    _bounding_box_in_root_frame(prim, root_prim, bbox_cache, meters_per_unit)
                    if directly_physical
                    else None
                ),
                joint=(_joint_info(prim, stage, root_prim, UsdPhysics) if prim.IsA(UsdPhysics.Joint) else None),
                collider=(
                    _collider_info(prim, stage, root_prim, UsdPhysics, meters_per_unit)
                    if prim.HasAPI(UsdPhysics.CollisionAPI)
                    else None
                ),
                rigid_body=(
                    _rigid_body_info(prim, UsdPhysics, kilograms_per_unit, meters_per_unit)
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI)
                    else None
                ),
                instance=UsdInstanceInfo(
                    instance=prim.IsInstance(),
                    instance_proxy=prim.IsInstanceProxy(),
                    instanceable=prim.IsInstanceable(),
                    prototype_path=str(prototype.GetPath()) if prototype and prototype.IsValid() else None,
                ),
            )
        )
    records.sort(key=lambda record: record.relative_path)
    return records


def load_usd_prim_tree(usd_path: str, prim_path: str | None = None) -> list[UsdPrimRecord]:
    """Return prim records for a selected physics/collision subtree of a USD asset.

    Args:
        usd_path: Local filesystem path or Nucleus/HTTPS URL for the USD.
        prim_path: Optional absolute stage path or default-prim-relative subtree.

    Returns:
        Sorted prim records with paths relative to the selected root.
    """
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_arena.utils.usd.helpers import open_stage

    local_usd_path = retrieve_file_path(usd_path)
    with open_stage(local_usd_path) as stage:
        stage.Load()
        if prim_path and prim_path.startswith("/"):
            absolute_path = prim_path
            root_prim = stage.GetPrimAtPath(absolute_path)
            assert root_prim and root_prim.IsValid(), f"USD prim does not exist: {prim_path}"
        else:
            default_prim = stage.GetDefaultPrim()
            assert default_prim and default_prim.IsValid(), f"USD has no valid default prim: {usd_path}"
            if prim_path:
                root_prim = stage.GetPrimAtPath(f"{default_prim.GetPath()}/{prim_path}")
                assert root_prim and root_prim.IsValid(), f"USD prim does not exist: {prim_path}"
            else:
                root_prim = default_prim
        return load_stage_prim_tree(stage, root_prim, include_physical_root=bool(prim_path))


################################################################################
# Bounds and rigid-body metadata
################################################################################


def _bounding_box_in_root_frame(prim, root_prim, bbox_cache, meters_per_unit: float) -> UsdBoundingBox | None:
    """Return meter-valued bounds in frame R, the inspected root's frame."""
    relative_range = bbox_cache.ComputeRelativeBound(prim, root_prim).ComputeAlignedRange()
    if relative_range.IsEmpty():
        return None
    center_R = relative_range.GetMidpoint()
    size_R = relative_range.GetSize()
    return UsdBoundingBox(
        center_m=tuple(float(component) * meters_per_unit for component in center_R),
        size_m=tuple(float(component) * meters_per_unit for component in size_R),
    )


def _rigid_body_info(
    prim,
    UsdPhysics,
    kilograms_per_unit: float,
    meters_per_unit: float,
) -> UsdRigidBodyInfo:
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    mass = UsdPhysics.MassAPI(prim)
    mass_value = _authored_attribute_value(mass.GetMassAttr())
    density_value = _authored_attribute_value(mass.GetDensityAttr())
    return UsdRigidBodyInfo(
        enabled=bool(_attribute_value(rigid_body.GetRigidBodyEnabledAttr(), True)),
        kinematic=bool(_attribute_value(rigid_body.GetKinematicEnabledAttr(), False)),
        mass_kg=None if mass_value is None else float(mass_value) * kilograms_per_unit,
        density_kg_m3=(
            None if density_value is None else float(density_value) * kilograms_per_unit / meters_per_unit**3
        ),
        center_of_mass=_optional_tuple(_authored_attribute_value(mass.GetCenterOfMassAttr())),
        diagonal_inertia=_optional_tuple(_authored_attribute_value(mass.GetDiagonalInertiaAttr())),
        disable_gravity=_authored_named_attribute_value(prim, "physxRigidBody:disableGravity"),
        linear_damping=_optional_float(_authored_named_attribute_value(prim, "physxRigidBody:linearDamping")),
        angular_damping=_optional_float(_authored_named_attribute_value(prim, "physxRigidBody:angularDamping")),
    )


################################################################################
# Joint and collider metadata
################################################################################


def _nearest_rigid_body_path(stage, target_path, root_prim, UsdPhysics) -> str | None:
    """Find the owning body across the stage, using absolute paths outside the inspected root."""
    prim = stage.GetPrimAtPath(target_path)
    root_path = root_prim.GetPath()
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim_path = prim.GetPath()
            return str(prim_path.MakeRelativePath(root_path)) if prim_path.HasPrefix(root_path) else str(prim_path)
        prim = prim.GetParent()
    return None


def _joint_body_path(stage, relationship, root_prim, UsdPhysics) -> str | None:
    targets = relationship.GetTargets() if relationship else []
    if len(targets) != 1:
        return None
    return _nearest_rigid_body_path(stage, targets[0], root_prim, UsdPhysics)


def _joint_info(prim, stage, root_prim, UsdPhysics) -> UsdJointInfo:
    joint = UsdPhysics.Joint(prim)
    axis = None
    lower_limit = None
    upper_limit = None
    limit_unit = None
    if prim.IsA(UsdPhysics.RevoluteJoint):
        typed_joint = UsdPhysics.RevoluteJoint(prim)
        axis = str(_attribute_value(typed_joint.GetAxisAttr()))
        lower_limit = _optional_float(_attribute_value(typed_joint.GetLowerLimitAttr()))
        upper_limit = _optional_float(_attribute_value(typed_joint.GetUpperLimitAttr()))
        limit_unit = "degrees"
    elif prim.IsA(UsdPhysics.PrismaticJoint):
        typed_joint = UsdPhysics.PrismaticJoint(prim)
        axis = str(_attribute_value(typed_joint.GetAxisAttr()))
        lower_limit = _optional_float(_attribute_value(typed_joint.GetLowerLimitAttr()))
        upper_limit = _optional_float(_attribute_value(typed_joint.GetUpperLimitAttr()))
        limit_unit = "stage_units"
    return UsdJointInfo(
        type_name=prim.GetTypeName(),
        enabled=bool(_attribute_value(joint.GetJointEnabledAttr(), True)),
        body_0_path=_joint_body_path(stage, joint.GetBody0Rel(), root_prim, UsdPhysics),
        body_1_path=_joint_body_path(stage, joint.GetBody1Rel(), root_prim, UsdPhysics),
        axis=axis,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        limit_unit=limit_unit,
    )


def _collider_info(prim, stage, root_prim, UsdPhysics, meters_per_unit: float) -> UsdColliderInfo:
    collision = UsdPhysics.CollisionAPI(prim)
    approximation = _attribute_value(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr())
    return UsdColliderInfo(
        type_name=prim.GetTypeName(),
        enabled=bool(_attribute_value(collision.GetCollisionEnabledAttr(), True)),
        rigid_body_owner_path=_nearest_rigid_body_path(stage, prim.GetPath(), root_prim, UsdPhysics),
        approximation=str(approximation) if approximation is not None else None,
        contact_offset_m=_distance_in_meters(
            _authored_named_attribute_value(prim, "physxCollision:contactOffset"), meters_per_unit
        ),
        rest_offset_m=_distance_in_meters(
            _authored_named_attribute_value(prim, "physxCollision:restOffset"), meters_per_unit
        ),
    )


################################################################################
# Value conversion helpers
################################################################################


def _json_value(value: Any) -> Any:
    """Convert nested metadata to JSON values without non-finite numeric tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _attribute_value(attribute, default=None):
    value = attribute.Get() if attribute else None
    return default if value is None else value


def _authored_attribute_value(attribute):
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None
    return attribute.Get()


def _authored_named_attribute_value(prim, name: str):
    return _authored_attribute_value(prim.GetAttribute(name))


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_tuple(value) -> tuple[float, ...] | None:
    return None if value is None else tuple(float(component) for component in value)


def _distance_in_meters(value, meters_per_unit: float) -> float | None:
    return None if value is None else float(value) * meters_per_unit
