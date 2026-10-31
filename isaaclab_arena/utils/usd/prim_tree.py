# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load USD prim trees with physics metadata for object_reference resolution."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.assets.object_type import ObjectType


@dataclass(frozen=True)
class UsdBoundingBox:
    """Axis-aligned bounds in the asset default-prim frame."""

    center_m: tuple[float, float, float]
    size_m: tuple[float, float, float]


@dataclass(frozen=True)
class UsdJointInfo:
    """Authored USD joint properties."""

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
    """Authored USD collider properties."""

    type_name: str
    enabled: bool
    rigid_body_owner_path: str | None
    approximation: str | None


@dataclass(frozen=True)
class UsdRigidBodyInfo:
    """Authored USD rigid-body and mass properties."""

    enabled: bool
    kinematic: bool
    mass_kg: float | None
    density_kg_m3: float | None
    center_of_mass: tuple[float, ...] | None
    diagonal_inertia: tuple[float, ...] | None


@dataclass(frozen=True)
class UsdPrimRecord:
    """One prim inside a USD asset, keyed by default-prim-relative suffix."""

    relative_path: str
    object_type: ObjectType
    joint_names: tuple[str, ...] = ()
    bounding_box_root: UsdBoundingBox | None = None
    joint: UsdJointInfo | None = None
    collider: UsdColliderInfo | None = None
    rigid_body: UsdRigidBodyInfo | None = None


def _format_tuple(values: tuple[float, ...]) -> str:
    return "[" + ",".join(f"{value:.6g}" for value in values) + "]"


def format_usd_prim_record_metadata(record: UsdPrimRecord) -> str:
    """Return compact physics and spatial metadata for a prim-tree record."""
    fields: list[str] = []
    if record.joint_names:
        fields.append(f"joints=[{','.join(record.joint_names)}]")
    if record.bounding_box_root is not None:
        bounds = record.bounding_box_root
        fields.extend((
            f"bbox_center_m={_format_tuple(bounds.center_m)}",
            f"bbox_size_m={_format_tuple(bounds.size_m)}",
        ))
    if record.joint is not None:
        joint = record.joint
        joint_fields = [
            f"type={joint.type_name}",
            f"enabled={str(joint.enabled).lower()}",
            f"body_0={joint.body_0_path or 'world'}",
            f"body_1={joint.body_1_path or 'world'}",
        ]
        if joint.axis is not None:
            joint_fields.append(f"axis={joint.axis}")
        if joint.lower_limit is not None or joint.upper_limit is not None:
            joint_fields.append(f"limits=[{joint.lower_limit},{joint.upper_limit}]")
            joint_fields.append(f"limit_unit={joint.limit_unit}")
        fields.append(f"joint({','.join(joint_fields)})")
    if record.collider is not None:
        collider = record.collider
        collider_fields = [
            f"type={collider.type_name}",
            f"enabled={str(collider.enabled).lower()}",
            f"rigid_body_owner={collider.rigid_body_owner_path or 'world'}",
            f"approximation={collider.approximation}",
        ]
        fields.append(f"collider({','.join(collider_fields)})")
    if record.rigid_body is not None:
        rigid_body = record.rigid_body
        rigid_body_fields = [
            f"enabled={str(rigid_body.enabled).lower()}",
            f"kinematic={str(rigid_body.kinematic).lower()}",
            f"mass_kg={rigid_body.mass_kg}",
            f"density_kg_m3={rigid_body.density_kg_m3}",
        ]
        if rigid_body.center_of_mass is not None:
            rigid_body_fields.append(f"center_of_mass={_format_tuple(rigid_body.center_of_mass)}")
        if rigid_body.diagonal_inertia is not None:
            rigid_body_fields.append(f"diagonal_inertia={_format_tuple(rigid_body.diagonal_inertia)}")
        fields.append(f"rigid_body({','.join(rigid_body_fields)})")
    return " ".join(fields)


def format_usd_prim_tree(prim_tree: list[UsdPrimRecord], *, heading: str = "USD PRIM TREE:") -> str:
    """Format prim records as a nested text tree with physics metadata."""
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


def _attribute_value(attribute, default=None):
    value = attribute.Get() if attribute else None
    return default if value is None else value


def _authored_attribute_value(attribute):
    if not attribute or not attribute.HasAuthoredValueOpinion():
        return None
    return attribute.Get()


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _optional_tuple(value) -> tuple[float, ...] | None:
    return None if value is None else tuple(float(component) for component in value)


def _bounding_box_in_root_frame(prim, default_prim, bbox_cache) -> UsdBoundingBox | None:
    relative_range = bbox_cache.ComputeRelativeBound(prim, default_prim).ComputeAlignedRange()
    if relative_range.IsEmpty():
        return None
    center = relative_range.GetMidpoint()
    size = relative_range.GetSize()
    return UsdBoundingBox(
        center_m=tuple(float(component) for component in center),
        size_m=tuple(float(component) for component in size),
    )


def _rigid_body_info(prim, UsdPhysics) -> UsdRigidBodyInfo:
    rigid_body = UsdPhysics.RigidBodyAPI(prim)
    mass = UsdPhysics.MassAPI(prim)
    return UsdRigidBodyInfo(
        enabled=bool(_attribute_value(rigid_body.GetRigidBodyEnabledAttr(), True)),
        kinematic=bool(_attribute_value(rigid_body.GetKinematicEnabledAttr(), False)),
        mass_kg=_optional_float(_authored_attribute_value(mass.GetMassAttr())),
        density_kg_m3=_optional_float(_authored_attribute_value(mass.GetDensityAttr())),
        center_of_mass=_optional_tuple(_authored_attribute_value(mass.GetCenterOfMassAttr())),
        diagonal_inertia=_optional_tuple(_authored_attribute_value(mass.GetDiagonalInertiaAttr())),
    )


def _nearest_rigid_body_path(stage, target_path, rigid_body_paths: set[str], default_prim) -> str | None:
    prim = stage.GetPrimAtPath(target_path)
    default_path = default_prim.GetPath()
    while prim and prim.IsValid() and prim.GetPath().HasPrefix(default_path):
        prim_path = str(prim.GetPath())
        if prim_path in rigid_body_paths:
            default_prefix = str(default_path).rstrip("/") + "/"
            return prim_path[len(default_prefix) :] if prim_path.startswith(default_prefix) else prim_path.lstrip("/")
        if prim == default_prim:
            break
        prim = prim.GetParent()
    return None


def _joint_body_path(stage, relationship, rigid_body_paths: set[str], default_prim) -> str | None:
    targets = relationship.GetTargets() if relationship else []
    if len(targets) != 1:
        return None
    return _nearest_rigid_body_path(stage, targets[0], rigid_body_paths, default_prim)


def _joint_info(prim, stage, rigid_body_paths: set[str], default_prim, UsdPhysics) -> UsdJointInfo:
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
        body_0_path=_joint_body_path(stage, joint.GetBody0Rel(), rigid_body_paths, default_prim),
        body_1_path=_joint_body_path(stage, joint.GetBody1Rel(), rigid_body_paths, default_prim),
        axis=axis,
        lower_limit=lower_limit,
        upper_limit=upper_limit,
        limit_unit=limit_unit,
    )


def _collider_info(prim, stage, rigid_body_paths: set[str], default_prim, UsdPhysics) -> UsdColliderInfo:
    collision = UsdPhysics.CollisionAPI(prim)
    approximation = _attribute_value(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr())
    return UsdColliderInfo(
        type_name=prim.GetTypeName(),
        enabled=bool(_attribute_value(collision.GetCollisionEnabledAttr(), True)),
        rigid_body_owner_path=_nearest_rigid_body_path(
            stage,
            prim.GetPath(),
            rigid_body_paths,
            default_prim,
        ),
        approximation=str(approximation) if approximation is not None else None,
    )


def _traverse_composed_stage(stage):
    """Traverse loaded prims, including prims beneath instance proxies."""
    from pxr import Usd

    stage.Load()
    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def find_nested_physics_roots(root_prim) -> dict[str, ObjectType]:
    """Return absolute paths of independently resettable physics roots.

    Standalone rigid bodies and articulation roots are reset roots. Articulation
    links are owned by their articulation root, while base and collision-only prims
    are not reset roots.
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
    """Remove physics roots whose reset state is owned by matching object references."""
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


def load_usd_prim_tree(usd_path: str) -> list[UsdPrimRecord]:
    """Return prim records for the physics/collision subtree of a USD asset.

    A prim is included when it directly participates in physics or collision, or
    when any of its descendants does. Retaining those ancestors keeps the returned
    records connected as a tree so the nested catalog can recover full paths.

    Args:
        usd_path: Local filesystem path or Nucleus/HTTPS URL for the USD. Remote
            URLs are downloaded via :func:`retrieve_file_path` first — bare
            ``Usd.Stage.Open(https://…)`` fails outside a Kit asset resolver
            (e.g. Streamlit generation).

    Returns:
        Sorted list of :class:`UsdPrimRecord` entries keyed by relative_path suffix.
    """
    from isaaclab.utils.assets import retrieve_file_path
    from pxr import Usd, UsdGeom, UsdPhysics

    from isaaclab_arena.utils.usd.helpers import (
        articulation_joint_names,
        has_physics_or_collision,
        object_type_for_prim,
        open_stage,
        relative_path_from_default_prim,
    )

    local_usd_path = retrieve_file_path(usd_path)
    records: list[UsdPrimRecord] = []
    with open_stage(local_usd_path) as stage:
        prims = tuple(_traverse_composed_stage(stage))
        default_prim = stage.GetDefaultPrim()
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        rigid_body_paths = {str(prim.GetPath()) for prim in prims if prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        # Collect prims that directly participate in physics or collision, then add
        # every ancestor so a prim is kept whenever any descendant is kept.
        # TODO(qianl): Ancestor-only prims are labeled base; non-leaf refs are valid today.
        # Revisit when relation solving adds descendant mesh exclusion; no issue observed yet.
        included_paths: set[str] = set()
        for prim in prims:
            if prim.IsPseudoRoot():
                continue
            if not has_physics_or_collision(prim) and not prim.IsA(UsdPhysics.Joint):
                continue
            ancestor = prim
            while ancestor and not ancestor.IsPseudoRoot():
                path = str(ancestor.GetPath())
                if path in included_paths:
                    break
                included_paths.add(path)
                ancestor = ancestor.GetParent()

        for prim in prims:
            if prim.IsPseudoRoot():
                continue
            if str(prim.GetPath()) not in included_paths:
                continue
            relative_path = relative_path_from_default_prim(stage, str(prim.GetPath()))
            if not relative_path:
                continue
            object_type = object_type_for_prim(prim)
            joint_names: tuple[str, ...] = ()
            if object_type == ObjectType.ARTICULATION:
                joint_names = articulation_joint_names(prim)
            directly_physical = has_physics_or_collision(prim)
            records.append(
                UsdPrimRecord(
                    relative_path=relative_path,
                    object_type=object_type,
                    joint_names=joint_names,
                    bounding_box_root=(
                        _bounding_box_in_root_frame(prim, default_prim, bbox_cache) if directly_physical else None
                    ),
                    joint=(
                        _joint_info(prim, stage, rigid_body_paths, default_prim, UsdPhysics)
                        if prim.IsA(UsdPhysics.Joint)
                        else None
                    ),
                    collider=(
                        _collider_info(prim, stage, rigid_body_paths, default_prim, UsdPhysics)
                        if prim.HasAPI(UsdPhysics.CollisionAPI)
                        else None
                    ),
                    rigid_body=_rigid_body_info(prim, UsdPhysics) if prim.HasAPI(UsdPhysics.RigidBodyAPI) else None,
                )
            )
    records.sort(key=lambda record: record.relative_path)
    return records
