# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load USD prim trees with physics metadata for object_reference resolution."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.assets.object_type import ObjectType


@dataclass(frozen=True)
class UsdPrimRecord:
    """One prim inside a USD asset, keyed by default-prim-relative suffix."""

    relative_path: str
    object_type: ObjectType
    joint_names: tuple[str, ...] = ()


def _traverse_composed_stage(stage):
    """Traverse loaded prims, including prims beneath instance proxies."""
    from pxr import Usd

    stage.Load()
    return Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())


def find_nested_physics_roots(root_prim) -> dict[str, ObjectType]:
    """Return absolute paths of independently resettable physics roots below a prim."""
    from pxr import Usd

    from isaaclab_arena.utils.usd_helpers import is_articulation_root, is_rigid_body

    root_path = root_prim.GetPath()
    articulation_paths = set()
    roots: dict[str, ObjectType] = {}
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        prim_path = prim.GetPath()
        if is_articulation_root(prim):
            articulation_paths.add(prim_path)
        if prim == root_prim:
            continue
        if is_articulation_root(prim):
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


def load_usd_physics_roots(usd_path: str) -> dict[str, ObjectType]:
    """Return independently resettable physics roots in a composed USD asset.

    Articulation links are owned by their articulation root and are omitted.
    Rigid bodies outside articulation subtrees remain separate records, including
    bodies connected by ordinary USD joints.

    Args:
        usd_path: Local filesystem path or Nucleus/HTTPS URL for the USD.

    Returns:
        Physics object types keyed by sorted default-prim-relative paths.
    """
    from isaaclab.utils.assets import retrieve_file_path

    from isaaclab_arena.utils.usd_helpers import open_stage, relative_path_from_default_prim

    local_usd_path = retrieve_file_path(usd_path)
    roots: dict[str, ObjectType] = {}
    with open_stage(local_usd_path) as stage:
        stage.Load()
        for prim_path, object_type in find_nested_physics_roots(stage.GetDefaultPrim()).items():
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

    from isaaclab_arena.utils.usd_helpers import (
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
        # Collect prims that directly participate in physics or collision, then add
        # every ancestor so a prim is kept whenever any descendant is kept.
        # TODO(qianl): Ancestor-only prims are labeled base; non-leaf refs are valid today.
        # Revisit when relation solving adds descendant mesh exclusion; no issue observed yet.
        included_paths: set[str] = set()
        for prim in prims:
            if prim.IsPseudoRoot():
                continue
            if not has_physics_or_collision(prim):
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
            records.append(
                UsdPrimRecord(
                    relative_path=relative_path,
                    object_type=object_type,
                    joint_names=joint_names,
                )
            )
    records.sort(key=lambda record: record.relative_path)
    return records
