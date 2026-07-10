# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load USD prim trees with physics metadata for object_reference resolution."""

from __future__ import annotations

from dataclasses import dataclass

from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.utils.usd_helpers import (
    articulation_joint_names,
    has_physics_or_collision,
    object_type_for_prim,
    open_stage,
    relative_path_from_default_prim,
)


@dataclass(frozen=True)
class UsdPrimRecord:
    """One prim inside a USD asset, keyed by default-prim-relative suffix."""

    relative_path: str
    object_type: ObjectType
    joint_names: tuple[str, ...] = ()


def load_usd_prim_tree(usd_path: str) -> list[UsdPrimRecord]:
    """Return prim records for the physics/collision subtree of a USD asset.

    A prim is included when it directly participates in physics or collision, or
    when any of its descendants does. Retaining those ancestors keeps the returned
    records connected as a tree so the nested catalog can recover full paths.

    Args:
        usd_path: Filesystem path to the USD.

    Returns:
        Sorted list of :class:`UsdPrimRecord` entries keyed by relative_path suffix.
    """
    records: list[UsdPrimRecord] = []
    with open_stage(usd_path) as stage:
        # Collect prims that directly participate in physics or collision, then add
        # every ancestor so a prim is kept whenever any descendant is kept.
        # TODO(qianl): Ancestor-only prims are labeled base; non-leaf refs are valid today.
        # Revisit when relation solving adds descendant mesh exclusion; no issue observed yet.
        included_paths: set[str] = set()
        for prim in stage.Traverse():
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

        for prim in stage.Traverse():
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
