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
    """Return prim records for every prim in a USD asset.

    Args:
        usd_path: Filesystem path to the USD.

    Returns:
        Sorted list of :class:`UsdPrimRecord` entries keyed by relative_path suffix.
    """
    records: list[UsdPrimRecord] = []
    with open_stage(usd_path) as stage:
        for prim in stage.Traverse():
            if prim.IsPseudoRoot():
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
