# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""USD spawning helpers for interactive nested background physics."""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.utils import clone

if TYPE_CHECKING:
    from pxr import Usd


def deinstance_nested_physics(prim: Usd.Prim) -> tuple[str, ...]:
    """De-instance subtrees that contribute dynamic physics prims."""
    from pxr import Usd, UsdPhysics

    instance_roots: dict[str, Usd.Prim] = {}
    for candidate in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not (candidate.HasAPI(UsdPhysics.RigidBodyAPI) or candidate.HasAPI(UsdPhysics.ArticulationRootAPI)):
            continue
        ancestor = candidate
        while ancestor.IsValid() and ancestor.GetPath().HasPrefix(prim.GetPath()):
            if ancestor.IsInstance():
                instance_roots[str(ancestor.GetPath())] = ancestor
                break
            if ancestor == prim:
                break
            ancestor = ancestor.GetParent()

    deinstanced_paths = []
    for path, instance_root in sorted(instance_roots.items()):
        assert instance_root.SetInstanceable(False), f"Failed to de-instance nested physics subtree '{path}'"
        deinstanced_paths.append(path)
    return tuple(deinstanced_paths)


@clone
def spawn_from_usd_with_resettable_nested_physics(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Spawn a USD background and materialize instance-proxy physics subtrees."""
    prim = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    deinstance_nested_physics(prim)
    return prim
