# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned physics overrides for selected parts of a USD asset."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.spawners.from_files import spawn_from_usd
from isaaclab.utils.configclass import configclass
from pxr import Usd


@configclass
class UsdPrimSpawnPhysicsCfg:
    """Base interface for prim physics edits after USD loading, before cloning and physics import."""

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Check settings and targets without editing the stage; override when needed.

        Args:
            prim: Resolved, editable target prim.
            root: Spawned asset root for resolving any asset-relative relationships.
        """

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Apply physics settings after target validation, before cloning and physics import.

        Args:
            prim: Resolved, editable target prim.
            root: Spawned asset root for resolving any asset-relative relationships.
        """
        raise NotImplementedError("Concrete UsdPrimSpawnPhysicsCfg subclasses must implement apply().")


@configclass
class UsdFileCfgPrimPhysicsWrapper(UsdFileCfg):
    """Internal UsdFileCfg wrapper retaining per-prim settings and the original USD spawner.

    Isaac Lab's config copy keeps declared fields but drops attributes added only at runtime.
    """

    prim_physics: dict[str, UsdPrimSpawnPhysicsCfg] = {}
    """Exact asset-relative prim paths and overrides applied after USD loading, before cloning."""

    usd_spawn_func: Callable | str = spawn_from_usd
    """Original Lab's USD spawner; its single-prim body runs before physics edits."""
