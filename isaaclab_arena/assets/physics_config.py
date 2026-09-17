# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned physics overrides for selected parts of a USD asset."""

from __future__ import annotations

from isaaclab.sim import UsdFileCfg
from isaaclab.utils.configclass import configclass
from pxr import Usd


@configclass
class UsdPrimSpawnPhysicsCfg:
    """Base interface for USD-prim physics edits during spawning, before cloning/import.

    Define concrete configclass subclasses alongside the environment or embodiment that
    needs them. Core does not prescribe physics fields or backend-specific schema APIs.
    Runtime variations such as object mass act per reset; build-time variations can
    configure this hook to apply sampled physics once USD prims exist.
    """

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Check settings and targets without editing the stage; override when needed.

        Args:
            prim: Resolved, editable target prim.
            root: Spawned asset root for resolving any asset-relative relationships.
        """

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Author physics after target validation, before cloning and physics import.

        Args:
            prim: Resolved, editable target prim.
            root: Spawned asset root for resolving any asset-relative relationships.
        """
        raise NotImplementedError("Concrete UsdPrimSpawnPhysicsCfg subclasses must implement apply().")


@configclass
class UsdFileCfgPrimPhysicsWrapper(UsdFileCfg):
    """Internal UsdFileCfg wrapper adding prim_physics storage that survives config copying."""

    prim_physics: dict[str, UsdPrimSpawnPhysicsCfg] = {}
    """Exact asset-relative prim paths and overrides applied after USD loading, before cloning."""
