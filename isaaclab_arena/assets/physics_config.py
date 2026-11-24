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
from pxr import Gf, Usd, UsdPhysics


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
class CollisionDisabledCfg(UsdPrimSpawnPhysicsCfg):
    """Turn off a collider."""

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Require a collider."""
        assert prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} is not a collider."

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Disable the collider."""
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)


@configclass
class RigidBodyMassCfg(UsdPrimSpawnPhysicsCfg):
    """Set a rigid body's mass, centre of mass and principal moments of inertia."""

    mass: float = 1.0
    """Mass [kg]."""

    center_of_mass: tuple[float, float, float] = (0.0, 0.0, 0.0)
    """Centre of mass in the prim's own coordinates, which a spawn scale scales [m]."""

    diagonal_inertia: tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Principal moments of inertia about the centre of mass along the prim's axes, unchanged by a spawn scale [kg m^2]."""

    def validate_target(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Require a rigid body."""
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI), f"{prim.GetPath()} is not a rigid body."

    def apply(self, prim: Usd.Prim, root: Usd.Prim) -> None:
        """Author the mass properties."""
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr(self.mass)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(*self.center_of_mass))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(*self.diagonal_inertia))
        mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))


@configclass
class UsdFileCfgPrimPhysicsWrapper(UsdFileCfg):
    """Internal UsdFileCfg wrapper retaining per-prim settings and the original USD spawner.

    Isaac Lab's config copy keeps declared fields but drops attributes added only at runtime.
    """

    prim_physics: dict[str, UsdPrimSpawnPhysicsCfg] = {}
    """Exact asset-relative prim paths and overrides applied after USD loading, before cloning."""

    usd_spawn_func: Callable | str = spawn_from_usd
    """Original Lab's USD spawner; its single-prim body runs before physics edits."""
