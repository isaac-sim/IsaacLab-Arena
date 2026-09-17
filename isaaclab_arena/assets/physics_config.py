# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Environment-owned physics overrides for selected parts of a USD asset."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.sim import UsdFileCfg
from isaaclab.sim.schemas import CollisionFragment, JointDriveFragment
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils.configclass import configclass


@configclass
class MujocoEqualityPropertiesCfg:
    """Tune an existing MuJoCo equality constraint without changing its joint relationship."""

    solref: tuple[float, float] | None = None
    """Constraint reference parameters: time constant and damping ratio, or negative direct format."""

    solimp: tuple[float, float, float, float, float] | None = None
    """Constraint impedance parameters: minimum, maximum, width, midpoint, and power."""


@configclass
class PrimPhysicsCfg:
    """Override physics on an explicitly selected asset-relative prim."""

    collision_props: list[CollisionFragment] | None = None
    """Collision fragments applied to this geometry prim; creates CollisionAPI when necessary."""

    physics_material: RigidBodyMaterialBaseCfg | None = None
    """Material created and bound only to this collider, leaving shared USD materials unchanged."""

    joint_drive_props: list[JointDriveFragment] | None = None
    """Joint-drive fragments applied to this revolute or prismatic joint."""

    mujoco_equality: MujocoEqualityPropertiesCfg | None = None
    """Response parameters for this prim's existing MjcEqualityJoint/Connect/WeldAPI."""

    filtered_pairs: list[str] = []
    """Asset-relative rigid-body or collider paths to add to this prim's collision exclusions."""


@configclass
class PhysicsUsdFileCfg(UsdFileCfg):
    """Spawn USD with environment-specific physics on named prims before cloning and model import."""

    func: Callable | str = "isaaclab_arena.assets.physics_spawner:spawn_usd_with_physics"

    prim_physics: dict[str, PrimPhysicsCfg] = {}
    """Exact asset-relative prim paths mapped to overrides; use '.' for the asset root.

    These overrides run after ordinary UsdFileCfg properties and before cloning. Instance proxies
    require the inherited make_uninstanceable option. No source USD or global builder is modified.
    """
