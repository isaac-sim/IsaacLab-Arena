# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Dexsuite-aligned procedural assets (table + lift object) for Newton / Arena environments."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim import CuboidCfg, RigidBodyMaterialCfg

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.utils.pose import Pose


# Kinematic table from Isaac Lab Dexsuite (``dexsuite_env_cfg.TABLE_SPAWN_CFG``).
_DEXSUITE_TABLE_SPAWN_CFG = sim_utils.CuboidCfg(
    size=(0.8, 1.5, 0.04),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visible=False,
)

# Single cuboid preset used with Newton (no multi-asset spawner); matches ``ObjectCfg.cube`` in Dexsuite.
_DEXSUITE_LIFT_CUBE_SPAWN_CFG = CuboidCfg(
    size=(0.05, 0.1, 0.1),
    physics_material=RigidBodyMaterialCfg(static_friction=0.5),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=0,
        disable_gravity=False,
    ),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
)


@register_asset
class DexsuiteManipTable(Object):
    """Kinematic cuboid table matching Isaac Lab Dexsuite (asset name ``table`` for command visuals)."""

    name = "dexsuite_manip_table"
    tags = ["background", "dexsuite"]

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        resolved_name = instance_name if instance_name is not None else "table"
        resolved_prim = prim_path if prim_path is not None else "{ENV_REGEX_NS}/table"
        pose = initial_pose or Pose(
            position_xyz=(-0.55, 0.0, 0.235),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        # ``usd_path`` is required by ``Object`` for RIGID; unused when spawn is procedural (``object_type`` is fixed).
        super().__init__(
            name=resolved_name,
            prim_path=resolved_prim,
            object_type=ObjectType.RIGID,
            usd_path="",
            initial_pose=pose,
        )
        self.disable_reset_pose()

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=_DEXSUITE_TABLE_SPAWN_CFG,
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)


@register_asset
class DexsuiteLiftManipObject(Object):
    """Lift-task manipuland: procedural cuboid matching Dexsuite ``ObjectCfg.cube`` (Newton-safe, single geometry)."""

    name = "dexsuite_lift_object"
    tags = ["object", "dexsuite"]

    def __init__(
        self,
        instance_name: str | None = None,
        prim_path: str | None = None,
        initial_pose: Pose | None = None,
    ):
        resolved_name = instance_name if instance_name is not None else "object"
        resolved_prim = prim_path if prim_path is not None else "{ENV_REGEX_NS}/Object"
        pose = initial_pose or Pose(
            position_xyz=(-0.55, 0.1, 0.35),
            rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )
        super().__init__(
            name=resolved_name,
            prim_path=resolved_prim,
            object_type=ObjectType.RIGID,
            usd_path="",
            initial_pose=pose,
        )
        self.disable_reset_pose()

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=_DEXSUITE_LIFT_CUBE_SPAWN_CFG,
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(cfg)
