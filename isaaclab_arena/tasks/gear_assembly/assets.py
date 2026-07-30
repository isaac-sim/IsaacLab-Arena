# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Scene assets for the Arena Gear Assembly task."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_base import ObjectType
from isaaclab_arena.utils.pose import Pose

GEAR_ASSET_ROOT = f"{ISAAC_NUCLEUS_DIR}/Props/Factory/gear_assets"


class GearAssemblyRigidObject(Object):
    """Rigid object wrapper preserving the source task's contact-sensor setting."""

    def _generate_rigid_cfg(self) -> RigidObjectCfg:
        assert self.object_type == ObjectType.RIGID
        object_cfg = RigidObjectCfg(
            prim_path=self.prim_path,
            spawn=self._get_spawn_cfg(activate_contact_sensors=False),
            **self.asset_cfg_addon,
        )
        return self._add_initial_pose_to_cfg(object_cfg)


def _gear_spawn_cfg(kinematic_enabled: bool) -> dict:
    return {
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=kinematic_enabled,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=None),
        "collision_props": sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    }


def make_factory_gear(name: str, prim_name: str, usd_leaf: str, pose: Pose, kinematic_enabled: bool = False) -> Object:
    """Create one source-parity Factory gear rigid object."""
    gear = GearAssemblyRigidObject(
        name=name,
        prim_path=f"{{ENV_REGEX_NS}}/{prim_name}",
        object_type=ObjectType.RIGID,
        usd_path=f"{GEAR_ASSET_ROOT}/{usd_leaf}/{usd_leaf}.usd",
        initial_pose=pose,
        spawn_cfg_addon=_gear_spawn_cfg(kinematic_enabled=kinematic_enabled),
    )
    gear.disable_reset_pose()
    return gear


def make_factory_gear_base(pose: Pose) -> Object:
    """Create the kinematic source Factory gear base."""
    return make_factory_gear(
        name="factory_gear_base",
        prim_name="FactoryGearBase",
        usd_leaf="factory_gear_base",
        pose=pose,
        kinematic_enabled=True,
    )


def make_factory_gear_small(pose: Pose) -> Object:
    """Create the source small Factory gear."""
    return make_factory_gear("factory_gear_small", "FactoryGearSmall", "factory_gear_small", pose)


def make_factory_gear_medium(pose: Pose) -> Object:
    """Create the source medium Factory gear."""
    return make_factory_gear("factory_gear_medium", "FactoryGearMedium", "factory_gear_medium", pose)


def make_factory_gear_large(pose: Pose) -> Object:
    """Create the source large Factory gear."""
    return make_factory_gear("factory_gear_large", "FactoryGearLarge", "factory_gear_large", pose)


def make_ground() -> Object:
    """Create the source ground plane."""
    return Object(
        name="ground",
        prim_path="/World/ground",
        object_type=ObjectType.BASE,
        spawner_cfg=sim_utils.GroundPlaneCfg(),
        initial_pose=Pose(position_xyz=(0.0, 0.0, -1.05)),
    )


def make_stand() -> Object:
    """Create the source vertical stand asset."""
    return Object(
        name="stand",
        prim_path="{ENV_REGEX_NS}/Stand",
        object_type=ObjectType.BASE,
        spawner_cfg=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )
