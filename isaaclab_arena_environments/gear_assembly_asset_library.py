# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Local gear-assembly assets awaiting publication to Arena's asset service."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.utils.pose import Pose

_ASSET_DIR = Path(__file__).with_name("gear_assembly_assets")
_NEAR_ZERO_NEWTON_FRICTION = 1.0e-4
"""Small supported friction for the otherwise frictionless fixed base."""


def _make_spawn_cfg_addon(*, mass: float, kinematic: bool, friction: float) -> dict:
    """Build the shared Newton rigid-body configuration for a Factory gear asset."""
    import isaaclab.sim as sim_utils
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    return {
        "make_uninstanceable": True,
        "rigid_props": sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            kinematic_enabled=kinematic,
            linear_damping=0.05,
            angular_damping=0.10,
            max_depenetration_velocity=1.0,
        ),
        "mass_props": sim_utils.MassPropertiesCfg(mass=mass),
        "collision_props": sim_utils.CollisionPropertiesCfg(
            contact_offset=1.0e-4,
            rest_offset=0.0,
        ),
        "physics_material": NewtonMaterialPropertiesCfg(
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
    }


def make_gear_assembly_base(initial_pose: Pose) -> Object:
    """Create the fixed gear base from its local prototype asset."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType

    return Object(
        name="gear_assembly_base",
        object_type=ObjectType.RIGID,
        usd_path=str(_ASSET_DIR / "factory_gear_base.usda"),
        initial_pose=initial_pose,
        tags=["object", "gear", "gear_assembly"],
        spawn_cfg_addon=_make_spawn_cfg_addon(
            mass=0.05,
            kinematic=True,
            friction=_NEAR_ZERO_NEWTON_FRICTION,
        ),
    )


def make_gear_assembly_medium_gear(initial_pose: Pose) -> Object:
    """Create the medium gear from its local prototype asset."""
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_base import ObjectType

    return Object(
        name="gear_assembly_medium_gear",
        object_type=ObjectType.RIGID,
        usd_path=str(_ASSET_DIR / "factory_gear_medium.usda"),
        initial_pose=initial_pose,
        tags=["object", "gear", "gear_assembly"],
        spawn_cfg_addon=_make_spawn_cfg_addon(mass=0.019, kinematic=False, friction=3.0),
    )
