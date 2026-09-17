# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Concrete test overrides, imported only after SimulationApp starts."""

import math

from isaaclab.utils.configclass import configclass
from pxr import UsdPhysics, UsdShade

from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg


@configclass
class MassCfg(UsdPrimSpawnPhysicsCfg):
    """Set mass on a rigid body to exercise a use-case-defined physics field."""

    mass: float = 0.25
    """Body mass in kilograms."""

    def validate_target(self, prim, root):
        """Require a rigid body and finite positive mass."""
        assert prim.HasAPI(UsdPhysics.RigidBodyAPI), "Mass target must be a rigid body."
        assert math.isfinite(self.mass) and self.mass > 0, "Mass must be finite and positive."

    def apply(self, prim, root):
        """Override the instance's mass."""
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(self.mass)


@configclass
class FrictionCfg(UsdPrimSpawnPhysicsCfg):
    """Bind a collider-local material without modifying shared source materials."""

    friction: float = 8.0
    """Static and dynamic friction coefficient."""

    def validate_target(self, prim, root):
        """Require a collider and a finite nonnegative coefficient."""
        assert prim.HasAPI(UsdPhysics.CollisionAPI), "Friction target must be a collider."
        assert math.isfinite(self.friction) and self.friction >= 0, "Friction must be finite and nonnegative."
        assert not prim.GetStage().GetPrimAtPath(prim.GetPath().AppendChild("TestPhysicsMaterial"))

    def apply(self, prim, root):
        """Create and bind an instance-local material whose path remaps during cloning."""
        material = UsdShade.Material.Define(prim.GetStage(), prim.GetPath().AppendChild("TestPhysicsMaterial"))
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(self.friction)
        physics.CreateDynamicFrictionAttr(self.friction)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            material, bindingStrength=UsdShade.Tokens.strongerThanDescendants, materialPurpose="physics"
        )
