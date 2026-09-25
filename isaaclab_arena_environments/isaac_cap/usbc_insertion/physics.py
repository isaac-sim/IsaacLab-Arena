# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP USB-C asset contacts, solver cleanup, and actuator tuning."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from isaaclab.sim import bind_physics_material
from isaaclab.sim.schemas import CollisionFragment, UsdPhysicsCollisionCfg, apply_collision_properties
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg, spawn_physics_material
from isaaclab.sim.utils import use_stage
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import NewtonMJWarpManager
from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg, NewtonMaterialPropertiesCfg
from pxr import Sdf, UsdGeom, UsdPhysics, Vt

from isaaclab_arena.assets.physics_config import UsdPrimSpawnPhysicsCfg

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg

_SOLREF = (0.004, 1.0)
_SOLIMP = (0.95, 0.999, 0.0005, 0.5, 2.0)
_LINK_6 = "Geometry/arm/link_1/link_2/link_3/link_4/link_5/link_6"
_FINGER_SHAPES = (
    "Capsule",
    "Capsule_1",
    "Capsule_2",
    "Box",
    "Box_1",
    "Sphere",
    "Sphere_1",
    "Sphere_2",
    "Sphere_3",
    "Sphere_4",
    "Sphere_5",
)


@configclass
class UsbcContactCfg(UsdPrimSpawnPhysicsCfg):
    """Apply USB-C task contact schemas to one asset-relative prim."""

    collision_props: list[CollisionFragment] = []
    """Collision schema fragments authored on the target."""

    physics_material: RigidBodyMaterialBaseCfg | None = None
    """Optional collider-local Newton material."""

    equality_solref: tuple[float, float] | None = None
    """Optional MuJoCo equality response parameters."""

    def validate_target(self, prim, root) -> None:
        """Validate the selected collider or equality prim before authoring."""
        if self.collision_props:
            assert prim.IsA(UsdGeom.Gprim), f"Collision target must be geometry: {prim.GetPath()}"
            assert all(isinstance(fragment, CollisionFragment) for fragment in self.collision_props)
        if self.physics_material is not None:
            assert (
                prim.HasAPI(UsdPhysics.CollisionAPI) or self.collision_props
            ), f"Physics material target must be a collider: {prim.GetPath()}"
            material_path = prim.GetPath().AppendChild("UsbcPhysicsMaterial")
            assert not prim.GetStage().GetPrimAtPath(material_path), f"Physics material already exists: {material_path}"
        if self.equality_solref is not None:
            assert any(
                schema in prim.GetAppliedSchemas()
                for schema in ("MjcEqualityJointAPI", "MjcEqualityConnectAPI", "MjcEqualityWeldAPI")
            ), f"Equality target has no MuJoCo equality schema: {prim.GetPath()}"
            assert len(self.equality_solref) == 2 and all(
                math.isfinite(value) for value in self.equality_solref
            ), f"Equality solref must contain two finite values: {prim.GetPath()}"

    def apply(self, prim, root) -> None:
        """Author collision, material, and equality properties on the selected prim."""
        stage = prim.GetStage()
        prim_path = str(prim.GetPath())
        if self.collision_props:
            assert apply_collision_properties(
                prim_path, self.collision_props, stage
            ), f"Failed to apply collision properties to {prim_path}"
        if self.physics_material is not None:
            material_path = f"{prim_path}/UsbcPhysicsMaterial"
            with use_stage(stage):
                spawn_physics_material(material_path, self.physics_material)
            bind_physics_material(prim_path, material_path, stage=stage)
        if self.equality_solref is not None:
            prim.CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set(Vt.DoubleArray(self.equality_solref))


def make_robot_spawn_cfg_addon() -> dict[str, dict]:
    """Build per-instance full-hand contacts for both YAM robots."""
    return {
        "left_robot": {"make_uninstanceable": True, "prim_physics": _robot_prim_physics()},
        "right_robot": {"make_uninstanceable": True, "prim_physics": _robot_prim_physics()},
    }


def _robot_prim_physics() -> dict[str, UsbcContactCfg]:
    """Build CAP's full-hand contacts and passive-jaw response for one YAM."""
    overrides = {}
    finger_roots = (
        f"{_LINK_6}/link_left_finger/lf_rot/lf_down",
        f"{_LINK_6}/link_right_finger/rf_rot/rf_down",
    )
    for root in finger_roots:
        for shape_name in _FINGER_SHAPES:
            overrides[f"{root}/{shape_name}"] = UsbcContactCfg(
                collision_props=[
                    NewtonCollisionCfg(contact_gap=0.0002),
                    MujocoCollisionCfg(condim=4, solref=_SOLREF, solimp=_SOLIMP),
                ],
                physics_material=NewtonMaterialPropertiesCfg(
                    static_friction=8.0,
                    dynamic_friction=8.0,
                    torsional_friction=0.002,
                    rolling_friction=0.0001,
                ),
            )
    for shape_name in ("Capsule", "Capsule_1", "Capsule_2"):
        overrides[f"{_LINK_6}/{shape_name}"] = UsbcContactCfg(
            collision_props=[
                UsdPhysicsCollisionCfg(collision_enabled=True),
                MujocoCollisionCfg(condim=3, solref=_SOLREF, solimp=_SOLIMP),
            ]
        )
    overrides[f"{_LINK_6}/link_left_finger/left_finger"] = UsbcContactCfg(equality_solref=_SOLREF)
    return overrides


def connector_prim_physics(relative_path: str, friction: float) -> dict[str, UsbcContactCfg]:
    """Build matched connector contact properties for one collider."""
    return {
        relative_path: UsbcContactCfg(
            collision_props=[MujocoCollisionCfg(solref=_SOLREF, solimp=_SOLIMP)],
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=friction,
                dynamic_friction=friction,
                contact_stiffness=62500.0,
                contact_damping=500.0,
            ),
        )
    }


def friction_prim_physics(relative_path: str, friction: float) -> dict[str, UsbcContactCfg]:
    """Build a collider-local friction material for one task fixture."""
    return {
        relative_path: UsbcContactCfg(
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=friction,
                dynamic_friction=friction,
            )
        )
    }


class NewtonUsbcManager(NewtonMJWarpManager):
    """Clean up the task's procedural cable hooks during manager teardown."""

    @classmethod
    def _solver_specific_clear(cls) -> None:
        from .cables import _remove_connector_cable_builder_hooks

        _remove_connector_cable_builder_hooks()
        super()._solver_specific_clear()


def configure_usbc_runtime(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
    *,
    apply_graph_override: Callable[[IsaacLabArenaManagerBasedRLEnvCfg], IsaacLabArenaManagerBasedRLEnvCfg],
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply the graph override, cable cleanup manager, and actuator tuning."""
    env_cfg = apply_graph_override(env_cfg)
    env_cfg.sim.physics.class_type = NewtonUsbcManager
    env_cfg.scene.replicate_physics = False
    for robot in (env_cfg.scene.left_robot, env_cfg.scene.right_robot):
        robot.init_state.joint_pos.update(joint2=1.047, joint3=1.047)
        for name, actuator in robot.actuators.items():
            if name.startswith("arm_"):
                actuator.stiffness = 1600.0
                actuator.damping = 70.0
                actuator.effort_limit_sim = 28.0 if name == "arm_joints_1_3" else 10.0
        robot.actuators["gripper"].stiffness = 40000.0
        robot.actuators["gripper"].damping = 40.0
        robot.actuators["gripper"].effort_limit_sim = 160.0
    return env_cfg
