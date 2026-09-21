# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CAP USB-C solver, asset-scoped contacts, and actuator tuning."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from copy import deepcopy
from typing import TYPE_CHECKING

from isaaclab_newton.physics import NewtonMJWarpManager

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


def _robot_prim_physics():
    """Build CAP's full-hand contact and passive-jaw overrides for one YAM."""
    from isaaclab.sim.schemas import UsdPhysicsCollisionCfg
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonCollisionCfg, NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.physics_config import MujocoEqualityPropertiesCfg, PrimPhysicsCfg

    overrides = {}
    finger_roots = (
        f"{_LINK_6}/link_left_finger/lf_rot/lf_down",
        f"{_LINK_6}/link_right_finger/rf_rot/rf_down",
    )
    for root in finger_roots:
        for shape_name in _FINGER_SHAPES:
            overrides[f"{root}/{shape_name}"] = PrimPhysicsCfg(
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
        overrides[f"{_LINK_6}/{shape_name}"] = PrimPhysicsCfg(
            collision_props=[
                UsdPhysicsCollisionCfg(collision_enabled=True),
                MujocoCollisionCfg(condim=3, solref=_SOLREF, solimp=_SOLIMP),
            ]
        )
    overrides[f"{_LINK_6}/link_left_finger/left_finger"] = PrimPhysicsCfg(
        mujoco_equality=MujocoEqualityPropertiesCfg(solref=_SOLREF)
    )
    return overrides


def _connector_prim_physics(relative_path: str, friction: float):
    """Build matched connector contact properties for one collider."""
    from isaaclab_newton.sim.schemas import MujocoCollisionCfg, NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.physics_config import PrimPhysicsCfg

    return {
        relative_path: PrimPhysicsCfg(
            collision_props=[MujocoCollisionCfg(solref=_SOLREF, solimp=_SOLIMP)],
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=friction,
                dynamic_friction=friction,
                contact_stiffness=62500.0,
                contact_damping=500.0,
            ),
        )
    }


def _friction_prim_physics(relative_path: str, friction: float):
    """Build a local friction material for one task fixture collider."""
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    from isaaclab_arena.assets.physics_config import PrimPhysicsCfg

    return {
        relative_path: PrimPhysicsCfg(
            physics_material=NewtonMaterialPropertiesCfg(
                static_friction=friction,
                dynamic_friction=friction,
            )
        )
    }


def _with_prim_physics(spawn, prim_physics):
    """Copy an existing USD spawn config and add task-scoped prim physics."""
    from isaaclab.sim import UsdFileCfg

    from isaaclab_arena.assets.physics_config import PhysicsUsdFileCfg

    assert isinstance(spawn, UsdFileCfg), f"USB-C physics requires a USD spawn config, got {type(spawn).__name__}."
    tuned = PhysicsUsdFileCfg(usd_path=spawn.usd_path, prim_physics=prim_physics)
    for field in dataclasses.fields(UsdFileCfg):
        if field.name != "func":
            setattr(tuned, field.name, deepcopy(getattr(spawn, field.name)))
    tuned.make_uninstanceable = True
    return tuned


def _apply_asset_physics(env_cfg: IsaacLabArenaManagerBasedRLEnvCfg) -> None:
    """Replace task asset spawners with isolated USB-C physics overrides."""
    robot_physics = _robot_prim_physics()
    for robot in (env_cfg.scene.left_robot, env_cfg.scene.right_robot):
        robot.spawn = _with_prim_physics(robot.spawn, deepcopy(robot_physics))

    env_cfg.scene.plug.spawn = _with_prim_physics(
        env_cfg.scene.plug.spawn,
        _connector_prim_physics("ArtistFrame/SourceCollisionMesh", friction=0.35),
    )
    env_cfg.scene.bench.spawn = _with_prim_physics(
        env_cfg.scene.bench.spawn,
        _friction_prim_physics("Geometry", friction=0.4),
    )
    if getattr(env_cfg.scene, "port", None) is not None:
        env_cfg.scene.port.spawn = _with_prim_physics(
            env_cfg.scene.port.spawn,
            _connector_prim_physics("Geometry", friction=0.35),
        )
    if getattr(env_cfg.scene, "bulkhead", None) is not None:
        env_cfg.scene.bulkhead.spawn = _with_prim_physics(
            env_cfg.scene.bulkhead.spawn,
            _connector_prim_physics("Geometry/bulkhead_01_obj_00/SourceCollisionMesh", friction=2.5),
        )


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
    """Apply graph and asset physics overrides, then install runtime-only tuning."""
    env_cfg = apply_graph_override(env_cfg)
    _apply_asset_physics(env_cfg)
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
