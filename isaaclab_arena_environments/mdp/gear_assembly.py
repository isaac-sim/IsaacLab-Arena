# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton and DROID configuration for the gear-assembly environment."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab_arena.embodiments.droid.droid import DroidEmbodimentBase
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


DROID_ARM_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
"""DROID arm joints in articulation order."""

DROID_GEAR_APPROACH_JOINT_POSITIONS = (
    1.4302369,
    -0.3727887,
    -1.0313725,
    -2.5031810,
    -0.4169911,
    2.2608662,
    0.4437634,
)
"""Stable initial arm pose above the source gear."""

DROID_GRIPPER_MIMIC_SIGNS = {
    "finger_joint": 1.0,
    "left_inner_finger_joint": -1.0,
    "left_inner_finger_knuckle_joint": -1.0,
    "right_outer_knuckle_joint": 1.0,
    "right_inner_finger_joint": 1.0,
    "right_inner_finger_knuckle_joint": -1.0,
}
"""Joint directions for the mechanically coupled Robotiq fingers."""


def configure_droid_for_newton_gear_assembly(embodiment: DroidEmbodimentBase) -> None:
    """Apply the Newton controller, actuator, and end-effector configuration to DROID."""
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
    from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
    from isaaclab_newton.sim.schemas import NewtonMaterialPropertiesCfg

    robot_cfg = deepcopy(embodiment.scene_config.robot)
    embodiment.scene_config.robot = robot_cfg
    robot_cfg.spawn.func = _get_newton_droid_spawn()
    robot_cfg.spawn.make_uninstanceable = True
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.spawn.physics_material = NewtonMaterialPropertiesCfg(
        static_friction=3.0,
        dynamic_friction=3.0,
        restitution=0.0,
    )
    robot_cfg.init_state.joint_pos.update(
        dict(zip(DROID_ARM_JOINT_NAMES, DROID_GEAR_APPROACH_JOINT_POSITIONS, strict=True))
    )

    gripper_joint_names = tuple(DROID_GRIPPER_MIMIC_SIGNS)
    robot_cfg.actuators["gripper"] = ImplicitActuatorCfg(
        joint_names_expr=list(gripper_joint_names),
        effort_limit=20.0,
        velocity_limit=1.2,
        stiffness=40.0,
        damping=8.0,
        armature=0.05,
    )

    embodiment.action_config.arm_action = deepcopy(embodiment.action_config.arm_action)
    embodiment.action_config.arm_action.body_name = "base_link"
    embodiment.action_config.arm_action.body_offset = None
    arm_controller = embodiment.action_config.arm_action.controller
    arm_controller.ik_method = "adaptive_dls"
    arm_controller.ik_params = {
        "lambda_min": 0.05,
        "lambda_max": 0.20,
        "sigma_thresh": 0.02,
    }
    arm_controller.joint_limit_avoidance_gain = 0.10
    arm_controller.joint_limit_avoidance_margin = 0.35

    open_command = dict.fromkeys(gripper_joint_names, 0.0)
    close_command = {name: sign * 0.461 for name, sign in DROID_GRIPPER_MIMIC_SIGNS.items()}
    embodiment.action_config.gripper_action = BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(gripper_joint_names),
        open_command_expr=open_command,
        close_command_expr=close_command,
    )

    embodiment.scene_config.ee_frame = deepcopy(embodiment.scene_config.ee_frame)
    target_frame = embodiment.scene_config.ee_frame.target_frames[0]
    target_frame.prim_path = "{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link"
    target_frame.offset = OffsetCfg()


def gear_assembly_newton_env_cfg_callback(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Configure Newton for a stable 30 Hz contact-rich manipulation loop."""
    from isaaclab_newton.physics import HydroelasticSDFCfg, NewtonCollisionPipelineCfg

    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import ArenaPhysicsCfg

    env_cfg.sim.dt = 1.0 / 240.0
    env_cfg.decimation = 8
    env_cfg.sim.render_interval = 8
    env_cfg.sim.physics = ArenaPhysicsCfg().newton
    env_cfg.sim.physics.num_substeps = 4
    env_cfg.sim.physics.default_shape_cfg.gap = 0.0
    env_cfg.sim.physics.solver_cfg.ls_iterations = 50
    env_cfg.sim.physics.solver_cfg.ccd_iterations = 35
    env_cfg.sim.physics.solver_cfg.njmax = 4096
    env_cfg.sim.physics.solver_cfg.nconmax = 4096
    env_cfg.sim.physics.collision_cfg = NewtonCollisionPipelineCfg(
        sdf_hydroelastic_config=HydroelasticSDFCfg(reduce_contacts=True, normal_matching=True)
    )
    env_cfg.scene.env_spacing = 1.5
    env_cfg.scene.replicate_physics = True
    env_cfg.events.randomize_franka_joint_state = None
    return env_cfg


_NEWTON_DROID_SPAWN = None


def _get_newton_droid_spawn():
    """Return the cached DROID spawner with the current USD compatibility repair."""
    global _NEWTON_DROID_SPAWN
    if _NEWTON_DROID_SPAWN is None:
        from isaaclab.sim import schemas
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        from isaaclab.sim.utils import clone
        from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

        @clone
        def spawn_newton_droid(
            prim_path: str,
            spawner_cfg,
            translation: tuple[float, float, float] | None = None,
            orientation: tuple[float, float, float, float] | None = None,
            **kwargs,
        ):
            prim = spawn_from_usd(
                prim_path,
                spawner_cfg,
                translation=translation,
                orientation=orientation,
                **kwargs,
            )
            _promote_droid_collision_meshes(prim)
            schemas.modify_rigid_body_properties(
                prim_path,
                MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
                prim.GetStage(),
            )
            return prim

        _NEWTON_DROID_SPAWN = spawn_newton_droid
    return _NEWTON_DROID_SPAWN


def _promote_droid_collision_meshes(root_prim) -> None:
    """Expose source Robotiq collision meshes to Newton without changing geometry."""
    from pxr import Usd, UsdGeom, UsdPhysics

    collision_groups = [
        candidate
        for candidate in Usd.PrimRange(root_prim)
        if candidate.HasAPI(UsdPhysics.CollisionAPI) and not candidate.IsA(UsdGeom.Gprim)
    ]
    for collision_group in collision_groups:
        collision_api = UsdPhysics.CollisionAPI(collision_group)
        collision_enabled = collision_api.GetCollisionEnabledAttr().Get()
        collision_enabled = True if collision_enabled is None else collision_enabled
        source_approximation = None
        if collision_group.HasAPI(UsdPhysics.MeshCollisionAPI):
            source_approximation = UsdPhysics.MeshCollisionAPI(collision_group).GetApproximationAttr().Get()

        meshes = [candidate for candidate in Usd.PrimRange(collision_group) if candidate.IsA(UsdGeom.Mesh)]
        assert meshes, f"Collision group '{collision_group.GetPath()}' has no source mesh"
        for mesh in meshes:
            mesh_collision_api = (
                UsdPhysics.CollisionAPI(mesh)
                if mesh.HasAPI(UsdPhysics.CollisionAPI)
                else UsdPhysics.CollisionAPI.Apply(mesh)
            )
            mesh_collision_api.CreateCollisionEnabledAttr().Set(collision_enabled)
            mesh_approximation_api = (
                UsdPhysics.MeshCollisionAPI(mesh)
                if mesh.HasAPI(UsdPhysics.MeshCollisionAPI)
                else UsdPhysics.MeshCollisionAPI.Apply(mesh)
            )
            mesh_approximation_api.CreateApproximationAttr().Set(source_approximation or "convexHull")

        collision_api.CreateCollisionEnabledAttr().Set(False)
