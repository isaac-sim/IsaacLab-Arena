# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Newton DisplayPort insertion environment ported from Shaurya's Isaac Lab task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from isaaclab_arena.assets.register import register_environment, register_retargeter
from isaaclab_arena.assets.retargeter_library import RetargetterBase
from isaaclab_arena.environments.arena_environment_factory import ArenaEnvironmentCfg, ArenaEnvironmentFactory

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import IsaacLabArenaManagerBasedRLEnvCfg


_DEFAULT_ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "local_stuff" / "connector_insertion" / "assets"
_PLUG_USD = "display_port_plug_newton_sdf.usda"
_SOCKET_USD = "display_port_socket_newton_sdf.usda"

_SDF_MAX_RESOLUTION = 256
_SDF_NARROW_BAND_RANGE = (-0.005, 0.005)
_SDF_PADDING = 0.005
_SDF_COLLIDER_COUNT = 6
_USE_SDF_COLLISION = True

# Geometry and grasp calibration copied from the validated upstream task.
_GEOMETRY_POS = (0.475, 0.125, 0.0375)
_SOCKET_ROT = (0.5, 0.5, 0.5, -0.5)
_PLUG_CLEARANCE_Z = 0.015
_SOCKET_INSERTION_OFFSET = (0.0375, 0.0, 0.0)
_PLUG_INSERTION_OFFSET = (0.0, 0.0, 0.0221)
_PLUG_GOAL_ROT = (0.0, -0.70711, 0.0, 0.70711)
_GRASP_OFFSET = (0.0025, 0.0, -0.1875)

_GRAV_GRIPPER_MIMIC_GEARING = {
    "finger_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "right_outer_knuckle_joint": 1.0,
    "left_outer_finger_joint": -1.0,
    "right_outer_finger_joint": -1.0,
}
_GRIPPER_OPEN_POSITION = 0.5
_GRIPPER_CLOSE_POSITION = -0.1
_GRIPPER_MAX_TARGET_STEP = 0.03


def _quat_rotate_vec(q_xyzw, vector):
    """Apply an XYZW quaternion rotation to a 3D vector."""
    qx, qy, qz, qw = q_xyzw
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def _quat_mul(q1_xyzw, q2_xyzw):
    """Multiply two XYZW quaternions."""
    x1, y1, z1, w1 = q1_xyzw
    x2, y2, z2, w2 = q2_xyzw
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def _compute_socket_root(geometry_pos, socket_rot):
    rotated = _quat_rotate_vec(socket_rot, _SOCKET_INSERTION_OFFSET)
    return tuple(geometry_pos[index] - rotated[index] for index in range(3))


def _compute_plug_pose(geometry_pos, socket_rot, z_clearance):
    plug_rot = _quat_mul(socket_rot, _PLUG_GOAL_ROT)
    plug_offset_world = _quat_rotate_vec(plug_rot, _PLUG_INSERTION_OFFSET)
    plug_root = (
        geometry_pos[0] - plug_offset_world[0],
        geometry_pos[1] - plug_offset_world[1],
        geometry_pos[2] - plug_offset_world[2] + z_clearance,
    )
    return plug_root, plug_rot


_SOCKET_ROOT = _compute_socket_root(_GEOMETRY_POS, _SOCKET_ROT)
_PLUG_ROOT, _PLUG_ROT = _compute_plug_pose(_GEOMETRY_POS, _SOCKET_ROT, _PLUG_CLEARANCE_Z)


@register_retargeter
class RizonDisplayportSpaceMouseRetargeter(RetargetterBase):
    """Use the native six-DoF SpaceMouse command directly with Rizon IK."""

    device = "spacemouse"
    embodiment = "rizon4s_grav_displayport_ik"

    def get_pipeline_builder(self, embodiment: object):
        return None


@dataclass
class DisplayportInsertionNewtonEnvironmentCfg(ArenaEnvironmentCfg):
    """Configure the validated Rizon/Newton DisplayPort insertion environment."""

    asset_directory: str | None = None
    """Directory containing the upstream DisplayPort Newton USD files."""

    teleop_device: str | None = "spacemouse"
    """Arena teleoperation device used by Isaac Lab's recorder."""


def _displayport_insertion_success(env, position_threshold: float = 0.003):
    """Return true when the plug and socket mating points are aligned."""
    import torch

    import isaaclab.utils.math as math_utils

    socket = env.scene["dp_socket"]
    plug = env.scene["dp_plug"]
    socket_pos = socket.data.root_link_pos_w.torch
    socket_quat = socket.data.root_link_quat_w.torch
    plug_pos = plug.data.root_link_pos_w.torch
    plug_quat = plug.data.root_link_quat_w.torch
    socket_offset = torch.tensor(_SOCKET_INSERTION_OFFSET, device=env.device).repeat(env.num_envs, 1)
    plug_offset = torch.tensor(_PLUG_INSERTION_OFFSET, device=env.device).repeat(env.num_envs, 1)
    socket_mate_pos, _ = math_utils.combine_frame_transforms(socket_pos, socket_quat, socket_offset)
    plug_mate_pos, _ = math_utils.combine_frame_transforms(plug_pos, plug_quat, plug_offset)
    return torch.linalg.vector_norm(socket_mate_pos - plug_mate_pos, dim=-1) < position_threshold


def _configure_displayport_newton_builder(_event_payload=None) -> None:
    """Restore MuJoCo attributes and build SDFs omitted by the non-replicated importer."""
    from isaaclab.sim import get_current_stage
    from isaaclab_newton.physics import NewtonManager
    from newton import GeoType, ShapeFlags
    from newton.solvers import SolverMuJoCo

    builder = NewtonManager._builder
    assert builder is not None, "Newton MODEL_INIT fired before the model builder was created"

    # Newton's non-replicated stage importer does not register MuJoCo custom
    # attributes before add_usd(), so authored mjc:gravcomp values are otherwise
    # lost even though they are present on every robot rigid-body prim.
    SolverMuJoCo.register_custom_attributes(builder)
    gravcomp = builder.custom_attributes["mujoco:gravcomp"]
    if gravcomp.values is None:
        gravcomp.values = {}
    stage = get_current_stage()
    for body_index, body_label in enumerate(builder.body_label):
        attribute = stage.GetPrimAtPath(body_label).GetAttribute("mjc:gravcomp")
        if attribute.IsValid() and attribute.Get() is not None:
            gravcomp.values[body_index] = float(attribute.Get())

    collider_count = 0
    for index, (label, source) in enumerate(zip(builder.shape_label, builder.shape_source)):
        is_plug_collider = label.endswith("/dp_plug/collision_mesh")
        is_socket_collider = "/dp_socket/tn__2584N111_DisplayportCord_jP/Body" in label and label.endswith("/Mesh")
        if not (is_plug_collider or is_socket_collider):
            continue
        if not builder.shape_flags[index] & ShapeFlags.COLLIDE_SHAPES:
            continue

        assert (
            builder.shape_type[index] == GeoType.MESH
        ), f"DisplayPort SDF collider was converted to geometry type {builder.shape_type[index]}: {label}"
        assert source is not None, f"DisplayPort SDF collider has no mesh source: {label}"
        collider_count += 1

        if source.sdf is None:
            scale = tuple(float(builder.shape_scale[index][axis]) for axis in range(3))
            source.build_sdf(
                device=NewtonManager.get_device(),
                narrow_band_range=_SDF_NARROW_BAND_RANGE,
                max_resolution=_SDF_MAX_RESOLUTION,
                margin=_SDF_PADDING,
                shape_margin=0.0,
                scale=scale,
                texture_format="uint16",
            )

    assert (
        collider_count == _SDF_COLLIDER_COUNT
    ), f"Expected {_SDF_COLLIDER_COUNT} DisplayPort SDF colliders, found {collider_count}"


def _use_cpu_fabric_hierarchy_sync(_env, _env_ids) -> None:
    """Use the compatible Fabric hierarchy path for rendered Newton transforms."""
    from isaaclab_newton.physics import NewtonManager

    # Isaac Sim 6.0 provides cubric IAdapter v0.2, but this Isaac Lab Newton
    # release calls it through a ctypes shim built for v0.1. The call updates
    # rigid-body matrices but leaves their visual descendants at identity. The
    # supported CPU fallback propagates the same matrices correctly and only
    # affects rendering, not simulation or collision behavior.
    if NewtonManager._cubric is not None and NewtonManager._cubric_adapter is not None:
        NewtonManager._cubric.release_adapter(NewtonManager._cubric_adapter)
    NewtonManager._cubric = None
    NewtonManager._cubric_adapter = None
    NewtonManager._cubric_bound_fabric_id = None
    NewtonManager._transforms_dirty = True


def _configure_newton_displayport_physics(
    env_cfg: IsaacLabArenaManagerBasedRLEnvCfg,
) -> IsaacLabArenaManagerBasedRLEnvCfg:
    """Apply the upstream Newton solver settings."""
    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg
    from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

    # Run collision at 200 Hz and the solver at 2 kHz while preserving the 35 ms control period.
    env_cfg.sim.dt = 1.0 / 200.0
    env_cfg.sim.render_interval = 7
    env_cfg.sim.physics_material = PhysxRigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
        restitution=0.0,
    )
    env_cfg.sim.physics = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=4096,
            nconmax=4096,
            impratio=10.0,
            cone="elliptic",
            iterations=100,
            ls_iterations=50,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(
            reduce_contacts=True,
            max_triangle_pairs=2**25,
        ),
        num_substeps=10,
        debug_mode=False,
    )
    env_cfg.decimation = 7
    # Isaac Lab 3.0 Beta 2's Newton replication path convex-hulls every mesh.
    # Import this single environment directly so the connector remains a triangle mesh.
    env_cfg.scene.replicate_physics = False
    if _USE_SDF_COLLISION:
        from isaaclab.physics import PhysicsEvent
        from isaaclab_newton.physics import NewtonManager

        NewtonManager.register_callback(
            _configure_displayport_newton_builder,
            PhysicsEvent.MODEL_INIT,
            name="displayport_newton_builder_compatibility",
            wrap_weak_ref=False,
        )
    return env_cfg


def _make_rizon_embodiment():
    """Build the upstream Rizon robot and a release-compatible IK action."""
    import math
    import torch

    import isaaclab.envs.mdp as mdp
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.controllers import DifferentialIKControllerCfg
    from isaaclab.envs.mdp.actions.actions_cfg import (
        BinaryJointPositionActionCfg,
        DifferentialInverseKinematicsActionCfg,
    )
    from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
    from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
    from isaaclab.managers import ActionTermCfg
    from isaaclab.managers import ObservationGroupCfg as ObsGroup
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_assets import FLEXIV_RIZON4S_GRAV_GRIPPER_CFG
    from isaaclab_physx.sim.schemas import PhysxArticulationRootPropertiesCfg, PhysxCollisionPropertiesCfg

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

    @configclass
    class RizonSceneCfg:
        """Rizon scene configuration."""

        robot: ArticulationCfg | None = None

    class PersistentTargetDifferentialIKAction(DifferentialInverseKinematicsAction):
        """Accumulate relative commands from the last target so zero input holds position."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._target_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        def process_actions(self, actions: torch.Tensor) -> None:
            self._raw_actions[:] = actions
            self._processed_actions[:] = self.raw_actions * self._scale
            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
                )

            ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
            use_target = self._target_initialized.unsqueeze(-1)
            command_base_pos = torch.where(use_target, self._ik_controller.ee_pos_des, ee_pos_curr)
            command_base_quat = torch.where(use_target, self._ik_controller.ee_quat_des, ee_quat_curr)
            self._ik_controller.set_command(self._processed_actions, command_base_pos, command_base_quat)
            self._target_initialized[:] = True

        def reset(self, env_ids=None) -> None:
            super().reset(env_ids)
            self._target_initialized[env_ids] = False

    class RateLimitedBinaryJointPositionAction(BinaryJointPositionAction):
        """Ramp binary gripper targets to avoid exciting the lightweight finger joints."""

        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._smoothed_target = self._close_command.unsqueeze(0).repeat(self.num_envs, 1)

        def process_actions(self, actions: torch.Tensor) -> None:
            super().process_actions(actions)
            target_delta = torch.clamp(
                self._processed_actions - self._smoothed_target,
                min=-_GRIPPER_MAX_TARGET_STEP,
                max=_GRIPPER_MAX_TARGET_STEP,
            )
            self._smoothed_target += target_delta
            self._processed_actions = self._smoothed_target

        def reset(self, env_ids=None) -> None:
            super().reset(env_ids)
            if env_ids is None:
                self._smoothed_target[:] = self._close_command
            else:
                self._smoothed_target[env_ids] = self._close_command

    @configclass
    class RizonActionsCfg:
        """Relative flange-pose action used by SpaceMouse teleoperation."""

        arm_action: ActionTermCfg = DifferentialInverseKinematicsActionCfg(
            class_type=PersistentTargetDifferentialIKAction,
            asset_name="robot",
            joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"],
            body_name="flange",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.01,
        )
        gripper_action: ActionTermCfg = BinaryJointPositionActionCfg(
            class_type=RateLimitedBinaryJointPositionAction,
            asset_name="robot",
            joint_names=list(_GRAV_GRIPPER_MIMIC_GEARING),
            open_command_expr={
                name: gearing * _GRIPPER_OPEN_POSITION for name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items()
            },
            close_command_expr={
                name: gearing * _GRIPPER_CLOSE_POSITION for name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items()
            },
        )

    @configclass
    class RizonObservationsCfg:
        """State observations recorded with demonstrations."""

        @configclass
        class PolicyCfg(ObsGroup):
            actions = ObsTerm(func=mdp.last_action)
            joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
            plug_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("dp_plug")})
            plug_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("dp_plug")})
            socket_pos = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("dp_socket")})
            socket_quat = ObsTerm(func=mdp.root_quat_w, params={"asset_cfg": SceneEntityCfg("dp_socket")})

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = False

        policy: PolicyCfg = PolicyCfg()

    class RizonDisplayportEmbodiment(EmbodimentBase):
        """Flexiv Rizon 4S with Grav gripper for DisplayPort insertion."""

        name = "rizon4s_grav_displayport_ik"
        default_arm_mode = ArmMode.SINGLE_ARM

        def __init__(self):
            super().__init__(enable_cameras=False)
            self.scene_config = RizonSceneCfg()
            self.action_config = RizonActionsCfg()
            self.observation_config = RizonObservationsCfg()

        def get_command_body_name(self) -> str:
            return "flange"

        def get_ee_frame_name(self, arm_mode: ArmMode) -> str:
            return "flange"

    embodiment = RizonDisplayportEmbodiment()
    robot_cfg = FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=FLEXIV_RIZON4S_GRAV_GRIPPER_CFG.spawn.replace(
            joint_drive_props=sim_utils.MujocoJointDrivePropertiesCfg(actuatorgravcomp=False),
            rigid_props=sim_utils.MujocoRigidBodyPropertiesCfg(gravcomp=1.0),
            articulation_props=PhysxArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
            ),
            collision_props=PhysxCollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": math.radians(32.44),
                "joint2": math.radians(-16.71),
                "joint3": math.radians(-5.69),
                "joint4": math.radians(128.38),
                "joint5": math.radians(6.74),
                "joint6": math.radians(55.95),
                "joint7": math.radians(111.54),
            },
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )
    robot_cfg.actuators["shoulder"].effort_limit_sim = 123.0
    robot_cfg.actuators["shoulder"].stiffness = 6000.0
    robot_cfg.actuators["shoulder"].damping = 108.5
    robot_cfg.actuators["elbow"].effort_limit_sim = 64.0
    robot_cfg.actuators["elbow"].stiffness = 4200.0
    robot_cfg.actuators["elbow"].damping = 90.7
    robot_cfg.actuators["wrist"].effort_limit_sim = 39.0
    robot_cfg.actuators["wrist"].stiffness = 1500.0
    robot_cfg.actuators["wrist"].damping = 54.2
    robot_cfg.actuators["gripper_drive"] = ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        effort_limit_sim=200.0,
        velocity_limit_sim=0.75,
        stiffness=2000.0,
        damping=50.0,
        friction=0.0,
        armature=0.1,
    )
    robot_cfg.actuators["gripper_passive"] = ImplicitActuatorCfg(
        joint_names_expr=[".*_knuckle_joint", ".*_outer_finger_joint"],
        effort_limit_sim=20.0,
        velocity_limit_sim=0.75,
        stiffness=2000.0,
        damping=50.0,
        friction=0.0,
        armature=0.05,
    )
    embodiment.scene_config.robot = robot_cfg
    return embodiment


def _make_displayport_task():
    """Build the reset event and minimal insertion task."""
    import torch
    from collections.abc import Callable

    import isaaclab.envs.mdp as mdp
    import isaaclab.utils.math as math_utils
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.envs.common import ViewerCfg
    from isaaclab.managers import EventTermCfg as EventTerm
    from isaaclab.managers import ManagerTermBase, SceneEntityCfg
    from isaaclab.managers import TerminationTermCfg as DoneTerm
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.utils.configclass import configclass
    from isaaclab_tasks.direct.automate import factory_control as fc

    from isaaclab_arena.embodiments.common.arm_mode import ArmMode
    from isaaclab_arena.tasks.task_base import TaskBase

    def set_finger_joint_pos_grav(
        joint_pos: torch.Tensor,
        reset_rows: list[int],
        finger_joints: list[int],
        finger_joint_position: float,
        joint_name_to_idx: dict[str, int] | None = None,
    ) -> None:
        del finger_joints
        assert joint_name_to_idx is not None, "Grav gripper reset requires a joint-name mapping"
        missing = [name for name in _GRAV_GRIPPER_MIMIC_GEARING if name not in joint_name_to_idx]
        assert not missing, f"Grav gripper joints not found on the robot: {missing}"
        for row in reset_rows:
            for joint_name, gearing in _GRAV_GRIPPER_MIMIC_GEARING.items():
                joint_pos[row, joint_name_to_idx[joint_name]] = gearing * finger_joint_position

    class SetRobotToObjectGraspPose(ManagerTermBase):
        """Set the Rizon to a calibrated plug grasp using reset-time IK."""

        def __init__(self, cfg: EventTerm, env: ManagerBasedEnv):
            super().__init__(cfg, env)
            self.robot: Articulation = env.scene[cfg.params["robot_asset_cfg"].name]
            self.target_object_name = cfg.params["target_object_name"]
            self.end_effector_body_name = cfg.params["end_effector_body_name"]
            self.num_arm_joints = cfg.params["num_arm_joints"]
            self.gripper_joint_setter_func: Callable = cfg.params["gripper_joint_setter_func"]
            self.grasp_offset = torch.tensor(cfg.params["grasp_offset"], device=env.device, dtype=torch.float32)
            self.grasp_rot_offset = torch.tensor(
                cfg.params["grasp_rot_offset"], device=env.device, dtype=torch.float32
            ).repeat(env.num_envs, 1)
            self.grasp_offsets = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
            self.hand_grasp_width = 0.3
            self.hand_hold_width = -0.1
            self.hand_close_width = -0.1

            eef_indices, _ = self.robot.find_bodies([self.end_effector_body_name])
            assert len(eef_indices) == 1, f"Expected one '{self.end_effector_body_name}' body, got {eef_indices}"
            self.eef_idx = eef_indices[0]
            self.jacobian_body_idx = self.eef_idx - 1
            all_joints, all_joint_names = self.robot.find_joints([".*"])
            self.all_joints = all_joints
            self.finger_joints = all_joints[self.num_arm_joints :]
            self.joint_name_to_idx = {name: index for index, name in zip(all_joints, all_joint_names)}

        def __call__(
            self,
            env: ManagerBasedEnv,
            env_ids: torch.Tensor,
            robot_asset_cfg: SceneEntityCfg,
            target_object_name: str,
            end_effector_body_name: str,
            num_arm_joints: int,
            grasp_offset: list[float],
            grasp_rot_offset: list[float],
            gripper_joint_setter_func: Callable,
            max_iterations: int = 150,
        ) -> None:
            del (
                robot_asset_cfg,
                target_object_name,
                end_effector_body_name,
                num_arm_joints,
                grasp_offset,
                grasp_rot_offset,
                gripper_joint_setter_func,
            )
            num_reset_envs = len(env_ids)
            target_object: RigidObject = env.scene[self.target_object_name]
            grasp_offsets = self.grasp_offsets[:num_reset_envs]
            grasp_rot_offset = self.grasp_rot_offset[env_ids]

            for _ in range(max_iterations):
                joint_pos = self.robot.data.joint_pos.torch[env_ids].clone()
                object_pos = target_object.data.root_link_pos_w.torch[env_ids]
                object_quat = target_object.data.root_link_quat_w.torch[env_ids]
                target_quat = math_utils.quat_mul(object_quat, grasp_rot_offset)
                grasp_offsets[:] = self.grasp_offset
                target_pos = object_pos + math_utils.quat_apply(target_quat, grasp_offsets)

                eef_pos = self.robot.data.body_pos_w.torch[env_ids, self.eef_idx]
                eef_quat = self.robot.data.body_quat_w.torch[env_ids, self.eef_idx]
                pos_error, axis_angle_error = fc.get_pose_error(
                    fingertip_midpoint_pos=eef_pos,
                    fingertip_midpoint_quat=eef_quat,
                    ctrl_target_fingertip_midpoint_pos=target_pos,
                    ctrl_target_fingertip_midpoint_quat=target_quat,
                    jacobian_type="geometric",
                    rot_error_type="axis_angle",
                )
                if torch.all(torch.linalg.vector_norm(pos_error, dim=-1) < 1.0e-6) and torch.all(
                    torch.linalg.vector_norm(axis_angle_error, dim=-1) < 1.0e-6
                ):
                    break

                delta_hand_pose = torch.cat((pos_error, axis_angle_error), dim=-1)
                jacobians = self.robot.data.body_link_jacobian_w.torch
                jacobian = jacobians[env_ids, self.jacobian_body_idx, :, self.robot.num_base_dofs :]
                delta_joint_pos = fc._get_delta_dof_pos(delta_hand_pose, "dls", jacobian, env.device)
                joint_pos += delta_joint_pos

                joint_limits = self.robot.data.joint_pos_limits.torch[env_ids, : self.num_arm_joints]
                joint_min = joint_limits[:, :, 0]
                joint_max = joint_limits[:, :, 1]
                joint_range = joint_max - joint_min
                finite_limits = torch.isfinite(joint_min) & torch.isfinite(joint_max) & torch.isfinite(joint_range)
                wrap_mask = finite_limits & (joint_range > 0)
                safe_joint_min = torch.where(wrap_mask, joint_min, torch.zeros_like(joint_min))
                safe_joint_range = torch.where(wrap_mask, joint_range, torch.ones_like(joint_range))
                arm_joint_pos = joint_pos[:, : self.num_arm_joints]
                wrapped_arm_joint_pos = safe_joint_min + torch.remainder(
                    arm_joint_pos - safe_joint_min, safe_joint_range
                )
                joint_pos[:, : self.num_arm_joints] = torch.where(wrap_mask, wrapped_arm_joint_pos, arm_joint_pos)
                joint_vel = torch.zeros_like(joint_pos)
                self.robot.set_joint_position_target_index(target=joint_pos, env_ids=env_ids)
                self.robot.set_joint_velocity_target_index(target=joint_vel, env_ids=env_ids)
                self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
                self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

            achieved_hand_pos = self.robot.data.body_pos_w.torch[env_ids, self.eef_idx].clone()
            achieved_hand_quat = self.robot.data.body_quat_w.torch[env_ids, self.eef_idx].clone()
            aligned_object_quat = math_utils.quat_mul(achieved_hand_quat, math_utils.quat_conjugate(grasp_rot_offset))
            grasp_offset_world = math_utils.quat_apply(achieved_hand_quat, grasp_offsets)
            aligned_object_pos = achieved_hand_pos - grasp_offset_world
            target_object.write_root_pose_to_sim_index(
                root_pose=torch.cat((aligned_object_pos, aligned_object_quat), dim=-1), env_ids=env_ids
            )
            target_object.write_root_velocity_to_sim_index(
                root_velocity=torch.zeros((num_reset_envs, 6), device=env.device), env_ids=env_ids
            )

            joint_pos = self.robot.data.joint_pos.torch[env_ids].clone()
            joint_vel = torch.zeros_like(joint_pos)
            reset_rows = list(range(num_reset_envs))
            self.gripper_joint_setter_func(
                joint_pos,
                reset_rows,
                self.finger_joints,
                self.hand_grasp_width,
                self.joint_name_to_idx,
            )
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
            self.gripper_joint_setter_func(
                joint_pos,
                reset_rows,
                self.finger_joints,
                self.hand_hold_width,
                self.joint_name_to_idx,
            )
            self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
            self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)
            self.gripper_joint_setter_func(
                joint_pos,
                reset_rows,
                self.finger_joints,
                self.hand_close_width,
                self.joint_name_to_idx,
            )
            self.robot.set_joint_position_target_index(target=joint_pos, joint_ids=self.all_joints, env_ids=env_ids)

    @configclass
    class EventsCfg:
        """DisplayPort grasp reset and robot material events."""

        use_cpu_fabric_hierarchy_sync = EventTerm(func=_use_cpu_fabric_hierarchy_sync, mode="startup")
        robot_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*finger.*"),
                "static_friction_range": (1.0, 1.0),
                "dynamic_friction_range": (1.0, 1.0),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 16,
            },
        )
        reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
        set_robot_to_grasp_pose = EventTerm(
            func=SetRobotToObjectGraspPose,
            mode="reset",
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "target_object_name": "dp_plug",
                "end_effector_body_name": "flange",
                "num_arm_joints": 7,
                "grasp_offset": list(_GRASP_OFFSET),
                "grasp_rot_offset": [0.0, 0.0, 0.0, 1.0],
                "gripper_joint_setter_func": set_finger_joint_pos_grav,
                "max_iterations": 150,
            },
        )

    @configclass
    class TerminationsCfg:
        """Minimal termination terms needed by the demo recorder."""

        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        success = DoneTerm(func=_displayport_insertion_success, params={"position_threshold": 0.003})

    class DisplayportInsertionTask(TaskBase):
        """Minimal task for manual Newton connector insertion."""

        def __init__(self):
            super().__init__(episode_length_s=120.0, task_description="Insert the DisplayPort plug into the socket.")
            self.scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
            self.events_cfg = EventsCfg()
            self.terminations_cfg = TerminationsCfg()

        def get_scene_cfg(self):
            return self.scene_cfg

        def get_termination_cfg(self):
            return self.terminations_cfg

        def get_events_cfg(self):
            return self.events_cfg

        def get_mimic_env_cfg(self, arm_mode: ArmMode):
            raise NotImplementedError("Mimic configuration is not part of this bring-up environment")

        def get_metrics(self):
            return []

        def get_viewer_cfg(self) -> ViewerCfg:
            return ViewerCfg(eye=(0.7, -0.45, 0.2975), lookat=(0.475, 0.125, 0.0675))

    return DisplayportInsertionTask()


@register_environment
class DisplayportInsertionNewtonEnvironment(ArenaEnvironmentFactory[DisplayportInsertionNewtonEnvironmentCfg]):
    """Build the validated Rizon/Newton DisplayPort insertion scene."""

    name = "displayport_insertion_newton"
    _legacy_argparse_cfg_type = DisplayportInsertionNewtonEnvironmentCfg

    def build(self, cfg: DisplayportInsertionNewtonEnvironmentCfg) -> IsaacLabArenaEnvironment:
        """Build a separate Newton environment without modifying ``connector_insertion``."""
        import isaaclab.sim as sim_utils
        from isaaclab.devices import Se3SpaceMouse, Se3SpaceMouseCfg
        from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

        from isaaclab_arena.assets.device_library import SpaceMouseCfg
        from isaaclab_arena.assets.object import Object
        from isaaclab_arena.assets.object_type import ObjectType
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        asset_directory = Path(cfg.asset_directory) if cfg.asset_directory is not None else _DEFAULT_ASSET_DIRECTORY
        plug_usd_path = asset_directory / _PLUG_USD
        socket_usd_path = asset_directory / _SOCKET_USD
        assert plug_usd_path.is_file(), f"DisplayPort plug Newton USD not found: {plug_usd_path}"
        assert socket_usd_path.is_file(), f"DisplayPort socket Newton USD not found: {socket_usd_path}"

        plug = Object(
            name="dp_plug",
            object_type=ObjectType.RIGID,
            usd_path=str(plug_usd_path),
            initial_pose=Pose(position_xyz=_PLUG_ROOT, rotation_xyzw=_PLUG_ROT),
            spawn_cfg_addon={
                "rigid_props": PhysxRigidBodyPropertiesCfg(
                    disable_gravity=False,
                    kinematic_enabled=False,
                    max_depenetration_velocity=0.5,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=128,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=None,
                ),
                "mass_props": sim_utils.MassPropertiesCfg(mass=0.03),
                "collision_props": PhysxCollisionPropertiesCfg(contact_offset=0.00001, rest_offset=-0.00005),
            },
        )
        socket = Object(
            name="dp_socket",
            object_type=ObjectType.RIGID,
            usd_path=str(socket_usd_path),
            initial_pose=Pose(position_xyz=_SOCKET_ROOT, rotation_xyzw=_SOCKET_ROT),
            spawn_cfg_addon={
                "rigid_props": PhysxRigidBodyPropertiesCfg(
                    disable_gravity=False,
                    kinematic_enabled=True,
                    max_depenetration_velocity=5.0,
                    linear_damping=0.0,
                    angular_damping=0.0,
                    max_linear_velocity=1000.0,
                    max_angular_velocity=3666.0,
                    enable_gyroscopic_forces=True,
                    solver_position_iteration_count=128,
                    solver_velocity_iteration_count=1,
                    max_contact_impulse=1e32,
                ),
                "collision_props": PhysxCollisionPropertiesCfg(contact_offset=0.0001, rest_offset=-0.0001),
            },
        )
        table = self.asset_registry.get_asset_by_name("table")()
        table.set_initial_pose(Pose(position_xyz=(0.5, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.707, 0.707)))
        ground = Object(
            name="ground",
            object_type=ObjectType.BASE,
            spawner_cfg=sim_utils.GroundPlaneCfg(),
            initial_pose=Pose(position_xyz=(0.0, 0.0, -1.05)),
        )
        light = self.asset_registry.get_asset_by_name("light")(
            spawner_cfg=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0)
        )

        embodiment = _make_rizon_embodiment()

        class InitiallyClosedSpaceMouse(Se3SpaceMouse):
            """SpaceMouse whose gripper toggle starts closed to preserve the reset grasp."""

            def __init__(self, cfg):
                super().__init__(cfg)
                self._close_gripper = True

            def reset(self):
                super().reset()
                self._close_gripper = True

        class DisplayportSpaceMouseCfg(SpaceMouseCfg):
            """SpaceMouse with a binary Grav gripper command."""

            def get_device_cfg(self, pipeline_builder=None, embodiment=None):
                return Se3SpaceMouseCfg(
                    pos_sensitivity=self.pos_sensitivity,
                    rot_sensitivity=self.rot_sensitivity,
                    gripper_term=True,
                    sim_device=self.sim_device,
                    class_type=InitiallyClosedSpaceMouse,
                )

        teleop_device = None
        if cfg.teleop_device is not None:
            assert cfg.teleop_device == "spacemouse", "This bring-up environment currently supports SpaceMouse"
            teleop_device = DisplayportSpaceMouseCfg(pos_sensitivity=0.15, rot_sensitivity=0.3)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[ground, table, plug, socket, light]),
            task=_make_displayport_task(),
            teleop_device=teleop_device,
            env_cfg_callback=_configure_newton_displayport_physics,
        )
