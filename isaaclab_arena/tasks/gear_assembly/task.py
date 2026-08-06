# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Arena task config for source-parity Gear Assembly."""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp
from isaaclab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.events import randomize_gear_type, set_robot_to_grasp_pose
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.noise_models import (
    ResetSampledConstantNoiseModelCfg,
    ResetSampledQuaternionNoiseModelCfg,
)
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.observations import (
    gear_pos_w,
    gear_quat_w,
    gear_shaft_pos_w,
    gear_shaft_quat_w,
)
from isaaclab_tasks.manager_based.manipulation.deploy.mdp.terminations import (
    reset_when_gear_dropped,
    reset_when_gear_orientation_exceeds_threshold,
)

from isaaclab_arena.assets.register import register_task
from isaaclab_arena.embodiments.droid.observations import gripper_pos as droid_gripper_pos
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.gear_assembly import rewards as gear_rewards
from isaaclab_arena.tasks.gear_assembly.events import (
    randomize_gears_and_base_pose_with_inactive_gear_parking,
    set_newton_rigid_body_material,
)
from isaaclab_arena.tasks.gear_assembly.specs import (
    GEAR_ASSEMBLED_ANGULAR_VELOCITY_THRESHOLD,
    GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS,
    GEAR_ASSEMBLED_LINEAR_VELOCITY_THRESHOLD,
    GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE,
    GEAR_ASSEMBLED_SUPPORT_Z_OFFSET,
    GEAR_ASSEMBLED_SUPPORT_Z_THRESHOLD,
    GEAR_ASSEMBLED_UPRIGHT_AXIS_THRESHOLD_DEG,
    GEAR_ASSEMBLED_XY_THRESHOLD,
    GEAR_ASSEMBLED_Z_THRESHOLD,
    GEAR_OFFSETS,
    GEAR_POSE_RANGE,
    GEAR_TABLETOP_ORIENTATION_XYZW,
    GEAR_TABLETOP_PARKING_POSITIONS,
    GEAR_TYPES,
    NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE,
    NEWTON_GEAR_ASSEMBLED_SUPPORT_Z_OFFSET,
    NEWTON_GEAR_OFFSETS,
    NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW,
    NEWTON_GEAR_TABLETOP_PARKING_POSITIONS,
    SELECTED_GEAR_POS_RANGE,
    GearAssemblyMode,
    GearAssemblyRobotSpec,
)
from isaaclab_arena.tasks.gear_assembly.terminations import selected_gear_on_base
from isaaclab_arena.tasks.task_base import TaskBase


@register_task
class GearAssemblyTask(TaskBase):
    """Source-parity Gear Assembly task implemented through Arena composition."""

    DEFAULT_EPISODE_LENGTH_S = 6.66

    def __init__(
        self,
        robot_spec: GearAssemblyRobotSpec,
        mode: GearAssemblyMode = "play",
        newton_backend: bool = False,
    ):
        super().__init__(episode_length_s=self.DEFAULT_EPISODE_LENGTH_S, task_description="Assemble the gear.")
        self.robot_spec = robot_spec
        self.mode = mode
        self.gear_offsets = NEWTON_GEAR_OFFSETS if newton_backend else GEAR_OFFSETS
        parking_positions = (
            NEWTON_GEAR_TABLETOP_PARKING_POSITIONS if newton_backend else GEAR_TABLETOP_PARKING_POSITIONS
        )
        tabletop_orientation = (
            NEWTON_GEAR_TABLETOP_ORIENTATION_XYZW if newton_backend else GEAR_TABLETOP_ORIENTATION_XYZW
        )
        assembled_root_z = (
            NEWTON_GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE if newton_backend else GEAR_ASSEMBLED_ROOT_Z_ABOVE_BASE
        )
        support_z_offset = NEWTON_GEAR_ASSEMBLED_SUPPORT_Z_OFFSET if newton_backend else GEAR_ASSEMBLED_SUPPORT_Z_OFFSET
        root_xy_offsets = (
            self.gear_offsets if newton_backend else {gear_type: [0.0, 0.0, 0.0] for gear_type in GEAR_TYPES}
        )
        self.observation_cfg = ObservationsCfg(robot_spec=robot_spec, mode=mode, gear_offsets=self.gear_offsets)
        self.events_cfg = EventsCfg(
            robot_spec=robot_spec,
            mode=mode,
            newton_backend=newton_backend,
            parking_positions=parking_positions,
            tabletop_orientation=tabletop_orientation,
        )
        self.rewards_cfg = RewardsCfg(robot_spec=robot_spec)
        self.termination_cfg = TerminationsCfg(
            robot_spec=robot_spec,
            mode=mode,
            assembled_root_z=assembled_root_z,
            root_xy_offsets=root_xy_offsets,
            support_z_offset=support_z_offset,
            base_support_prim_name="platform" if newton_backend else None,
        )

    def get_scene_cfg(self) -> Any:
        return None

    def get_termination_cfg(self) -> Any:
        return self.termination_cfg

    def get_events_cfg(self) -> Any:
        return self.events_cfg

    def get_observation_cfg(self) -> Any:
        return self.observation_cfg

    def get_rewards_cfg(self) -> Any:
        return self.rewards_cfg

    def get_mimic_env_cfg(self, arm_mode) -> Any:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return []

    def runtime_env_attrs(self) -> dict[str, Any]:
        """Attributes required by source Gear Assembly manager terms at env construction."""
        return {
            "gear_offsets": self.gear_offsets,
            "gear_offsets_grasp": self.robot_spec.gear_offsets_grasp,
            "hand_grasp_width": self.robot_spec.hand_grasp_width,
            "hand_close_width": self.robot_spec.hand_close_width,
            "end_effector_body_name": self.robot_spec.end_effector_body_name,
            "num_arm_joints": self.robot_spec.num_arm_joints,
            "grasp_rot_offset": self.robot_spec.grasp_rot_offset,
            "gripper_joint_setter_func": self.robot_spec.gripper_joint_setter_func,
            "joint_action_scale": 0.025,
        }


@configclass
class ObservationsCfg:
    """Observation terms for Gear Assembly."""

    @configclass
    class PolicyCfg(ObservationGroupCfg):
        joint_pos: ObservationTermCfg = MISSING
        gripper_pos: ObservationTermCfg | None = None
        joint_vel: ObservationTermCfg = MISSING
        gear_shaft_pos: ObservationTermCfg = MISSING
        gear_shaft_quat: ObservationTermCfg = MISSING

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationGroupCfg):
        joint_pos: ObservationTermCfg = MISSING
        joint_vel: ObservationTermCfg = MISSING
        gear_shaft_pos: ObservationTermCfg = MISSING
        gear_shaft_quat: ObservationTermCfg = MISSING
        gear_pos: ObservationTermCfg = MISSING
        gear_quat: ObservationTermCfg = MISSING

    policy: PolicyCfg = MISSING
    critic: CriticCfg = MISSING

    def __init__(
        self,
        robot_spec: GearAssemblyRobotSpec,
        mode: GearAssemblyMode,
        gear_offsets: dict[str, list[float]],
    ):
        joint_cfg = SceneEntityCfg("robot", joint_names=robot_spec.joint_names)
        self.policy = self.PolicyCfg(
            joint_pos=ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": joint_cfg}),
            joint_vel=ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": joint_cfg}),
            gear_shaft_pos=ObservationTermCfg(
                func=gear_shaft_pos_w,
                params={"gear_offsets": gear_offsets},
                noise=ResetSampledConstantNoiseModelCfg(
                    noise_cfg=UniformNoiseCfg(n_min=-0.01, n_max=0.01, operation="add")
                ),
            ),
            gear_shaft_quat=ObservationTermCfg(
                func=gear_shaft_quat_w,
                noise=ResetSampledQuaternionNoiseModelCfg(
                    roll_range=(-0.03491, 0.03491),
                    pitch_range=(-0.03491, 0.03491),
                    yaw_range=(-0.03491, 0.03491),
                ),
            ),
        )
        if mode == "play":
            self.policy.enable_corruption = False
            self.policy.concatenate_terms = False
            self.policy.gripper_pos = ObservationTermCfg(func=droid_gripper_pos)
        self.critic = self.CriticCfg(
            joint_pos=ObservationTermCfg(func=mdp.joint_pos, params={"asset_cfg": joint_cfg}),
            joint_vel=ObservationTermCfg(func=mdp.joint_vel, params={"asset_cfg": joint_cfg}),
            gear_shaft_pos=ObservationTermCfg(func=gear_shaft_pos_w, params={"gear_offsets": gear_offsets}),
            gear_shaft_quat=ObservationTermCfg(func=gear_shaft_quat_w),
            gear_pos=ObservationTermCfg(func=gear_pos_w),
            gear_quat=ObservationTermCfg(func=gear_quat_w),
        )


@configclass
class EventsCfg:
    """Reset and startup events for Gear Assembly."""

    robot_joint_stiffness_and_damping: EventTermCfg | None = None
    joint_friction: EventTermCfg | None = None
    small_gear_physics_material: EventTermCfg = MISSING
    medium_gear_physics_material: EventTermCfg = MISSING
    large_gear_physics_material: EventTermCfg = MISSING
    gear_base_physics_material: EventTermCfg = MISSING
    robot_physics_material: EventTermCfg = MISSING
    randomize_gear_type: EventTermCfg = MISSING
    reset_all: EventTermCfg = MISSING
    randomize_gears_and_base_pose: EventTermCfg = MISSING
    set_robot_to_grasp_pose: EventTermCfg = MISSING

    def __init__(
        self,
        robot_spec: GearAssemblyRobotSpec,
        mode: GearAssemblyMode,
        newton_backend: bool,
        parking_positions: dict[str, tuple[float, float, float]],
        tabletop_orientation: tuple[float, float, float, float],
    ):
        self.robot_joint_stiffness_and_damping = None
        self.joint_friction = None
        if robot_spec.reset_randomizes_robot:
            self.robot_joint_stiffness_and_damping = EventTermCfg(
                func=mdp.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
                    "stiffness_distribution_params": (0.75, 1.5),
                    "damping_distribution_params": (0.3, 3.0),
                    "operation": "scale",
                    "distribution": "log_uniform",
                },
            )
            self.joint_friction = EventTermCfg(
                func=mdp.randomize_joint_parameters,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*", "wrist_.*"]),
                    "friction_distribution_params": (0.3, 0.7),
                    "operation": "add",
                    "distribution": "uniform",
                },
            )

        self.small_gear_physics_material = _material_event(
            "factory_gear_small", ".*", robot_spec.startup_materials, newton_backend
        )
        self.medium_gear_physics_material = _material_event(
            "factory_gear_medium", ".*", robot_spec.startup_materials, newton_backend
        )
        self.large_gear_physics_material = _material_event(
            "factory_gear_large", ".*", robot_spec.startup_materials, newton_backend
        )
        self.gear_base_physics_material = _material_event(
            "factory_gear_base", ".*", robot_spec.startup_materials, newton_backend
        )
        self.robot_physics_material = _material_event(
            "robot", ".*finger.*", robot_spec.startup_materials, newton_backend
        )

        gear_types = list(GEAR_TYPES)
        pose_range = dict(GEAR_POSE_RANGE)
        selected_gear_pos_range = dict(SELECTED_GEAR_POS_RANGE)

        self.randomize_gear_type = EventTermCfg(
            func=randomize_gear_type,
            mode="reset",
            params={"gear_types": gear_types},
        )
        self.reset_all = EventTermCfg(func=mdp.reset_scene_to_default, mode="reset")
        self.randomize_gears_and_base_pose = EventTermCfg(
            func=randomize_gears_and_base_pose_with_inactive_gear_parking,
            mode="reset",
            params={
                "pose_range": pose_range,
                "gear_pos_range": selected_gear_pos_range,
                "parking_positions": parking_positions,
                "parking_orientation_xyzw": tabletop_orientation,
                "velocity_range": {},
            },
        )
        self.set_robot_to_grasp_pose = EventTermCfg(
            func=set_robot_to_grasp_pose,
            mode="reset",
            params={
                "robot_asset_cfg": SceneEntityCfg("robot"),
                # The source reset term samples this range on every IK iteration. Keep play resets deterministic
                # so the solver converges to one grasp target; randomized training retains source behavior.
                "pos_randomization_range": None if mode == "play" else robot_spec.set_grasp_pos_randomization_range,
                "gear_offsets_grasp": robot_spec.gear_offsets_grasp,
                "end_effector_body_name": robot_spec.end_effector_body_name,
                "num_arm_joints": robot_spec.num_arm_joints,
                "grasp_rot_offset": robot_spec.grasp_rot_offset,
                "gripper_joint_setter_func": robot_spec.gripper_joint_setter_func,
            },
        )


@configclass
class RewardsCfg:
    """Reward terms for Gear Assembly."""

    end_effector_gear_keypoint_tracking: RewardTermCfg = MISSING
    end_effector_gear_keypoint_tracking_exp: RewardTermCfg = MISSING
    end_effector_base_keypoint_tracking: RewardTermCfg = MISSING
    end_effector_base_keypoint_tracking_exp: RewardTermCfg = MISSING
    action_rate: RewardTermCfg = MISSING

    def __init__(self, robot_spec: GearAssemblyRobotSpec):
        self.end_effector_gear_keypoint_tracking = RewardTermCfg(
            func=gear_rewards.keypoint_entity_error,
            weight=-1.5,
            params={"asset_cfg_1": SceneEntityCfg("factory_gear_base"), "keypoint_scale": 0.15},
        )
        self.end_effector_gear_keypoint_tracking_exp = RewardTermCfg(
            func=gear_rewards.keypoint_entity_error_exp,
            weight=1.5,
            params={
                "asset_cfg_1": SceneEntityCfg("factory_gear_base"),
                "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
                "kp_use_sum_of_exps": False,
                "keypoint_scale": 0.15,
            },
        )
        ee_params = {
            "robot_asset_cfg": SceneEntityCfg("robot"),
            "keypoint_scale": 0.15,
            "ee_gear_threshold": 0.0,
            "weight_ramp_start": 0.0,
            "weight_ramp_steps": 512_000,
            "end_effector_body_name": robot_spec.end_effector_body_name,
            "grasp_rot_offset": robot_spec.grasp_rot_offset,
            "gear_offsets_grasp": robot_spec.gear_offsets_grasp,
        }
        self.end_effector_base_keypoint_tracking = RewardTermCfg(
            func=gear_rewards.keypoint_ee_gear_error,
            weight=-0.5,
            params=dict(ee_params),
        )
        self.end_effector_base_keypoint_tracking_exp = RewardTermCfg(
            func=gear_rewards.keypoint_ee_gear_error_exp,
            weight=0.5,
            params={
                **ee_params,
                "kp_exp_coeffs": [(50, 0.0001), (300, 0.0001)],
                "kp_use_sum_of_exps": False,
            },
        )
        self.action_rate = RewardTermCfg(func=mdp.action_rate_l2, weight=-5.0e-06)


@configclass
class TerminationsCfg:
    """Termination terms for Gear Assembly."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp.time_out, time_out=True)
    success: TerminationTermCfg | None = None
    gear_dropped: TerminationTermCfg = MISSING
    gear_orientation_exceeded: TerminationTermCfg = MISSING

    def __init__(
        self,
        robot_spec: GearAssemblyRobotSpec,
        mode: GearAssemblyMode,
        assembled_root_z: dict[str, float],
        root_xy_offsets: dict[str, list[float]],
        support_z_offset: dict[str, float],
        base_support_prim_name: str | None,
    ):
        self.time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
        self.success = None
        if mode == "play":
            self.success = TerminationTermCfg(
                func=selected_gear_on_base,
                params={
                    "base_asset_cfg": SceneEntityCfg("factory_gear_base"),
                    "root_z_above_base": assembled_root_z,
                    "root_xy_offset_from_base": root_xy_offsets,
                    "xy_threshold": GEAR_ASSEMBLED_XY_THRESHOLD,
                    "z_threshold": GEAR_ASSEMBLED_Z_THRESHOLD,
                    "upright_axis_threshold_deg": GEAR_ASSEMBLED_UPRIGHT_AXIS_THRESHOLD_DEG,
                    "linear_velocity_threshold": GEAR_ASSEMBLED_LINEAR_VELOCITY_THRESHOLD,
                    "angular_velocity_threshold": GEAR_ASSEMBLED_ANGULAR_VELOCITY_THRESHOLD,
                    "support_z_offset": support_z_offset,
                    "base_support_prim_name": base_support_prim_name,
                    "support_z_threshold": GEAR_ASSEMBLED_SUPPORT_Z_THRESHOLD,
                    "consecutive_success_steps": GEAR_ASSEMBLED_CONSECUTIVE_SUCCESS_STEPS,
                },
            )
        self.gear_dropped = TerminationTermCfg(
            func=reset_when_gear_dropped,
            params={
                "distance_threshold": 0.15,
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "gear_offsets_grasp": robot_spec.gear_offsets_grasp,
                "end_effector_body_name": robot_spec.end_effector_body_name,
                "grasp_rot_offset": robot_spec.grasp_rot_offset,
            },
        )
        self.gear_orientation_exceeded = TerminationTermCfg(
            func=reset_when_gear_orientation_exceeds_threshold,
            params={
                "roll_threshold_deg": 15.0,
                "pitch_threshold_deg": 15.0,
                "yaw_threshold_deg": 180.0,
                "robot_asset_cfg": SceneEntityCfg("robot"),
                "end_effector_body_name": robot_spec.end_effector_body_name,
                "grasp_rot_offset": robot_spec.grasp_rot_offset,
            },
        )


def _material_event(
    asset_name: str,
    body_names: str,
    materials: dict[str, tuple[float, float, float]],
    newton_backend: bool,
) -> EventTermCfg:
    static_friction, dynamic_friction, restitution = materials[asset_name]
    if newton_backend:
        return EventTermCfg(
            func=set_newton_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(asset_name, body_names=body_names),
                "static_friction": static_friction,
                "restitution": restitution,
            },
        )
    return EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(asset_name, body_names=body_names),
            "static_friction_range": (static_friction, static_friction),
            "dynamic_friction_range": (dynamic_friction, dynamic_friction),
            "restitution_range": (restitution, restitution),
            "num_buckets": 16,
        },
    )
