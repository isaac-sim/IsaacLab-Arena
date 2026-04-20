# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple gear insertion task for the assembled NIST board.

The medium gear is a separate object that the robot must insert
onto the peg.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Callable
from dataclasses import MISSING, dataclass, field

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import gear_insertion_observations
from isaaclab_arena.tasks.observations.gear_insertion_observations import body_pos_in_env_frame, body_quat_canonical
from isaaclab_arena.tasks.rewards import gear_insertion_rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import gear_dropped_from_gripper, gear_mesh_insertion_success
from isaaclab_arena.terms.events import place_gear_in_gripper
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object


@dataclass
class GraspConfig:
    """Configuration for placing the gear in the robot's gripper at reset.

    Groups all embodiment-specific grasp parameters so the task constructor
    stays focused on task-level concerns (geometry, success criteria).
    """

    num_arm_joints: int = 7
    hand_grasp_width: float = 0.03
    hand_close_width: float = 0.0
    gripper_joint_setter_func: Callable | None = None
    end_effector_body_name: str = "panda_hand"
    # xyzw identity; the environment overrides with task-specific orientation
    grasp_rot_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    grasp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    arm_joint_names: str = "panda_joint.*"
    finger_body_names: str = ".*finger"


class NistGearInsertionTask(TaskBase):
    """Gear insertion task: insert the medium gear onto the peg.

    Peg position is computed from a fixed asset (board or separate gear-base USD)
    plus an offset in that asset's local frame (assembly peg-insert convention).
    """

    def __init__(
        self,
        assembled_board: Asset,
        held_gear: Asset,
        background_scene: Asset,
        peg_offset_from_board: list[float] | None = None,
        peg_offset_for_obs: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        gear_base_asset: Asset | None = None,
        # Success geometry: Z threshold = gear_peg_height * success_z_fraction
        # e.g. 0.02 * 0.20 = 4 mm.  Tune these together.
        gear_peg_height: float = 0.02,
        success_z_fraction: float = 0.30,
        xy_threshold: float = 0.0025,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        grasp_cfg: GraspConfig | None = None,
        enable_randomization: bool = False,
        peg_offset_xy_noise: float = 0.005,
        disable_drop_terminations: bool = True,
        rl_training_mode: bool = False,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.assembled_board = assembled_board
        self.held_gear = held_gear
        self.background_scene = background_scene
        self._gear_base_asset = gear_base_asset if gear_base_asset is not None else assembled_board
        self.peg_offset_from_board = peg_offset_from_board or [2.025e-2, 0.0, 0.0]
        self.peg_offset_for_obs = peg_offset_for_obs
        self.held_gear_base_offset = (
            held_gear_base_offset if held_gear_base_offset is not None else [2.025e-2, 0.0, 0.0]
        )
        self.peg_offset_xy_noise = peg_offset_xy_noise
        self.gear_peg_height = gear_peg_height
        self.success_z_fraction = success_z_fraction
        self.xy_threshold = xy_threshold
        self.grasp_cfg = grasp_cfg
        self.enable_randomization = enable_randomization
        self.disable_drop_terminations = disable_drop_terminations
        self.rl_training_mode = rl_training_mode
        self.task_description = (
            f"Insert the {held_gear.name} onto the gear base on the {assembled_board.name}"
            if task_description is None
            else task_description
        )

    def get_scene_cfg(self):
        return None

    def get_observation_cfg(self):
        peg_obs = self.peg_offset_for_obs if self.peg_offset_for_obs is not None else self.peg_offset_from_board
        return _GearInsertionObservationsCfg(
            gear_name=self.held_gear.name,
            board_name=self._gear_base_asset.name,
            peg_offset=peg_obs,
            held_gear_base_offset=self.held_gear_base_offset,
        )

    def get_rewards_cfg(self):
        return _GearInsertionRewardsCfg(
            gear_name=self.held_gear.name,
            board_name=self._gear_base_asset.name,
            peg_offset=self.peg_offset_from_board,
            held_gear_base_offset=self.held_gear_base_offset,
            gear_peg_height=self.gear_peg_height,
            success_z_fraction=self.success_z_fraction,
            xy_threshold=self.xy_threshold,
            peg_offset_xy_noise=self.peg_offset_xy_noise,
        )

    def get_termination_cfg(self):
        success = TerminationTermCfg(
            func=gear_mesh_insertion_success,
            params={
                "held_object_cfg": SceneEntityCfg(self.held_gear.name),
                "fixed_object_cfg": SceneEntityCfg(self._gear_base_asset.name),
                "gear_base_offset": self.peg_offset_from_board,
                "held_gear_base_offset": self.held_gear_base_offset,
                "gear_peg_height": self.gear_peg_height,
                "success_z_fraction": self.success_z_fraction,
                "xy_threshold": self.xy_threshold,
                "rl_training": self.rl_training_mode,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.held_gear.name),
            },
        )
        cfg = _TerminationsCfg(success=success, object_dropped=object_dropped)
        if self.grasp_cfg is not None:
            cfg.gear_dropped_from_gripper = TerminationTermCfg(
                func=gear_dropped_from_gripper,
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_body_name": self.grasp_cfg.end_effector_body_name,
                    "distance_threshold": 0.15,
                },
            )
        if self.disable_drop_terminations:
            cfg.object_dropped = None
            cfg.gear_dropped_from_gripper = None
        return cfg

    def get_events_cfg(self):
        cfg = _EventsCfg()
        gc = self.grasp_cfg
        if gc is not None and gc.gripper_joint_setter_func is not None:
            cfg.place_gear = EventTermCfg(
                func=place_gear_in_gripper,
                mode="reset",
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "num_arm_joints": gc.num_arm_joints,
                    "hand_grasp_width": gc.hand_grasp_width,
                    "hand_close_width": gc.hand_close_width,
                    "gripper_joint_setter_func": gc.gripper_joint_setter_func,
                    "end_effector_body_name": gc.end_effector_body_name,
                    "grasp_rot_offset": gc.grasp_rot_offset,
                    "grasp_offset": gc.grasp_offset,
                },
            )
        if self.enable_randomization:
            arm_joints = gc.arm_joint_names if gc is not None else "panda_joint.*"
            finger_bodies = gc.finger_body_names if gc is not None else ".*finger"
            cfg.held_physics_material = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(self.held_gear.name),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )
            cfg.robot_physics_material = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 1,
                },
            )
            cfg.fixed_physics_material = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(self._gear_base_asset.name),
                    "static_friction_range": (0.25, 1.25),
                    "dynamic_friction_range": (0.25, 0.25),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 128,
                },
            )
            cfg.held_object_mass = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self.held_gear.name),
                    "mass_distribution_params": (-0.005, 0.005),
                    "operation": "add",
                    "distribution": "uniform",
                },
            )
            cfg.fixed_asset_pose = EventTermCfg(
                func=mdp_isaac_lab.reset_root_state_uniform,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg(self._gear_base_asset.name),
                    "pose_range": {
                        "x": (0.0, 0.0),
                        "y": (0.0, 0.0),
                        "z": (0.0, 0.0),
                        "roll": (0.0, 0.0),
                        "pitch": (0.0, 0.0),
                        "yaw": (0.0, 0.2617993877991494),
                    },
                    "velocity_range": {},
                },
            )
            cfg.robot_actuator_gains = EventTermCfg(
                func=mdp_isaac_lab.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints),
                    "stiffness_distribution_params": (0.75, 1.5),
                    "damping_distribution_params": (0.3, 3.0),
                    "operation": "scale",
                    "distribution": "log_uniform",
                },
            )
            cfg.robot_joint_friction = EventTermCfg(
                func=mdp_isaac_lab.randomize_joint_parameters,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=arm_joints),
                    "friction_distribution_params": (0.3, 0.7),
                    "operation": "add",
                    "distribution": "uniform",
                },
            )
            cfg.gear_physics_material = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(self.held_gear.name, body_names=".*"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )
            cfg.board_physics_material = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg(self._gear_base_asset.name, body_names=".*"),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )
            cfg.robot_finger_physics_material = EventTermCfg(
                func=mdp_isaac_lab.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=finger_bodies),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )
        return cfg

    def get_prompt(self):
        raise NotImplementedError("Function not implemented yet.")

    def get_mimic_env_cfg(self, embodiment_name: str):
        raise NotImplementedError("Function not implemented yet.")

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_gear,
            offset=np.array([1.5, -0.5, 1.0]),
        )


@configclass
class _TerminationsCfg:
    """Termination terms for the gear insertion task."""

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg | None = MISSING
    gear_dropped_from_gripper: TerminationTermCfg | None = None


@configclass
class _EventsCfg:
    """Events: reset to default poses plus optional deploy-style randomization."""

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    place_gear: EventTermCfg | None = None
    fixed_asset_pose: EventTermCfg | None = None
    held_physics_material: EventTermCfg | None = None
    robot_physics_material: EventTermCfg | None = None
    fixed_physics_material: EventTermCfg | None = None
    held_object_mass: EventTermCfg | None = None
    gripper_init_yaw_noise: EventTermCfg | None = None
    robot_actuator_gains: EventTermCfg | None = None
    robot_joint_friction: EventTermCfg | None = None
    gear_physics_material: EventTermCfg | None = None
    board_physics_material: EventTermCfg | None = None
    robot_finger_physics_material: EventTermCfg | None = None
    held_asset_pos_noise: EventTermCfg | None = None


@configclass
class _GearInsertionObservationsCfg:
    """Task-specific observations for the gear insertion task."""

    task_obs: ObsGroup = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float],
        held_gear_base_offset: list[float] | None = None,
    ):
        hgo = held_gear_base_offset if held_gear_base_offset is not None else [2.025e-2, 0.0, 0.0]

        @configclass
        class _TaskObsCfg(ObsGroup):
            gear_pos = ObsTerm(
                func=mdp_isaac_lab.root_pos_w,
                params={"asset_cfg": SceneEntityCfg(gear_name)},
            )
            gear_quat = ObsTerm(
                func=mdp_isaac_lab.root_quat_w,
                params={"make_quat_unique": True, "asset_cfg": SceneEntityCfg(gear_name)},
            )
            peg_pos = ObsTerm(
                func=gear_insertion_observations.peg_pos_in_env_frame,
                params={"board_cfg": SceneEntityCfg(board_name), "peg_offset": peg_offset},
            )
            board_quat = ObsTerm(
                func=mdp_isaac_lab.root_quat_w,
                params={"make_quat_unique": True, "asset_cfg": SceneEntityCfg(board_name)},
            )
            peg_delta = ObsTerm(
                func=gear_insertion_observations.peg_delta_from_held_gear_base,
                params={
                    "gear_cfg": SceneEntityCfg(gear_name),
                    "board_cfg": SceneEntityCfg(board_name),
                    "peg_offset": peg_offset,
                    "held_gear_base_offset": hgo,
                },
            )
            joint_pos = ObsTerm(
                func=mdp_isaac_lab.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            joint_vel = ObsTerm(
                func=mdp_isaac_lab.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg("robot")},
            )
            ee_pos_noiseless = ObsTerm(
                func=body_pos_in_env_frame,
                params={"body_name": "panda_fingertip_centered"},
            )
            ee_quat_noiseless = ObsTerm(
                func=body_quat_canonical,
                params={"body_name": "panda_fingertip_centered"},
            )

            def __post_init__(self):
                self.enable_corruption = False
                self.concatenate_terms = True

        self.task_obs = _TaskObsCfg()


@configclass
class _GearInsertionRewardsCfg:
    """Keypoint squashing, bonuses, and insertion regularisers for gear insertion.

    Reward weights are hardcoded here (matching the lift-task pattern) rather
    than being surfaced through the task constructor.
    """

    kp_baseline: RewardTermCfg = MISSING
    kp_coarse: RewardTermCfg = MISSING
    kp_fine: RewardTermCfg = MISSING
    engagement_bonus: RewardTermCfg = MISSING
    success_bonus: RewardTermCfg = MISSING
    action_penalty_asset: RewardTermCfg = MISSING
    action_grad_penalty: RewardTermCfg = MISSING
    contact_penalty: RewardTermCfg = MISSING
    success_pred_error: RewardTermCfg = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float],
        held_gear_base_offset: list[float],
        gear_peg_height: float,
        success_z_fraction: float,
        xy_threshold: float,
        peg_offset_xy_noise: float = 0.005,
    ):
        hgo = held_gear_base_offset
        gear_cfg = SceneEntityCfg(gear_name)
        board_cfg = SceneEntityCfg(board_name)
        common_params = {
            "gear_cfg": gear_cfg,
            "board_cfg": board_cfg,
            "peg_offset": peg_offset,
            "held_gear_base_offset": hgo,
            "keypoint_scale": 0.15,
            "num_keypoints": 4,
            "peg_offset_xy_noise": peg_offset_xy_noise,
        }
        bonus_params = {
            "gear_cfg": gear_cfg,
            "board_cfg": board_cfg,
            "peg_offset": peg_offset,
            "held_gear_base_offset": hgo,
        }

        self.kp_baseline = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**common_params, "squash_a": 5.0, "squash_b": 4.0},
        )
        self.kp_coarse = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**common_params, "squash_a": 50.0, "squash_b": 2.0},
        )
        self.kp_fine = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**common_params, "squash_a": 100.0, "squash_b": 0.0},
        )
        self.engagement_bonus = RewardTermCfg(
            func=gear_insertion_rewards.gear_insertion_engagement_bonus,
            weight=1.0,
            params={**bonus_params, "engage_z_fraction": 0.90, "xy_threshold": xy_threshold},
        )
        self.success_bonus = RewardTermCfg(
            func=gear_insertion_rewards.gear_insertion_success_bonus,
            weight=1.0,
            params={**bonus_params, "success_z_fraction": success_z_fraction},
        )
        self.action_penalty_asset = RewardTermCfg(
            func=gear_insertion_rewards.osc_action_magnitude_penalty,
            weight=-0.0005,
            params={"pos_action_threshold": 0.02, "rot_action_threshold": 0.097},
        )
        self.action_grad_penalty = RewardTermCfg(
            func=gear_insertion_rewards.osc_action_delta_penalty,
            weight=-0.01,
            params={},
        )
        self.contact_penalty = RewardTermCfg(
            func=gear_insertion_rewards.wrist_contact_force_penalty,
            weight=-0.001,
            params={},
        )
        self.success_pred_error = RewardTermCfg(
            func=gear_insertion_rewards.success_prediction_error,
            weight=-1.0,
            params={
                "gear_cfg": gear_cfg,
                "board_cfg": board_cfg,
                "peg_offset": peg_offset,
                "held_gear_base_offset": hgo,
                "gear_peg_height": gear_peg_height,
                "success_z_fraction": success_z_fraction,
                "xy_threshold": xy_threshold,
                "delay_until_ratio": 0.25,
            },
        )
