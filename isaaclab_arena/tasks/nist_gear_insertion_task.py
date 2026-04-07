# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Simple gear insertion task for the assembled NIST board.

The medium gear is a separate object that the robot must pick up and insert
onto the peg. The peg can be on the assembled board or on the gearbase
asset (gears_and_base; USD gearbase_and_gears.usd). Use :attr:`gear_base_asset` to specify the
asset whose pose + offset defines the peg (fixed asset + local offset); if omitted, the
assembled board is used.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, ObservationGroupCfg as ObsGroup, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.observations import gear_insertion_observations
from isaaclab_arena_environments.mdp.observations import body_pos_in_env_frame, body_quat_canonical
from isaaclab_arena.tasks.rewards import gear_insertion_rewards
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.terminations import (
    gear_dropped_from_gripper,
    gear_mesh_insertion_success,
    gear_orientation_exceeded,
)
from isaaclab_arena.terms.events import place_gear_in_gripper
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


class NistGearInsertionTask(TaskBase):
    """Gear insertion task: insert the medium gear onto the peg.

    Peg position is computed from a fixed asset (board or separate gear-base USD)
    plus an offset in that asset's local frame (assembly peg-insert convention). Use
    :attr:`gear_base_asset` when the peg is on the gearbase (gears_and_base / gearbase_and_gears.usd).
    """

    def __init__(
        self,
        assembled_board: Asset,
        held_gear: Asset,
        background_scene: Asset,
        peg_offset_from_board: list[float] | None = None,
        held_gear_base_offset: list[float] | None = None,
        gear_peg_height: float = 0.02,
        success_z_fraction: float = 0.30,
        xy_threshold: float = 0.0025,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        start_in_gripper: bool = False,
        num_arm_joints: int = 7,
        hand_grasp_width: float = 0.03,
        hand_close_width: float = 0.0,
        gripper_joint_setter_func: Callable[
            [torch.Tensor, Sequence[int], Sequence[int], float], None
        ] | None = None,
        end_effector_body_name: str = "panda_hand",
        grasp_rot_offset: list[float] | None = None,
        grasp_offset: list[float] | None = None,
        include_success_bonus: bool = True,
        enable_randomization: bool = False,
        arm_joint_names: str = "panda_joint.*",
        finger_body_names: str = ".*finger",
        peg_offset_xy_noise: float = 0.0,
        gear_base_asset: Asset | None = None,
        peg_offset_for_obs: list[float] | None = None,
        disable_drop_terminations: bool = False,
        engagement_xy_threshold: float | None = None,
        success_bonus_weight: float | None = None,
        include_insertion_regularizers: bool = False,
        pos_action_threshold: float = 0.02,
        rot_action_threshold: float = 0.097,
        action_penalty_weight: float = -0.0005,
        action_grad_penalty_weight: float = -0.01,
        contact_penalty_weight: float = -0.001,
        success_pred_error_weight: float = -1.0,
        success_pred_error_delay: float = 0.25,
        extra_event_terms: dict[str, EventTermCfg] | None = None,
    ):
        super().__init__(episode_length_s=episode_length_s)
        self.assembled_board = assembled_board
        self.held_gear = held_gear
        self.background_scene = background_scene
        self._gear_base_asset = gear_base_asset if gear_base_asset is not None else assembled_board
        self.peg_offset_from_board = peg_offset_from_board or [2.025e-2, 0.0, 0.0]
        self.peg_offset_for_obs = peg_offset_for_obs
        self.held_gear_base_offset = held_gear_base_offset if held_gear_base_offset is not None else [2.025e-2, 0.0, 0.0]
        self.peg_offset_xy_noise = peg_offset_xy_noise
        self.gear_peg_height = gear_peg_height
        self.success_z_fraction = success_z_fraction
        self.xy_threshold = xy_threshold
        self.start_in_gripper = start_in_gripper
        self.num_arm_joints = num_arm_joints
        self.hand_grasp_width = hand_grasp_width
        self.hand_close_width = hand_close_width
        self.gripper_joint_setter_func = gripper_joint_setter_func
        self.end_effector_body_name = end_effector_body_name
        self.grasp_rot_offset = grasp_rot_offset or [1.0, 0.0, 0.0, 0.0]
        self.grasp_offset = grasp_offset or [0.0, 0.0, 0.0]
        self.include_success_bonus = include_success_bonus
        self.enable_randomization = enable_randomization
        self.arm_joint_names = arm_joint_names
        self.finger_body_names = finger_body_names
        self.disable_drop_terminations = disable_drop_terminations
        self.engagement_xy_threshold = engagement_xy_threshold
        self.success_bonus_weight = success_bonus_weight
        self.include_insertion_regularizers = include_insertion_regularizers
        self.pos_action_threshold = pos_action_threshold
        self.rot_action_threshold = rot_action_threshold
        self.action_penalty_weight = action_penalty_weight
        self.action_grad_penalty_weight = action_grad_penalty_weight
        self.contact_penalty_weight = contact_penalty_weight
        self.success_pred_error_weight = success_pred_error_weight
        self.success_pred_error_delay = success_pred_error_delay
        self.extra_event_terms = extra_event_terms or {}
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
            include_success_bonus=self.include_success_bonus,
            peg_offset_xy_noise=self.peg_offset_xy_noise,
            gear_peg_height=self.gear_peg_height,
            success_z_fraction=self.success_z_fraction,
            xy_threshold=self.xy_threshold,
            engagement_xy_threshold=self.engagement_xy_threshold,
            success_bonus_weight=self.success_bonus_weight,
            include_insertion_regularizers=self.include_insertion_regularizers,
            pos_action_threshold=self.pos_action_threshold,
            rot_action_threshold=self.rot_action_threshold,
            action_penalty_weight=self.action_penalty_weight,
            action_grad_penalty_weight=self.action_grad_penalty_weight,
            contact_penalty_weight=self.contact_penalty_weight,
            success_pred_error_weight=self.success_pred_error_weight,
            success_pred_error_delay=self.success_pred_error_delay,
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
        if self.start_in_gripper:
            cfg.gear_dropped_from_gripper = TerminationTermCfg(
                func=gear_dropped_from_gripper,
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_body_name": self.end_effector_body_name,
                    "distance_threshold": 0.15,
                },
            )
        if self.disable_drop_terminations:
            cfg.object_dropped = None
            cfg.gear_dropped_from_gripper = None
        return cfg

    def get_events_cfg(self):
        cfg = _EventsCfg()
        if self.start_in_gripper and self.gripper_joint_setter_func is not None:
            cfg.place_gear = EventTermCfg(
                func=place_gear_in_gripper,
                mode="reset",
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "num_arm_joints": self.num_arm_joints,
                    "hand_grasp_width": self.hand_grasp_width,
                    "hand_close_width": self.hand_close_width,
                    "gripper_joint_setter_func": self.gripper_joint_setter_func,
                    "end_effector_body_name": self.end_effector_body_name,
                    "grasp_rot_offset": self.grasp_rot_offset,
                    "grasp_offset": self.grasp_offset,
                },
            )
        if self.enable_randomization:
            cfg.robot_actuator_gains = EventTermCfg(
                func=mdp_isaac_lab.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[self.arm_joint_names]),
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
                    "asset_cfg": SceneEntityCfg("robot", joint_names=[self.arm_joint_names]),
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
                    "asset_cfg": SceneEntityCfg("robot", body_names=self.finger_body_names),
                    "static_friction_range": (0.75, 0.75),
                    "dynamic_friction_range": (0.75, 0.75),
                    "restitution_range": (0.0, 0.0),
                    "num_buckets": 16,
                },
            )
        for name, term_cfg in self.extra_event_terms.items():
            setattr(cfg, name, term_cfg)
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
    gear_orientation_exceeded: TerminationTermCfg | None = None


@configclass
class _EventsCfg:
    """Events: reset to default poses plus optional deploy-style randomization."""

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    fixed_asset_pose: EventTermCfg | None = None
    place_gear: EventTermCfg | None = None
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
                func=gear_insertion_observations.gear_pos_in_env_frame,
                params={"gear_cfg": SceneEntityCfg(gear_name)},
            )
            gear_quat = ObsTerm(
                func=gear_insertion_observations.gear_quat_canonical,
                params={"gear_cfg": SceneEntityCfg(gear_name)},
            )
            peg_pos = ObsTerm(
                func=gear_insertion_observations.peg_pos_in_env_frame,
                params={"board_cfg": SceneEntityCfg(board_name), "peg_offset": peg_offset},
            )
            board_quat = ObsTerm(
                func=gear_insertion_observations.board_quat_canonical,
                params={"board_cfg": SceneEntityCfg(board_name)},
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
    """Keypoint squashing, bonuses, and optional insertion regularizers for gear insertion."""

    kp_baseline: RewardTermCfg = MISSING
    kp_coarse: RewardTermCfg = MISSING
    kp_fine: RewardTermCfg = MISSING
    engagement_bonus: RewardTermCfg | None = None
    success_bonus: RewardTermCfg | None = None

    action_penalty_asset: RewardTermCfg | None = None
    action_grad_penalty: RewardTermCfg | None = None
    contact_penalty: RewardTermCfg | None = None
    success_pred_error: RewardTermCfg | None = None

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float],
        held_gear_base_offset: list[float] | None = None,
        include_success_bonus: bool = True,
        peg_offset_xy_noise: float = 0.0,
        gear_peg_height: float = 0.02,
        success_z_fraction: float = 0.05,
        xy_threshold: float = 0.0025,
        engagement_xy_threshold: float | None = None,
        success_bonus_weight: float | None = None,
        include_insertion_regularizers: bool = False,
        pos_action_threshold: float = 0.02,
        rot_action_threshold: float = 0.097,
        action_penalty_weight: float = -0.0005,
        action_grad_penalty_weight: float = -0.01,
        contact_penalty_weight: float = -0.001,
        success_pred_error_weight: float = -1.0,
        success_pred_error_delay: float = 0.25,
    ):
        self.action_penalty_asset = None
        self.action_grad_penalty = None
        self.contact_penalty = None
        self.success_pred_error = None
        hgo = held_gear_base_offset if held_gear_base_offset is not None else [2.025e-2, 0.0, 0.0]
        common_params = {
            "gear_cfg": SceneEntityCfg(gear_name),
            "board_cfg": SceneEntityCfg(board_name),
            "peg_offset": peg_offset,
            "held_gear_base_offset": hgo,
            "keypoint_scale": 0.15,
            "num_keypoints": 4,
            "peg_offset_xy_noise": peg_offset_xy_noise,
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
        bonus_params = {
            "gear_cfg": SceneEntityCfg(gear_name),
            "board_cfg": SceneEntityCfg(board_name),
            "peg_offset": peg_offset,
            "held_gear_base_offset": hgo,
        }
        if include_success_bonus:
            self.engagement_bonus = RewardTermCfg(
                func=gear_insertion_rewards.gear_insertion_engagement_bonus,
                weight=1.0,
                params={
                    **bonus_params,
                    "engage_z_fraction": 0.90,
                    "xy_threshold": 0.015 if engagement_xy_threshold is None else engagement_xy_threshold,
                },
            )
            self.success_bonus = RewardTermCfg(
                func=gear_insertion_rewards.gear_insertion_success_bonus,
                weight=1.0 if success_bonus_weight is None else success_bonus_weight,
                params={**bonus_params, "success_z_fraction": success_z_fraction},
            )
        else:
            self.engagement_bonus = None
            self.success_bonus = None
        if include_insertion_regularizers:
            self.action_penalty_asset = RewardTermCfg(
                func=gear_insertion_rewards.osc_action_magnitude_penalty,
                weight=action_penalty_weight,
                params={
                    "pos_action_threshold": pos_action_threshold,
                    "rot_action_threshold": rot_action_threshold,
                },
            )
            self.action_grad_penalty = RewardTermCfg(
                func=gear_insertion_rewards.osc_action_delta_penalty,
                weight=action_grad_penalty_weight,
                params={},
            )
            self.contact_penalty = RewardTermCfg(
                func=gear_insertion_rewards.wrist_contact_force_penalty,
                weight=contact_penalty_weight,
                params={},
            )
            self.success_pred_error = RewardTermCfg(
                func=gear_insertion_rewards.success_prediction_error,
                weight=success_pred_error_weight,
                params={
                    "gear_cfg": SceneEntityCfg(gear_name),
                    "board_cfg": SceneEntityCfg(board_name),
                    "peg_offset": peg_offset,
                    "held_gear_base_offset": hgo,
                    "gear_peg_height": gear_peg_height,
                    "success_z_fraction": success_z_fraction,
                    "xy_threshold": xy_threshold,
                    "delay_until_ratio": success_pred_error_delay,
                },
            )
