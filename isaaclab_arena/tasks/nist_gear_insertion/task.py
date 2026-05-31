# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""NIST gear insertion task.

The task wiring follows the structure of Isaac Lab Factory/Forge insertion
tasks, with Arena assets and task registration.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import MISSING, field

import isaaclab.envs.mdp as mdp_isaac_lab
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from . import geometry as gear_insertion_geometry
from . import observations as gear_insertion_observations
from . import rewards as gear_insertion_rewards
from .events import GraspCfg, place_gear_in_gripper
from .terminations import gear_dropped_from_gripper, gear_mesh_insertion_success

_DEFAULT_PEG_OFFSET = (2.025e-2, 0.0, 0.0)


@configclass
class GearInsertionGeometryCfg:
    """Geometry parameters shared by observations, rewards, and terminations.

    Offsets are expressed in the local frame of their asset. ``peg_offset_for_obs``
    may differ from ``peg_offset_from_board`` when the policy should target the
    peg tip while success checks use the peg base.
    """

    peg_offset_from_board: list[float] = field(default_factory=lambda: list(_DEFAULT_PEG_OFFSET))
    peg_offset_for_obs: list[float] | None = None
    held_gear_base_offset: list[float] = field(default_factory=lambda: list(_DEFAULT_PEG_OFFSET))
    gear_peg_height: float = 0.02
    success_z_fraction: float = 0.30
    xy_threshold: float = 0.0025
    peg_offset_xy_noise: float = 0.005


class NistGearInsertionRLTask(TaskBase):
    """RL task for inserting the held gear onto a fixed NIST peg.

    The task owns scene-level wiring: generic observations and geometry rewards,
    reset events, success/drop terminations, and success-rate metrics. The OSC
    policy observation and controller penalties are configured by the
    environment MDP layer because they depend on a specific action term.
    """

    def __init__(
        self,
        assembled_board: Asset,
        held_gear: Asset,
        background_scene: Asset,
        gear_base_asset: Asset | None = None,
        geometry_cfg: GearInsertionGeometryCfg | None = None,
        episode_length_s: float | None = None,
        task_description: str | None = None,
        grasp_cfg: GraspCfg | None = None,
        fingertip_body_name: str = "panda_fingertip_centered",
        enable_randomization: bool = False,
        disable_drop_terminations: bool = True,
        disable_success_termination: bool = False,
    ):
        super().__init__(episode_length_s=episode_length_s, task_description=task_description)
        self.assembled_board = assembled_board
        self.held_gear = held_gear
        self.background_scene = background_scene
        self.held_gear.disable_reset_pose()
        self._gear_base_asset = gear_base_asset if gear_base_asset is not None else assembled_board
        self.geometry_cfg = geometry_cfg if geometry_cfg is not None else GearInsertionGeometryCfg()
        self.grasp_cfg = grasp_cfg
        self.fingertip_body_name = fingertip_body_name
        self.enable_randomization = enable_randomization
        self.disable_drop_terminations = disable_drop_terminations
        self.disable_success_termination = disable_success_termination
        if self.task_description is None:
            self.task_description = f"Insert the {held_gear.name} onto the gear base on the {assembled_board.name}"

    def get_scene_cfg(self):
        """Return no additional scene config.

        Arena constructs the scene from assets supplied by the environment.
        """

    def get_observation_cfg(self):
        """Return generic task observations for gear and peg geometry."""
        geometry_cfg = self.geometry_cfg
        # Policies can observe the peg tip while geometry rewards and
        # terminations evaluate insertion against the peg base.
        peg_obs = (
            geometry_cfg.peg_offset_for_obs
            if geometry_cfg.peg_offset_for_obs is not None
            else geometry_cfg.peg_offset_from_board
        )
        return GearInsertionObservationsCfg(
            gear_name=self.held_gear.name,
            board_name=self._gear_base_asset.name,
            peg_offset=peg_obs,
            held_gear_base_offset=geometry_cfg.held_gear_base_offset,
            fingertip_body_name=self.fingertip_body_name,
        )

    def get_rewards_cfg(self):
        """Return geometry-only reward terms for the task."""
        geometry_cfg = self.geometry_cfg
        return GearInsertionRewardsCfg(
            gear_name=self.held_gear.name,
            board_name=self._gear_base_asset.name,
            peg_offset=geometry_cfg.peg_offset_from_board,
            held_gear_base_offset=geometry_cfg.held_gear_base_offset,
            gear_peg_height=geometry_cfg.gear_peg_height,
            success_z_fraction=geometry_cfg.success_z_fraction,
            xy_threshold=geometry_cfg.xy_threshold,
            peg_offset_xy_noise=geometry_cfg.peg_offset_xy_noise,
        )

    def get_termination_cfg(self):
        """Return success and optional drop termination terms."""
        geometry_cfg = self.geometry_cfg
        success = TerminationTermCfg(
            func=gear_mesh_insertion_success,
            params={
                "held_object_cfg": SceneEntityCfg(self.held_gear.name),
                "fixed_object_cfg": SceneEntityCfg(self._gear_base_asset.name),
                "gear_base_offset": geometry_cfg.peg_offset_from_board,
                "held_gear_base_offset": geometry_cfg.held_gear_base_offset,
                "gear_peg_height": geometry_cfg.gear_peg_height,
                "success_z_fraction": geometry_cfg.success_z_fraction,
                "xy_threshold": geometry_cfg.xy_threshold,
                "disable_success_termination": self.disable_success_termination,
            },
        )

        cfg = GearInsertionTerminationsCfg(success=success, object_dropped=None)
        # Drop checks are disabled during training to allow recovery.
        if not self.disable_drop_terminations:
            cfg.object_dropped = TerminationTermCfg(
                func=mdp_isaac_lab.root_height_below_minimum,
                params={
                    "minimum_height": self.background_scene.object_min_z,
                    "asset_cfg": SceneEntityCfg(self.held_gear.name),
                },
            )
        if not self.disable_drop_terminations and self.grasp_cfg is not None:
            cfg.gear_dropped_from_gripper = TerminationTermCfg(
                func=gear_dropped_from_gripper,
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "ee_body_name": self.grasp_cfg.end_effector_body_name,
                    "distance_threshold": 0.15,
                },
            )
        return cfg

    def get_events_cfg(self):
        """Return reset and randomization events for gear insertion."""
        cfg = GearInsertionEventsCfg()
        self._add_grasp_reset_event(cfg)
        if self.enable_randomization:
            self._add_randomization_events(cfg)
        return cfg

    def _add_grasp_reset_event(self, cfg: GearInsertionEventsCfg) -> None:
        """Add the reset event that places the held gear in the gripper."""
        grasp_cfg = self.grasp_cfg
        if grasp_cfg is not None and grasp_cfg.gripper_joint_setter_func is not None:
            # The held gear does not use its default reset pose in this task;
            # the embodiment-specific grasp event places it in the gripper.
            cfg.place_gear = EventTermCfg(
                func=place_gear_in_gripper,
                mode="reset",
                params={
                    "gear_cfg": SceneEntityCfg(self.held_gear.name),
                    "grasp_cfg": grasp_cfg,
                },
            )

    def _add_randomization_events(self, cfg: GearInsertionEventsCfg) -> None:
        """Add optional Factory/Forge-style domain randomization events."""
        grasp_cfg = self.grasp_cfg
        assert grasp_cfg is not None, "NIST gear insertion randomization requires an embodiment grasp configuration."
        arm_joints = grasp_cfg.arm_joint_names

        self._add_asset_randomization_events(cfg)
        self._add_robot_randomization_events(cfg, arm_joints)

    def _add_asset_randomization_events(self, cfg: GearInsertionEventsCfg) -> None:
        """Add reset randomization for the held gear and fixed gear base."""
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
                # Keep translational variation fixed; only the board yaw is
                # randomized as in the Forge insertion setup.
                "pose_range": {
                    "yaw": (0.0, math.radians(15.0)),
                },
                "velocity_range": {},
            },
        )

    def _add_robot_randomization_events(self, cfg: GearInsertionEventsCfg, arm_joints: str) -> None:
        """Add actuator and joint-parameter randomization for the arm."""
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

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        """Raise because this task currently only defines the RL setup."""
        del arm_mode
        raise NotImplementedError("NIST gear insertion does not define a Mimic configuration yet.")

    def get_metrics(self) -> list[MetricBase]:
        """Return task metrics used during evaluation."""
        return [SuccessRateMetric()]

    def get_viewer_cfg(self) -> ViewerCfg:
        """Return a camera view focused on the held gear and peg area."""
        return get_viewer_cfg_look_at_object(
            lookat_object=self.held_gear,
            offset=np.array([1.5, -0.5, 1.0]),
        )


@configclass
class GearInsertionTerminationsCfg:
    """Termination terms for the gear insertion task.

    Success is always registered so metrics can read it. Drop checks are
    optional and are disabled for recovery-focused RL training.
    """

    time_out: TerminationTermCfg = TerminationTermCfg(func=mdp_isaac_lab.time_out)
    success: TerminationTermCfg = MISSING
    object_dropped: TerminationTermCfg | None = None
    gear_dropped_from_gripper: TerminationTermCfg | None = None


@configclass
class GearInsertionEventsCfg:
    """Reset and randomization events for gear insertion.

    ``reset_all`` restores the base scene first; task-specific terms then place
    the held gear in the gripper and apply optional domain randomization.
    """

    reset_all: EventTermCfg = EventTermCfg(
        func=mdp_isaac_lab.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    place_gear: EventTermCfg | None = None
    fixed_asset_pose: EventTermCfg | None = None
    held_object_mass: EventTermCfg | None = None
    robot_actuator_gains: EventTermCfg | None = None
    robot_joint_friction: EventTermCfg | None = None


@configclass
class GearInsertionTaskObsCfg(ObsGroup):
    """Generic observation group for gear insertion task state.

    These terms expose object poses, peg geometry, robot joints, and noiseless
    end-effector pose. Controller-specific policy packing remains outside this
    generic task config.
    """

    gear_pos: ObsTerm = MISSING
    gear_quat: ObsTerm = MISSING
    peg_pos: ObsTerm = MISSING
    board_quat: ObsTerm = MISSING
    peg_delta: ObsTerm = MISSING
    joint_pos: ObsTerm = MISSING
    joint_vel: ObsTerm = MISSING
    ee_pos_noiseless: ObsTerm = MISSING
    ee_quat_noiseless: ObsTerm = MISSING

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class GearInsertionObservationsCfg:
    """Observation config for the generic gear insertion task state."""

    task_obs: GearInsertionTaskObsCfg = MISSING

    def __init__(
        self,
        gear_name: str,
        board_name: str,
        peg_offset: list[float],
        held_gear_base_offset: list[float] | None = None,
        fingertip_body_name: str = "panda_fingertip_centered",
    ):
        held_offset = held_gear_base_offset if held_gear_base_offset is not None else [2.025e-2, 0.0, 0.0]
        self.task_obs = GearInsertionTaskObsCfg()

        self.task_obs.gear_pos = ObsTerm(
            func=mdp_isaac_lab.root_pos_w,
            params={"asset_cfg": SceneEntityCfg(gear_name)},
        )
        self.task_obs.gear_quat = ObsTerm(
            func=mdp_isaac_lab.root_quat_w,
            params={"make_quat_unique": True, "asset_cfg": SceneEntityCfg(gear_name)},
        )
        self.task_obs.board_quat = ObsTerm(
            func=mdp_isaac_lab.root_quat_w,
            params={"make_quat_unique": True, "asset_cfg": SceneEntityCfg(board_name)},
        )
        self.task_obs.peg_pos = ObsTerm(
            func=gear_insertion_geometry.peg_pos_in_env_frame,
            params={"board_cfg": SceneEntityCfg(board_name), "peg_offset": peg_offset},
        )
        self.task_obs.peg_delta = ObsTerm(
            func=gear_insertion_geometry.peg_delta_from_held_gear_base,
            params={
                "gear_cfg": SceneEntityCfg(gear_name),
                "board_cfg": SceneEntityCfg(board_name),
                "peg_offset": peg_offset,
                "held_gear_base_offset": held_offset,
            },
        )
        self.task_obs.joint_pos = ObsTerm(
            func=mdp_isaac_lab.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.task_obs.joint_vel = ObsTerm(
            func=mdp_isaac_lab.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.task_obs.ee_pos_noiseless = ObsTerm(
            func=gear_insertion_observations.body_pos_in_env_frame,
            params={"body_name": fingertip_body_name},
        )
        self.task_obs.ee_quat_noiseless = ObsTerm(
            func=gear_insertion_observations.body_quat_canonical,
            params={"body_name": fingertip_body_name},
        )


@configclass
class GearInsertionRewardsCfg:
    """Reward terms for gear insertion.

    Keypoint shaping and insertion-depth bonuses mirror the Factory/Forge
    assembly reward structure used for peg and gear insertion.
    """

    kp_baseline: RewardTermCfg = MISSING
    kp_coarse: RewardTermCfg = MISSING
    kp_fine: RewardTermCfg = MISSING
    engagement_bonus: RewardTermCfg = MISSING
    success_bonus: RewardTermCfg = MISSING

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
        gear_cfg = SceneEntityCfg(gear_name)
        board_cfg = SceneEntityCfg(board_name)
        keypoint_params = {
            "gear_cfg": gear_cfg,
            "board_cfg": board_cfg,
            "peg_offset": peg_offset,
            "held_gear_base_offset": held_gear_base_offset,
            "keypoint_scale": 0.15,
            "num_keypoints": 4,
            "peg_offset_xy_noise": peg_offset_xy_noise,
        }
        geometry_params = {
            "gear_cfg": gear_cfg,
            "board_cfg": board_cfg,
            "peg_offset": peg_offset,
            "held_gear_base_offset": held_gear_base_offset,
            "gear_peg_height": gear_peg_height,
            "xy_threshold": xy_threshold,
        }

        # The keypoint terms share geometry and differ only by squashing
        # strength, giving coarse-to-fine shaping around the peg.
        self.kp_baseline = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**keypoint_params, "squash_a": 5.0, "squash_b": 4.0},
        )
        self.kp_coarse = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**keypoint_params, "squash_a": 50.0, "squash_b": 2.0},
        )
        self.kp_fine = RewardTermCfg(
            func=gear_insertion_rewards.gear_peg_keypoint_squashing,
            weight=1.0,
            params={**keypoint_params, "squash_a": 100.0, "squash_b": 0.0},
        )

        # Sparse insertion bonuses use the same success geometry as the task
        # termination with different insertion-depth thresholds.
        self.engagement_bonus = RewardTermCfg(
            func=gear_insertion_rewards.gear_insertion_geometry_bonus,
            weight=1.0,
            params={**geometry_params, "z_fraction": 0.90},
        )
        self.success_bonus = RewardTermCfg(
            func=gear_insertion_rewards.gear_insertion_geometry_bonus,
            weight=1.0,
            params={**geometry_params, "z_fraction": success_z_fraction},
        )
