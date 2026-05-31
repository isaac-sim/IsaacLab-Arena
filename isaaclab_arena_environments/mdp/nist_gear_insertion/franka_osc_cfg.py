# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Franka OSC action and observation configs for NIST gear insertion."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.controllers import OperationalSpaceControllerCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_action import NistGearInsertionOscActionCfg
from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_observations import NistGearInsertionPolicyObservations


@configclass
class FrankaNistGearInsertionObservationsCfg:
    """Policy observations for a Franka gear-insertion OSC environment.

    The fixed asset name and peg offset are supplied by the environment because
    they describe scene geometry.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Gear-insertion policy observation group."""

        nist_gear_insertion_policy_obs: ObsTerm = MISSING

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = MISSING

    def __init__(
        self,
        fixed_asset_name: str,
        peg_offset: tuple[float, float, float],
        fingertip_body_name: str = "panda_fingertip_centered",
        concatenate_observation_terms: bool = False,
    ):
        self.policy = self.PolicyCfg()
        self.policy.nist_gear_insertion_policy_obs = ObsTerm(
            func=NistGearInsertionPolicyObservations,
            params={
                "robot_name": "robot",
                "board_name": fixed_asset_name,
                "peg_offset": list(peg_offset),
                "fingertip_body_name": fingertip_body_name,
                "force_body_name": "force_sensor",
                "pos_noise_level": 0.0,
                "rot_noise_level_deg": 0.0,
                "force_noise_level": 0.0,
            },
        )
        self.policy.concatenate_terms = concatenate_observation_terms


@configclass
class FrankaNistGearInsertionOscActionsCfg:
    """Action terms for the Franka gear-insertion OSC policy.

    The seven-dimensional arm policy controls the insertion OSC term. The
    gripper is not exposed as a policy action; reset events place the held gear
    in the hand and maintain the grasp target.
    """

    arm_action: ActionTermCfg = MISSING
    gripper_action: ActionTermCfg | None = None

    def __init__(
        self,
        fixed_asset_name: str,
        peg_offset: tuple[float, float, float],
    ):
        self.arm_action = NistGearInsertionOscActionCfg(
            asset_name="robot",
            joint_names=["panda_joint[1-7]"],
            body_name="panda_fingertip_centered",
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_abs"],
                impedance_mode="fixed",
                inertial_dynamics_decoupling=True,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=[565.0, 565.0, 565.0, 28.0, 28.0, 28.0],
                motion_damping_ratio_task=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                nullspace_control="position",
                nullspace_stiffness=10.0,
                nullspace_damping_ratio=1.0,
            ),
            position_scale=1.0,
            orientation_scale=1.0,
            nullspace_joint_pos_target="default",
            fixed_asset_name=fixed_asset_name,
            peg_offset=peg_offset,
            force_body_name="force_sensor",
        )
        self.gripper_action = None
