# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import torch
from dataclasses import dataclass

import pytest

_PEG_BASE_OFFSET = [0.02025, 0.0, 0.0]
_PEG_TIP_OFFSET = [0.02025, 0.0, 0.025]


@dataclass
class _DummyAsset:
    name: str
    object_min_z: float | None = None
    reset_pose_disabled: bool = False

    def disable_reset_pose(self) -> None:
        self.reset_pose_disabled = True


def _make_nist_task(**kwargs):
    from isaaclab_arena.tasks.nist_gear_insertion.task import GearInsertionGeometryCfg, NistGearInsertionRLTask

    return NistGearInsertionRLTask(
        assembled_board=_DummyAsset("nist_board_assembled"),
        held_gear=_DummyAsset("medium_nist_gear"),
        background_scene=_DummyAsset("table", object_min_z=0.0),
        gear_base_asset=_DummyAsset("gears_and_base"),
        geometry_cfg=GearInsertionGeometryCfg(
            peg_offset_from_board=list(_PEG_BASE_OFFSET),
            peg_offset_for_obs=list(_PEG_TIP_OFFSET),
            held_gear_base_offset=list(_PEG_BASE_OFFSET),
            success_z_fraction=0.20,
            xy_threshold=0.0025,
        ),
        **kwargs,
    )


def test_task_geometry_config_feeds_observations_rewards_and_success():
    from isaaclab_arena.tasks.nist_gear_insertion import geometry, rewards
    from isaaclab_arena.tasks.nist_gear_insertion.terminations import gear_mesh_insertion_success

    task = _make_nist_task(disable_success_termination=True)

    obs_cfg = task.get_observation_cfg()
    reward_cfg = task.get_rewards_cfg()
    termination_cfg = task.get_termination_cfg()

    assert task.held_gear.reset_pose_disabled
    assert obs_cfg.task_obs.peg_pos.func is geometry.peg_pos_in_env_frame
    assert obs_cfg.task_obs.peg_pos.params["board_cfg"].name == "gears_and_base"
    assert obs_cfg.task_obs.peg_pos.params["peg_offset"] == _PEG_TIP_OFFSET

    assert obs_cfg.task_obs.peg_delta.func is geometry.peg_delta_from_held_gear_base
    assert obs_cfg.task_obs.peg_delta.params["gear_cfg"].name == "medium_nist_gear"
    assert obs_cfg.task_obs.peg_delta.params["held_gear_base_offset"] == _PEG_BASE_OFFSET
    assert obs_cfg.task_obs.concatenate_terms

    for term in (reward_cfg.kp_baseline, reward_cfg.kp_coarse, reward_cfg.kp_fine):
        assert term.func is rewards.gear_peg_keypoint_squashing
        assert term.params["peg_offset"] == _PEG_BASE_OFFSET
        assert term.params["held_gear_base_offset"] == _PEG_BASE_OFFSET

    assert reward_cfg.engagement_bonus.func is rewards.gear_insertion_geometry_bonus
    assert reward_cfg.engagement_bonus.params["z_fraction"] == 0.90
    assert reward_cfg.success_bonus.func is rewards.gear_insertion_geometry_bonus
    assert reward_cfg.success_bonus.params["z_fraction"] == 0.20

    assert termination_cfg.success.func is gear_mesh_insertion_success
    assert termination_cfg.success.params["gear_base_offset"] == _PEG_BASE_OFFSET
    assert termination_cfg.success.params["held_gear_base_offset"] == _PEG_BASE_OFFSET
    assert termination_cfg.success.params["disable_success_termination"]
    assert termination_cfg.object_dropped is None
    assert termination_cfg.gear_dropped_from_gripper is None


def test_task_events_add_grasp_reset_and_optional_randomization():
    from isaaclab_arena.embodiments.franka.nist_gear_insertion.gear_grasp import franka_gripper_joint_setter
    from isaaclab_arena.tasks.nist_gear_insertion.events import GraspCfg, place_gear_in_gripper

    grasp_cfg = GraspCfg(
        gripper_joint_setter_func=franka_gripper_joint_setter,
        arm_joint_names="panda_joint[1-7]",
    )

    events_cfg = _make_nist_task(grasp_cfg=grasp_cfg).get_events_cfg()

    assert events_cfg.place_gear.func is place_gear_in_gripper
    assert events_cfg.place_gear.params["gear_cfg"].name == "medium_nist_gear"
    assert events_cfg.place_gear.params["grasp_cfg"].gripper_joint_setter_func is franka_gripper_joint_setter
    assert events_cfg.place_gear.params["grasp_cfg"].arm_joint_names == "panda_joint[1-7]"
    assert events_cfg.held_object_mass is None
    assert events_cfg.fixed_asset_pose is None
    assert events_cfg.robot_actuator_gains is None
    assert events_cfg.robot_joint_friction is None

    randomized_cfg = _make_nist_task(grasp_cfg=grasp_cfg, enable_randomization=True).get_events_cfg()

    assert randomized_cfg.held_object_mass.params["asset_cfg"].name == "medium_nist_gear"
    assert randomized_cfg.fixed_asset_pose.params["asset_cfg"].name == "gears_and_base"
    assert randomized_cfg.fixed_asset_pose.params["pose_range"]["yaw"] == (0.0, math.radians(15.0))
    assert randomized_cfg.robot_actuator_gains.params["asset_cfg"].joint_names == "panda_joint[1-7]"
    assert randomized_cfg.robot_joint_friction.params["asset_cfg"].joint_names == "panda_joint[1-7]"

    with pytest.raises(AssertionError):
        _make_nist_task(enable_randomization=True).get_events_cfg()


def test_grasp_cfg_width_setter_maps_total_width_to_finger_joints():
    from isaaclab_arena.embodiments.franka.nist_gear_insertion.gear_grasp import (
        franka_gripper_joint_setter,
        get_franka_nist_gear_insertion_grasp_config,
    )

    grasp_cfg = get_franka_nist_gear_insertion_grasp_config()
    joint_pos = torch.zeros(2, 9)

    franka_gripper_joint_setter(joint_pos, torch.tensor([0, 1]), [7, 8], width=0.04)

    assert torch.allclose(joint_pos[:, 7], torch.full((2,), 0.02))
    assert torch.allclose(joint_pos[:, 8], torch.full((2,), 0.02))
    assert grasp_cfg.gripper_joint_setter_func is franka_gripper_joint_setter
    assert grasp_cfg.hand_grasp_width == 0.03
    assert grasp_cfg.hand_close_width == 0.0
    assert grasp_cfg.grasp_rot_offset == [1.0, 0.0, 0.0, 0.0]
    assert grasp_cfg.grasp_offset == [0.02, 0.0, -0.128]


def test_franka_osc_action_observation_and_reward_configs_use_scene_geometry():
    from isaaclab_arena_environments.mdp.nist_gear_insertion import osc_rewards
    from isaaclab_arena_environments.mdp.nist_gear_insertion.franka_osc_cfg import (
        FrankaNistGearInsertionObservationsCfg,
        FrankaNistGearInsertionOscActionsCfg,
    )
    from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_action import ACTION_DIM, NistGearInsertionOscActionCfg
    from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_observations import (
        POLICY_OBS_DIM,
        POLICY_OBS_LAYOUT,
        NistGearInsertionPolicyObservations,
    )

    action_cfg = FrankaNistGearInsertionOscActionsCfg(
        fixed_asset_name="gears_and_base",
        peg_offset=tuple(_PEG_TIP_OFFSET),
    )
    obs_cfg = FrankaNistGearInsertionObservationsCfg(
        fixed_asset_name="gears_and_base",
        peg_offset=tuple(_PEG_TIP_OFFSET),
        fingertip_body_name="panda_fingertip_centered",
        concatenate_observation_terms=True,
    )
    reward_cfg = osc_rewards.NistGearInsertionOscRewardsCfg(
        gear_name="medium_nist_gear",
        board_name="gears_and_base",
        peg_offset=_PEG_BASE_OFFSET,
        held_gear_base_offset=_PEG_BASE_OFFSET,
        gear_peg_height=0.02,
        success_z_fraction=0.20,
        xy_threshold=0.0025,
    )

    assert isinstance(action_cfg.arm_action, NistGearInsertionOscActionCfg)
    assert action_cfg.arm_action.fixed_asset_name == "gears_and_base"
    assert action_cfg.arm_action.peg_offset == tuple(_PEG_TIP_OFFSET)
    assert action_cfg.arm_action.body_name == "panda_fingertip_centered"
    assert action_cfg.gripper_action is None
    assert ACTION_DIM == 7

    policy_obs = obs_cfg.policy.nist_gear_insertion_policy_obs
    assert policy_obs.func is NistGearInsertionPolicyObservations
    assert policy_obs.params["board_name"] == "gears_and_base"
    assert policy_obs.params["peg_offset"] == _PEG_TIP_OFFSET
    assert policy_obs.params["fingertip_body_name"] == "panda_fingertip_centered"
    assert obs_cfg.policy.concatenate_terms
    assert POLICY_OBS_DIM == sum(size for _, size in POLICY_OBS_LAYOUT)

    assert reward_cfg.action_magnitude_penalty.func is osc_rewards.osc_action_magnitude_penalty
    assert reward_cfg.action_delta_penalty.func is osc_rewards.osc_action_delta_penalty
    assert reward_cfg.contact_penalty.func is osc_rewards.wrist_contact_force_penalty
    assert reward_cfg.success_prediction_error.func is osc_rewards.success_prediction_error
    assert reward_cfg.success_prediction_error.params["peg_offset"] == _PEG_BASE_OFFSET
    assert reward_cfg.success_prediction_error.params["success_z_fraction"] == 0.20


def test_gear_insertion_geometry_success_thresholds():
    from isaaclab_arena.tasks.nist_gear_insertion.geometry import check_gear_insertion_geometry

    held_base_pos = torch.tensor([
        [0.0, 0.0, 0.001],
        [0.01, 0.0, 0.001],
        [0.0, 0.0, 0.01],
    ])
    peg_pos = torch.zeros(3, 3)

    success = check_gear_insertion_geometry(
        held_base_pos=held_base_pos,
        peg_pos=peg_pos,
        gear_peg_height=0.02,
        z_fraction=0.2,
        xy_threshold=0.0025,
    )

    assert success.tolist() == [True, False, False]


def test_osc_yaw_mapping_wraps_to_policy_interval():
    from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_action import (
        _target_yaw_to_action,
        _wrap_yaw_to_action_range,
    )

    yaw = torch.tensor([-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, 0.75 * math.pi, math.pi])
    wrapped_yaw = _wrap_yaw_to_action_range(yaw)
    action = _target_yaw_to_action(wrapped_yaw)

    assert torch.allclose(wrapped_yaw, torch.tensor([-math.pi, -math.pi / 2.0, 0.0, math.pi / 2.0, -math.pi, -math.pi]))
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_policy_observation_layout_masks_symmetric_quaternion_and_computes_velocity():
    from isaaclab_arena_environments.mdp.nist_gear_insertion.osc_observations import (
        POLICY_OBS_LAYOUT,
        PREV_ACTION_DIM,
        _compute_pose_velocities,
        _make_reported_quat,
    )

    assert POLICY_OBS_LAYOUT[-1] == ("prev_actions", PREV_ACTION_DIM)

    flip = torch.tensor([1.0, -1.0])
    noisy_quat = torch.tensor([[0.1, 0.2, 0.3, 0.4], [-0.2, 0.1, -0.4, 0.3]])
    reported_quat = _make_reported_quat(noisy_quat, flip)

    assert torch.allclose(reported_quat[:, 2:], torch.zeros(2, 2))
    assert torch.allclose(reported_quat[:, :2], noisy_quat[:, :2] * flip.unsqueeze(-1))

    prev_pos = torch.zeros(2, 3)
    noisy_pos = torch.tensor([[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]])
    ee_linvel, ee_angvel = _compute_pose_velocities(prev_pos, reported_quat, noisy_pos, reported_quat, dt=0.05)

    assert torch.allclose(ee_linvel, noisy_pos / 0.05)
    assert torch.allclose(ee_angvel, torch.zeros_like(ee_angvel), atol=1e-6)


def test_nist_osc_environment_cli_is_fixed_to_specialized_embodiment():
    from isaaclab_arena_environments.nist_assembled_gearmesh_osc_environment import NISTAssembledGearMeshOSCEnvironment

    parser = argparse.ArgumentParser()
    NISTAssembledGearMeshOSCEnvironment.add_cli_args(parser)

    args_cli = parser.parse_args([])

    assert not hasattr(args_cli, "embodiment")
    assert not args_cli.disable_success_termination
    assert parser.parse_args(["--rl_training_mode"]).disable_success_termination
    with pytest.raises(SystemExit):
        parser.parse_args(["--embodiment", "franka_ik"])
