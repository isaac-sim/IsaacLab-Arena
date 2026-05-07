# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

class _TestAsset:
    def __init__(self, name: str, object_min_z: float | None = None):
        self.name = name
        self.object_min_z = object_min_z
        self.reset_pose = True

    def disable_reset_pose(self) -> None:
        self.reset_pose = False


def _asset(name: str, object_min_z: float | None = None) -> _TestAsset:
    return _TestAsset(name=name, object_min_z=object_min_z)


def _nist_task(**kwargs):
    from isaaclab_arena.tasks.nist_gear_insertion_task import GearInsertionGeometryCfg, NistGearInsertionTask

    return NistGearInsertionTask(
        assembled_board=_asset("nist_board_assembled"),
        held_gear=_asset("medium_nist_gear"),
        background_scene=_asset("table", object_min_z=0.0),
        gear_base_asset=_asset("gears_and_base"),
        geometry_cfg=GearInsertionGeometryCfg(
            peg_offset_from_board=[0.02025, 0.0, 0.0],
            peg_offset_for_obs=[0.02025, 0.0, 0.025],
            held_gear_base_offset=[0.02025, 0.0, 0.0],
            success_z_fraction=0.20,
            xy_threshold=0.0025,
        ),
        **kwargs,
    )


def test_franka_nist_gear_osc_embodiment_registers_task_terms():
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import (
        FrankaNistGearInsertionObservationsCfg,
        FrankaNistGearInsertionOscEmbodiment,
    )
    from isaaclab_arena.tasks.observations.gear_insertion_observations import NistGearInsertionPolicyObservations
    from isaaclab_arena_environments.mdp.nist_gear_insertion_osc_action import NistGearInsertionOscActionCfg

    asset_registry = AssetRegistry()
    embodiment_cls = asset_registry.get_asset_by_name("franka_nist_gear_osc")
    embodiment = embodiment_cls(fixed_asset_name="gears_and_base", peg_offset=(0.02025, 0.0, 0.025))

    assert embodiment_cls is FrankaNistGearInsertionOscEmbodiment
    assert isinstance(embodiment.action_config.arm_action, NistGearInsertionOscActionCfg)
    assert embodiment.action_config.arm_action.fixed_asset_name == "gears_and_base"
    assert embodiment.action_config.arm_action.peg_offset == (0.02025, 0.0, 0.025)
    assert embodiment.action_config.arm_action.body_name == "panda_fingertip_centered"

    assert isinstance(embodiment.observation_config, FrankaNistGearInsertionObservationsCfg)
    policy_obs = embodiment.observation_config.policy
    assert policy_obs.nist_gear_policy_obs.func is NistGearInsertionPolicyObservations
    assert policy_obs.nist_gear_policy_obs.params["board_name"] == "gears_and_base"
    assert policy_obs.nist_gear_policy_obs.params["peg_offset"] == [0.02025, 0.0, 0.025]
    assert policy_obs.nist_gear_policy_obs.params["fingertip_body_name"] == "panda_fingertip_centered"


def test_nist_object_library_uses_shared_nucleus_assets():
    from isaaclab_arena.assets.registries import AssetRegistry

    asset_registry = AssetRegistry()
    expected_paths = {
        "gears_and_base": (
            "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Arena/assets/object_library/NIST/"
            "gearbase_and_gears_gearbase_root.usd"
        ),
        "medium_nist_gear": (
            "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Arena/assets/object_library/NIST/gear_medium.usd"
        ),
        "nist_board_assembled": (
            "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Arena/assets/object_library/NIST/"
            "nist_board_assembled.usd"
        ),
        "nist_gear_base": (
            "omniverse://isaac-dev.ov.nvidia.com/Isaac/IsaacLab/Arena/assets/object_library/NIST/gear_base.usd"
        ),
    }

    for asset_name, expected_path in expected_paths.items():
        asset_cls = asset_registry.get_asset_by_name(asset_name)
        assert asset_cls.usd_path == expected_path


def test_nist_task_observation_terms_use_task_geometry():
    from isaaclab.managers import ObservationGroupCfg as ObsGroup

    from isaaclab_arena.tasks.observations.gear_insertion_observations import (
        peg_delta_from_held_gear_base,
        peg_pos_in_env_frame,
    )

    obs_cfg = _nist_task().get_observation_cfg()

    assert isinstance(obs_cfg.task_obs, ObsGroup)
    assert obs_cfg.task_obs.peg_pos.func is peg_pos_in_env_frame
    assert obs_cfg.task_obs.peg_pos.params["board_cfg"].name == "gears_and_base"
    assert obs_cfg.task_obs.peg_pos.params["peg_offset"] == [0.02025, 0.0, 0.025]

    assert obs_cfg.task_obs.peg_delta.func is peg_delta_from_held_gear_base
    assert obs_cfg.task_obs.peg_delta.params["gear_cfg"].name == "medium_nist_gear"
    assert obs_cfg.task_obs.peg_delta.params["board_cfg"].name == "gears_and_base"
    assert obs_cfg.task_obs.peg_delta.params["held_gear_base_offset"] == [0.02025, 0.0, 0.0]

    assert obs_cfg.task_obs.ee_pos_noiseless.func.__name__ == "body_pos_in_env_frame"
    assert obs_cfg.task_obs.ee_pos_noiseless.params["body_name"] == "panda_fingertip_centered"
    assert obs_cfg.task_obs.ee_quat_noiseless.func.__name__ == "body_quat_canonical"
    assert obs_cfg.task_obs.ee_quat_noiseless.params["body_name"] == "panda_fingertip_centered"
    assert obs_cfg.task_obs.concatenate_terms


def test_nist_task_reward_terms_share_insertion_geometry():
    from isaaclab_arena.tasks.rewards import gear_insertion_rewards

    rewards_cfg = _nist_task().get_rewards_cfg()

    keypoint_terms = [rewards_cfg.kp_baseline, rewards_cfg.kp_coarse, rewards_cfg.kp_fine]
    for term in keypoint_terms:
        assert term.func is gear_insertion_rewards.gear_peg_keypoint_squashing
        assert term.params["gear_cfg"].name == "medium_nist_gear"
        assert term.params["board_cfg"].name == "gears_and_base"
        assert term.params["peg_offset"] == [0.02025, 0.0, 0.0]
        assert term.params["held_gear_base_offset"] == [0.02025, 0.0, 0.0]

    assert rewards_cfg.engagement_bonus.func is gear_insertion_rewards.gear_insertion_geometry_bonus
    assert rewards_cfg.success_bonus.func is gear_insertion_rewards.gear_insertion_geometry_bonus
    assert rewards_cfg.engagement_bonus.params["z_fraction"] == 0.90
    assert rewards_cfg.success_bonus.params["z_fraction"] == 0.20
    assert rewards_cfg.success_bonus.params["xy_threshold"] == 0.0025

    assert rewards_cfg.action_penalty_asset.func is gear_insertion_rewards.osc_action_magnitude_penalty
    assert rewards_cfg.action_grad_penalty.func is gear_insertion_rewards.osc_action_delta_penalty
    assert rewards_cfg.contact_penalty.func is gear_insertion_rewards.wrist_contact_force_penalty
    assert rewards_cfg.success_pred_error.func is gear_insertion_rewards.success_prediction_error


def test_nist_task_events_use_embodiment_grasp_and_randomization_cfg():
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import franka_gripper_joint_setter
    from isaaclab_arena.tasks.events import place_gear_in_gripper
    from isaaclab_arena.tasks.nist_gear_insertion_task import GraspCfg

    embodiment = AssetRegistry().get_asset_by_name("franka_nist_gear_osc")()
    task = _nist_task(
        grasp_cfg=GraspCfg(**embodiment.get_gear_insertion_grasp_config()),
        enable_randomization=True,
    )

    events_cfg = task.get_events_cfg()

    assert events_cfg.place_gear.func is place_gear_in_gripper
    assert events_cfg.place_gear.params["gear_cfg"].name == "medium_nist_gear"
    assert events_cfg.place_gear.params["hand_grasp_width"] == 0.03
    assert events_cfg.place_gear.params["hand_close_width"] == 0.03
    assert events_cfg.place_gear.params["gripper_joint_setter_func"] is franka_gripper_joint_setter
    assert events_cfg.place_gear.params["end_effector_body_name"] == "panda_hand"
    assert events_cfg.place_gear.params["finger_joint_names"] == "panda_finger_joint[1-2]"

    assert events_cfg.fixed_asset_pose.params["asset_cfg"].name == "gears_and_base"
    assert events_cfg.held_object_mass.params["asset_cfg"].name == "medium_nist_gear"
    assert events_cfg.robot_actuator_gains.params["asset_cfg"].joint_names == "panda_joint[1-7]"
    assert events_cfg.robot_joint_friction.params["asset_cfg"].joint_names == "panda_joint[1-7]"


def test_nist_task_termination_terms_include_success_and_optional_drop_checks():
    from isaaclab_arena.embodiments.franka.franka import franka_gripper_joint_setter
    from isaaclab_arena.tasks.nist_gear_insertion_task import GraspCfg
    from isaaclab_arena.tasks.terminations import gear_dropped_from_gripper, gear_mesh_insertion_success

    terminations_cfg = _nist_task(rl_training_mode=True).get_termination_cfg()

    assert terminations_cfg.success.func is gear_mesh_insertion_success
    assert terminations_cfg.success.params["held_object_cfg"].name == "medium_nist_gear"
    assert terminations_cfg.success.params["fixed_object_cfg"].name == "gears_and_base"
    assert terminations_cfg.success.params["gear_base_offset"] == [0.02025, 0.0, 0.0]
    assert terminations_cfg.success.params["held_gear_base_offset"] == [0.02025, 0.0, 0.0]
    assert terminations_cfg.success.params["rl_training"]
    assert terminations_cfg.object_dropped is None
    assert terminations_cfg.gear_dropped_from_gripper is None

    drop_cfg = _nist_task(
        grasp_cfg=GraspCfg(gripper_joint_setter_func=franka_gripper_joint_setter),
        disable_drop_terminations=False,
    ).get_termination_cfg()

    assert drop_cfg.object_dropped is not None
    assert drop_cfg.object_dropped.params["minimum_height"] == 0.0
    assert drop_cfg.gear_dropped_from_gripper.func is gear_dropped_from_gripper
    assert drop_cfg.gear_dropped_from_gripper.params["gear_cfg"].name == "medium_nist_gear"
    assert drop_cfg.gear_dropped_from_gripper.params["ee_body_name"] == "panda_hand"


def test_nist_environment_defaults_to_nist_franka_embodiment():
    import argparse

    from isaaclab_arena_environments.nist_assembled_gearmesh_osc_environment import NISTAssembledGearMeshOSCEnvironment

    parser = argparse.ArgumentParser()
    NISTAssembledGearMeshOSCEnvironment.add_cli_args(parser)
    args_cli = parser.parse_args([])

    assert args_cli.embodiment == "franka_nist_gear_osc"
