# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test named robot configurations and controls."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_unkeyed_franka_defaults(simulation_app, embodiment_name):
    from dataclasses import fields

    import isaaclab.envs.mdp as mdp
    from isaaclab.managers import ObservationTermCfg

    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment, FrankaJointPosEmbodiment
    from isaaclab_arena.utils.pose import Pose

    robot_type = {"franka_ik": FrankaIKEmbodiment, "franka_joint_pos": FrankaJointPosEmbodiment}[embodiment_name]
    robot = robot_type(enable_cameras=True)
    robot.set_initial_pose(Pose.identity())
    assert robot.name == embodiment_name and robot.get_scene_key() == "robot"

    scene = robot.get_scene_cfg()
    assert [field.name for field in fields(scene)] == ["robot", "ee_frame", "wrist_cam"]
    assert scene.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert scene.ee_frame.prim_path.startswith("{ENV_REGEX_NS}/Robot/")
    assert scene.wrist_cam.prim_path.startswith("{ENV_REGEX_NS}/Robot/")
    assert [frame.name for frame in scene.ee_frame.target_frames] == [
        "end_effector",
        "tool_rightfinger",
        "tool_leftfinger",
    ]
    actions = robot.get_action_cfg()
    assert [field.name for field in fields(actions)] == ["arm_action", "gripper_action"]
    assert actions.arm_action.asset_name == actions.gripper_action.asset_name == "robot"
    observations = robot.get_observation_cfg()
    assert [field.name for field in fields(observations)] == ["policy", "camera_obs"]
    policy = observations.policy
    assert [field.name for field in fields(policy) if isinstance(getattr(policy, field.name), ObservationTermCfg)] == [
        "actions",
        "joint_pos",
        "joint_vel",
        "eef_pos",
        "eef_quat",
        "gripper_pos",
    ]
    assert not policy.concatenate_terms
    assert observations.camera_obs.wrist_cam_rgb.params["sensor_cfg"].name == "wrist_cam"
    assert [field.name for field in fields(robot.get_events_cfg())] == [
        "randomize_franka_joint_state",
        "robot_reset_pose",
    ]
    rewards = robot.get_rewards_cfg()
    assert [field.name for field in fields(rewards)] == ["action_rate", "joint_vel"]
    assert rewards.joint_vel.params["asset_cfg"].name == "robot"
    recorder = robot.get_recorder_term_cfg(record_trajectories=True).record_end_effector_poses_0
    assert recorder.asset_name == "robot" and recorder.frame_transformer_name == "ee_frame"

    # Unkeyed robots keep the whole action tensor in observations and action-rate rewards.
    assert policy.actions.func is mdp.last_action and not policy.actions.params
    assert rewards.action_rate.func is mdp.action_rate_l2 and not rewards.action_rate.params
    return True


@pytest.mark.parametrize("embodiment_name", ["franka_ik", "franka_joint_pos"])
def test_unkeyed_franka_defaults(embodiment_name):
    assert run_function_with_persistent_simulation_app(_test_unkeyed_franka_defaults, embodiment_name=embodiment_name)
