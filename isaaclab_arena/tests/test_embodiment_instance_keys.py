# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test robot instance keys: keyed names, refused modes, keyed stepping, and unchanged unkeyed robots."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_keyed_franka_configurations(simulation_app):
    from dataclasses import fields

    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment, FrankaJointPosEmbodiment
    from isaaclab_arena.terms.actions import robot_action_rate_l2, robot_last_action
    from isaaclab_arena.utils.pose import Pose

    pose = Pose(position_xyz=(-1.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0))
    for robot_type, key in ((FrankaJointPosEmbodiment, "left"), (FrankaIKEmbodiment, "right")):
        robot = robot_type(instance_key=key, enable_cameras=True)
        robot.set_initial_pose(pose)
        robot_path = f"{{ENV_REGEX_NS}}/{key}"
        assert robot.name == key and robot.embodiment_type == robot_type.name
        assert robot.get_scene_key() == key and robot.get_scene_root_keys() == (key,)

        scene = robot.get_scene_cfg()
        assert [field.name for field in fields(scene)] == [key, f"{key}_ee_frame", f"{key}_wrist_cam"]
        assert getattr(scene, key).prim_path == robot_path
        assert getattr(scene, key).init_state.pos == (-1.0, 0.0, 0.0)
        ee_frame = getattr(scene, f"{key}_ee_frame")
        assert ee_frame.prim_path.startswith(f"{robot_path}/")
        assert all(frame.prim_path.startswith(f"{robot_path}/") for frame in ee_frame.target_frames)
        assert [frame.name for frame in ee_frame.target_frames] == [
            f"{key}_end_effector",
            f"{key}_tool_rightfinger",
            f"{key}_tool_leftfinger",
        ]
        assert getattr(scene, f"{key}_wrist_cam").prim_path.startswith(f"{robot_path}/")

        actions = robot.get_action_cfg()
        action_names = (f"{key}_arm_action", f"{key}_gripper_action")
        assert tuple(field.name for field in fields(actions)) == action_names
        assert all(getattr(actions, name).asset_name == key for name in action_names)

        observations = robot.get_observation_cfg()
        assert [field.name for field in fields(observations)] == [f"{key}_policy", "camera_obs"]
        policy = getattr(observations, f"{key}_policy")
        assert policy.actions.func is robot_last_action and policy.actions.params["action_names"] == action_names
        for term, parameter, entity in (
            (policy.joint_pos, "asset_cfg", key),
            (policy.joint_vel, "asset_cfg", key),
            (policy.eef_pos, "ee_frame_cfg", f"{key}_ee_frame"),
            (policy.eef_quat, "ee_frame_cfg", f"{key}_ee_frame"),
            (policy.gripper_pos, "robot_cfg", key),
        ):
            assert term.params[parameter].name == entity
        camera_term = getattr(observations.camera_obs, f"{key}_wrist_cam_rgb")
        assert camera_term.params["sensor_cfg"].name == f"{key}_wrist_cam"
        assert {variation.camera_name for variation in robot.get_variations()} == {f"{key}_wrist_cam"}

        events = robot.get_events_cfg()
        assert getattr(events, f"{key}_randomize_franka_joint_state").params["asset_cfg"].name == key
        assert getattr(events, f"{key}_robot_reset_pose").params["scene_writes"][0][0] == key

        rewards = robot.get_rewards_cfg()
        action_rate = getattr(rewards, f"{key}_action_rate")
        assert action_rate.func is robot_action_rate_l2 and action_rate.params["action_names"] == action_names
        assert getattr(rewards, f"{key}_joint_vel").params["asset_cfg"].name == key

        recorders = robot.get_recorder_term_cfg(record_trajectories=True)
        ee_recorder = getattr(recorders, f"{key}_record_end_effector_poses_0")
        assert ee_recorder.frame_transformer_name == f"{key}_ee_frame" and ee_recorder.asset_name == key

        gripper = robot.get_gripper()
        assert (gripper.asset_name, gripper.frame_transformer_name) == (key, f"{key}_ee_frame")
        assert gripper.target_frame_name == f"{key}_end_effector"
        assert robot.get_ee_frame_name(robot.arm_mode) == f"{key}_ee_frame"
        assert robot.get_command_body_name() == "panda_hand"

    for invalid_key in ("", "robot", "Left", "class", "two robots"):
        with pytest.raises(AssertionError, match="lowercase ASCII identifier"):
            FrankaJointPosEmbodiment(instance_key=invalid_key)
    return True


def test_keyed_franka_configurations():
    assert run_function_with_persistent_simulation_app(_test_keyed_franka_configurations)


def _test_keyed_robot_modes(simulation_app):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.assets.registries import DeviceRegistry, RetargeterRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    robot = FrankaJointPosEmbodiment(instance_key="left")
    device = SimpleNamespace(name="keyboard", get_device_cfg=lambda **kwargs: kwargs)
    converter = SimpleNamespace(get_pipeline_builder=lambda embodiment: embodiment)
    with patch.object(RetargeterRegistry, "get_component_by_name", return_value=lambda: converter) as lookup:
        DeviceRegistry().get_teleop_device_cfg(device, robot)
    lookup.assert_called_once_with(RetargeterRegistry().convert_tuple_to_str(("keyboard", "franka_joint_pos")))

    for mode in ("mimic", "teleop", "xr"):
        definition = IsaacLabArenaEnvironment("keyed_robot_mode", Scene(assets=[]), embodiments=[robot])
        definition.teleop_device = device if mode == "teleop" else None
        builder = ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(mimic=mode == "mimic", solve_relations=False))
        with patch("isaaclab_arena.environments.arena_env_builder.get_settings_manager") as settings:
            settings.return_value.get.return_value = mode == "xr"
            with pytest.raises(AssertionError, match="require an embodiment without an instance key"):
                builder.compose_manager_cfg()
    return True


def test_keyed_robot_modes():
    assert run_function_with_persistent_simulation_app(_test_keyed_robot_modes)


def _test_keyed_franka_steps(simulation_app):
    import torch

    from isaaclab_arena.embodiments.franka.franka import FrankaIKEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.cube_goal_pose_environment import (
        CubeGoalPoseEnvironment,
        CubeGoalPoseEnvironmentCfg,
    )

    definition = CubeGoalPoseEnvironment().build(CubeGoalPoseEnvironmentCfg())
    definition.embodiments = [
        FrankaIKEmbodiment(instance_key="arm", initial_pose=definition.embodiments[0].get_initial_pose())
    ]
    env = ArenaEnvBuilder(
        definition, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, record_trajectories=True)
    ).make_registered()
    try:
        env.reset()
        assert "arm" in env.unwrapped.scene.articulations
        assert "arm_ee_frame" in env.unwrapped.scene.sensors
        assert env.unwrapped.action_manager.active_terms == ["arm_arm_action", "arm_gripper_action"]
        assert "arm_policy" in env.unwrapped.observation_manager.active_terms
        for _ in range(2):
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    finally:
        env.close()
    return True


def test_keyed_franka_steps():
    assert run_function_with_persistent_simulation_app(_test_keyed_franka_steps)


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


def _test_external_franka_configuration(simulation_app):
    from isaaclab_arena_examples.external_environments.advanced import SoftFrankaIKEmbodiment

    for key in (None, "soft_arm"):
        robot = SoftFrankaIKEmbodiment(instance_key=key)
        articulation = getattr(robot.get_scene_cfg(), key or "robot")
        for actuator_name in ("panda_shoulder", "panda_forearm"):
            assert articulation.actuators[actuator_name].stiffness == 200.0
            assert articulation.actuators[actuator_name].damping == 40.0
    return True


def test_external_franka_configuration():
    assert run_function_with_persistent_simulation_app(_test_external_franka_configuration)
