# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Check independent robot configurations and the unchanged unkeyed interface."""

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_instance_configurations(simulation_app):
    import copy
    from dataclasses import fields

    import pytest
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.utils.instance_rename import rename_instance_cfg, robot_last_action
    from isaaclab_arena.utils.pose import Pose

    original = FrankaJointPosEmbodiment()
    saved = copy.deepcopy(original.observation_config.to_dict())
    left = FrankaJointPosEmbodiment(instance_key="left", enable_cameras=True)
    right = FrankaJointPosEmbodiment(instance_key="right", enable_cameras=True)
    left.set_initial_pose(Pose(position_xyz=(-1.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    for robot, key in ((left, "left"), (right, "right")):
        scene = robot.get_scene_cfg()
        assert getattr(scene, key).prim_path == f"{{ENV_REGEX_NS}}/{key.title()}"
        assert not hasattr(scene, "robot")
        assert robot.name == key and robot.embodiment_type == "franka_joint_pos"
        assert all(field.name == key or field.name.startswith(f"{key}_") for field in fields(scene))
        actions = robot.get_action_cfg()
        assert all(getattr(actions, field.name).asset_name == key for field in fields(actions))
        observations = robot.get_observation_cfg()
        policy = getattr(observations, f"{key}_policy")
        assert policy.joint_pos.params["asset_cfg"].name == key
        assert policy.eef_pos.params["ee_frame_cfg"].name == f"{key}_ee_frame"
        assert policy.actions.func is robot_last_action
        assert policy.actions.params["action_names"] == (f"{key}_arm_action", f"{key}_gripper_action")
        assert all(variation.camera_name.startswith(f"{key}_") for variation in robot.get_variations())
        assert getattr(robot.get_rewards_cfg(), f"{key}_joint_vel").params["asset_cfg"].name == key
    camera_group = left.get_observation_cfg().camera_obs
    assert not hasattr(left.get_observation_cfg(), "left_camera_obs")
    assert camera_group.left_wrist_cam_rgb.params["sensor_cfg"].name == "left_wrist_cam"
    assert left.get_scene_cfg().left_wrist_cam.prim_path.startswith("{ENV_REGEX_NS}/Left/")
    unnamed = FrankaJointPosEmbodiment(instance_key="unnamed")
    unnamed.scene_config.ee_frame.target_frames[0].name = None
    implicit_target = unnamed.scene_config.ee_frame.target_frames[0].prim_path.rsplit("/", 1)[-1]
    assert unnamed.get_scene_cfg().unnamed_ee_frame.target_frames[0].name == f"unnamed_{implicit_target}"
    original_frames = [frame.name for frame in original.get_scene_cfg().ee_frame.target_frames]
    assert len(original_frames) == 3
    for robot, key in ((left, "left"), (right, "right")):
        for _ in range(2):
            sensor = getattr(robot.get_scene_cfg(), f"{key}_ee_frame")
            assert [frame.name for frame in sensor.target_frames] == [f"{key}_{name}" for name in original_frames]
        assert [frame.name for frame in robot.scene_config.ee_frame.target_frames] == original_frames
    assert {variation.name for variation in left.get_variations()} == {
        "camera_extrinsics_wrist_cam",
        "camera_intrinsics_wrist_cam",
    }
    recorders = left.get_recorder_term_cfg(record_trajectories=True)
    assert recorders.left_record_end_effector_poses_0.asset_name == "left"
    assert recorders.left_record_end_effector_poses_0.frame_transformer_name == "left_ee_frame"
    assert all(field.name.startswith("left_") for field in fields(recorders))
    reset = left.get_events_cfg().left_robot_reset_pose
    assert reset.params["scene_writes"][0][0] == "left"
    left_scene = left.get_scene_cfg()
    left_scene.left.init_state.pos = (99.0, 0.0, 0.0)
    assert left.get_scene_cfg().left.init_state.pos != left_scene.left.init_state.pos
    assert right.scene_config.robot.prim_path == "{ENV_REGEX_NS}/Robot"
    assert original.get_observation_cfg().to_dict() == saved
    for getter, attribute in (
        ("get_scene_cfg", "scene_config"),
        ("get_action_cfg", "action_config"),
        ("get_events_cfg", "event_config"),
        ("get_rewards_cfg", "reward_config"),
    ):
        assert getattr(original, getter)().to_dict() == getattr(original, attribute).to_dict()
    invalid = copy.deepcopy(original.observation_config)
    invalid.policy.joint_pos.params = {}
    with pytest.raises(AssertionError, match="joint_pos.*asset_cfg.*robot"):
        rename_instance_cfg(invalid, "left", ("robot", "ee_frame"), ("arm_action", "gripper_action"), "observations")
    g1 = AssetRegistry().get_asset_by_name("g1_wbc_joint")(instance_key="humanoid")
    with pytest.raises(AssertionError, match="literal action-term lookup 'g1_action'"):
        g1.get_events_cfg()
    left.action_config.gripper_action = None
    assert left.get_observation_cfg().left_policy.actions.params["action_names"] == ("left_arm_action",)
    with pytest.raises(AssertionError, match="lowercase ASCII"):
        FrankaJointPosEmbodiment(instance_key="Left").get_scene_cfg()
    import torch
    from types import SimpleNamespace

    actions = {
        "left_arm_action": SimpleNamespace(raw_actions=torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
        "left_gripper_action": SimpleNamespace(raw_actions=torch.tensor([[5.0], [6.0]])),
        "right_arm_action": SimpleNamespace(raw_actions=torch.tensor([[90.0], [91.0]])),
    }
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=actions.__getitem__))
    assert torch.equal(
        robot_last_action(env, ("left_arm_action", "left_gripper_action")),
        torch.tensor([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]]),
    )

    def keyword_lookup(env):
        return env.action_manager.get_term(name="arm_action").raw_actions

    class HiddenSceneLookup:
        def __init__(self, cfg, env):
            self.robot = env.scene["robot"]

        def __call__(self, env):
            return self.robot

    for func, message in ((keyword_lookup, "literal action-term lookup"), (HiddenSceneLookup, "literal scene lookup")):
        invalid.policy.joint_pos.func = func
        with pytest.raises(AssertionError, match=message):
            rename_instance_cfg(invalid, "left", ("robot", "ee_frame"), ("arm_action",), "observations")
    # The explicit parameter remains an ordinary Isaac Lab entity configuration.
    assert isinstance(left.get_observation_cfg().left_policy.joint_pos.params["asset_cfg"], SceneEntityCfg)
    return True


def test_instance_configurations():
    assert run_function_with_persistent_simulation_app(_test_instance_configurations)


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
    definition.embodiment = FrankaIKEmbodiment(
        instance_key="arm", initial_pose=definition.embodiment.get_initial_pose()
    )
    env = ArenaEnvBuilder(
        definition, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, record_trajectories=True)
    ).make_registered()
    try:
        env.reset()
        assert "arm" in env.unwrapped.scene.articulations
        for _ in range(2):
            env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
    finally:
        env.close()
    return True


def test_keyed_franka_steps():
    assert run_function_with_persistent_simulation_app(_test_keyed_franka_steps)


def _test_unkeyed_reference(simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tests.utils.legacy_franka_configuration import LegacyFrankaIKEmbodiment
    from isaaclab_arena_environments.cube_goal_pose_environment import (
        CubeGoalPoseEnvironment,
        CubeGoalPoseEnvironmentCfg,
    )

    actual_definition = CubeGoalPoseEnvironment().build(CubeGoalPoseEnvironmentCfg(enable_cameras=True))
    reference_definition = CubeGoalPoseEnvironment().build(CubeGoalPoseEnvironmentCfg(enable_cameras=True))
    reference = LegacyFrankaIKEmbodiment(enable_cameras=True)
    reference.set_initial_pose(reference_definition.embodiment.get_initial_pose())
    reference.set_initial_joint_pose([0.0444, -0.1894, -0.1107, -2.5148, 0.0044, 2.3775, 0.6952, 0.0400, 0.0400])
    reference_definition.embodiment = reference
    cfg = ArenaEnvBuilderCfg(
        num_envs=1, solve_relations=False, record_trajectories=True, recorder_dataset_filename="unkeyed_reference"
    )
    actual, _ = ArenaEnvBuilder(actual_definition, cfg).compose_manager_cfg()
    expected, _ = ArenaEnvBuilder(reference_definition, cfg).compose_manager_cfg()
    assert actual.to_dict() == expected.to_dict()
    return True


def test_unkeyed_matches_prechange_composed_configuration():
    assert run_function_with_persistent_simulation_app(_test_unkeyed_reference)
