# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Test environments with several robots: composition, recorded identity, and refused compositions."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def make_two_robot_definition(enable_cameras=False, relations=False):
    """Build two spaced keyed Frankas with optional cameras and placement relations."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
    from isaaclab_arena.relations.relations import AtPosition, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    robots = [FrankaJointPosEmbodiment(instance_key=key, enable_cameras=enable_cameras) for key in ("left", "right")]
    for robot, x in zip(robots, (-2.0, 2.0)):
        if relations:
            robot.add_relation(AtPosition(x=x, y=0.0, z=0.0))
        else:
            robot.set_initial_pose(Pose(position_xyz=(x, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        if enable_cameras:
            robot.get_variations()[0].enable()
    assets = [registry.get_asset_by_name(name)() for name in ("ground_plane", "light")]
    if relations:
        anchor = registry.get_asset_by_name("dex_cube")()
        anchor.set_initial_pose(Pose(position_xyz=(0.0, 5.0, 0.5), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
        anchor.add_relation(IsAnchor())
        assets.append(anchor)
    return IsaacLabArenaEnvironment(
        name="multi_robot_test",
        scene=Scene(assets=assets),
        embodiments=robots,
        # Allow both robots to travel from the shared anchor to their requested positions.
        placer_params=ObjectPlacerParams(
            solver_params=RelationSolverParams(max_iters=4000), min_unique_layouts_per_env=1
        ),
    )


def _test_two_robots(simulation_app, output_dir, cameras=False):
    import h5py
    import json
    import torch
    from dataclasses import fields
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.terms.recorders import TrajectoryRecorderTermsBaseCfg

    definition = make_two_robot_definition(enable_cameras=cameras, relations=cameras)
    definition.embodiments.reverse()
    builder = ArenaEnvBuilder(
        definition,
        ArenaEnvBuilderCfg(
            num_envs=2,
            solve_relations=cameras,
            record_trajectories=True,
            recorder_dataset_export_dir_path=str(output_dir),
            recorder_dataset_filename="two_robots",
        ),
    )
    with patch(
        "isaaclab_arena.environments.arena_env_builder.solve_and_apply_relation_placement",
        wraps=solve_and_apply_relation_placement,
    ) as solve:
        env = builder.make_registered()
        if cameras:
            assert {"left", "right"} <= {asset.get_scene_key() for asset in solve.call_args.args[0]}
    path = output_dir / "episodes.jsonl"
    keys = {robot.get_scene_key() for robot in definition.embodiments}
    try:
        env.unwrapped.episode_recorder.set_output_path(path)
        observations, _ = env.reset()
        assert set(env.unwrapped.scene.articulations) == keys
        if cameras:
            for key, x in (("left", -2.0), ("right", 2.0)):
                position = env.unwrapped.scene[key].data.root_pos_w.torch - env.unwrapped.scene.env_origins
                assert torch.allclose(position[:, 0], torch.full_like(position[:, 0], x), atol=0.02)
            assert set(observations["camera_obs"]) == {"left_wrist_cam_rgb", "right_wrist_cam_rgb"}
            assert env.unwrapped.cfg.demo_recorder_config is not None
            assert {"left", "right"} <= set(builder.get_all_variations())
            for key in ("left", "right"):
                variation = getattr(env.unwrapped.cfg.events, f"{key}_wrist_cam_extrinsics_variation")
                assert variation.params["asset_cfg"].name == f"{key}_wrist_cam"
        # The builder records the scene-wide trajectory terms once, and each robot its own end effector.
        recorders = env.unwrapped.cfg.recorders
        for field in fields(TrajectoryRecorderTermsBaseCfg):
            assert getattr(recorders, field.name) is not None
            assert not hasattr(recorders, f"left_{field.name}")
            assert not hasattr(recorders, f"right_{field.name}")
        assert recorders.left_record_end_effector_poses_0.asset_name == "left"
        assert recorders.right_record_end_effector_poses_0.asset_name == "right"
        assert env.action_space.shape[-1] == sum(env.unwrapped.action_manager.action_term_dim)
        assert env.unwrapped.action_manager.active_terms == [
            "right_arm_action",
            "right_gripper_action",
            "left_arm_action",
            "left_gripper_action",
        ]
        manager = env.unwrapped.action_manager
        widths = dict.fromkeys(keys, 0)
        for name, width in zip(manager.active_terms, manager.action_term_dim, strict=True):
            widths[manager.get_term(name).cfg.asset_name] += width
        assert widths == {"left": 8, "right": 8}
        # Each robot's action observation shows only its own raw action columns.
        for right_action in (0.25, -0.5):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, :8] = right_action
            observations, _, _, _, _ = env.step(actions)
            torch.testing.assert_close(observations["right_policy"]["actions"], actions[:, :8])
            torch.testing.assert_close(observations["left_policy"]["actions"], actions[:, 8:])
        # Each robot's action-rate reward reads only its own action columns.
        manager.action.zero_()
        manager.prev_action.zero_()
        manager.action[:, :8] = 1.0
        rewards = env.unwrapped.cfg.rewards
        assert torch.equal(
            rewards.right_action_rate.func(env.unwrapped, **rewards.right_action_rate.params),
            torch.full((2,), 8.0, device=env.unwrapped.device),
        )
        assert torch.equal(
            rewards.left_action_rate.func(env.unwrapped, **rewards.left_action_rate.params),
            torch.zeros(2, device=env.unwrapped.device),
        )
        env.reset()
    finally:
        env.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    expected = {robot.get_scene_key(): robot.embodiment_type for robot in definition.embodiments}
    assert all(record["embodiments"] == expected for record in records)
    with h5py.File(output_dir / "two_robots.hdf5", "r") as dataset:
        assert len(dataset["data"]) == 2
        for demo in dataset["data"].values():
            for key in ("left", "right"):
                assert demo[f"states/kinematics/{key}_end_effector/position"].shape == (2, 3)
                assert demo[f"initial_state/kinematics/{key}_end_effector/position"].shape == (1, 3)
    if cameras:
        for record in records:
            assert any(name.startswith("left.") for name in record["variations"])
            assert any(name.startswith("right.") for name in record["variations"])
    return True


@pytest.mark.with_cameras
def test_two_keyed_frankas_with_cameras_and_relations(tmp_path):
    assert run_function_with_persistent_simulation_app(
        _test_two_robots, enable_cameras=True, cameras=True, output_dir=tmp_path
    )


def test_two_keyed_frankas(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_two_robots, output_dir=tmp_path)


def _test_single_robot_composition(simulation_app):
    from dataclasses import fields

    from isaaclab.managers import RecorderTermCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.terms.recorders import TrajectoryRecorderTermsBaseCfg
    from isaaclab_arena_environments.cube_goal_pose_environment import (
        CubeGoalPoseEnvironment,
        CubeGoalPoseEnvironmentCfg,
    )

    definition = CubeGoalPoseEnvironment().build(CubeGoalPoseEnvironmentCfg())
    cfg, _ = ArenaEnvBuilder(
        definition,
        ArenaEnvBuilderCfg(
            num_envs=2, solve_relations=False, record_trajectories=True, recorder_dataset_filename="single"
        ),
    ).compose_manager_cfg()

    assert {"robot", "ee_frame"} <= {field.name for field in fields(cfg.scene)}
    assert [field.name for field in fields(cfg.actions)] == ["arm_action", "gripper_action"]
    # Robot events run before the scene's.
    event_names = [field.name for field in fields(cfg.events)]
    scene_event_names = [name for name in event_names if name in definition.scene.assets]
    assert scene_event_names and event_names.index("robot_reset_pose") < event_names.index(scene_event_names[0])
    # Environment recorders come first, once, and the robot's own end-effector poses last.
    recorder_names = [
        field.name for field in fields(cfg.recorders) if isinstance(getattr(cfg.recorders, field.name), RecorderTermCfg)
    ]
    trajectory_names = [field.name for field in fields(TrajectoryRecorderTermsBaseCfg)]
    assert recorder_names[-1] == "record_end_effector_poses_0"
    assert recorder_names[-1 - len(trajectory_names) : -1] == trajectory_names
    assert cfg.episode_recorders.core.params["embodiments"] == {"robot": "franka_ik"}
    return True


def test_single_robot_composition():
    assert run_function_with_persistent_simulation_app(_test_single_robot_composition)


def _test_invalid_compositions(simulation_app):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    # Several robots each need an instance key, even when their scene keys differ.
    for robots in (
        [FrankaJointPosEmbodiment(), FrankaJointPosEmbodiment()],
        [FrankaJointPosEmbodiment(instance_key="left"), FrankaJointPosEmbodiment()],
    ):
        with pytest.raises(AssertionError, match="instance key on every robot"):
            IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiments=robots)
    with pytest.raises(AssertionError, match="unique"):
        IsaacLabArenaEnvironment(
            "invalid",
            Scene(assets=[]),
            embodiments=[FrankaJointPosEmbodiment(instance_key="arm"), FrankaJointPosEmbodiment(instance_key="arm")],
        )
    # No robot, or one robot alone, needs no key.
    for robots in ([], [FrankaJointPosEmbodiment()]):
        IsaacLabArenaEnvironment("valid", Scene(assets=[]), embodiments=robots)

    definition = make_two_robot_definition()
    with pytest.raises(AssertionError, match="exactly one"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(mimic=True)).compose_manager_cfg()
    definition.teleop_device = object()
    with pytest.raises(AssertionError, match="exactly one"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    definition.teleop_device = None
    with patch(
        "isaaclab_arena.environments.arena_env_builder.get_settings_manager",
        return_value=SimpleNamespace(get=lambda *args: True),
    ):
        with pytest.raises(AssertionError, match="XR require exactly one"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    # The builder checks robots added after construction as well.
    definition.embodiments.append(definition.embodiments[0])
    with pytest.raises(AssertionError, match="unique"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    return True


def test_invalid_compositions_fail_before_building():
    assert run_function_with_persistent_simulation_app(_test_invalid_compositions)


def _test_observation_precedence(simulation_app):
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.utils.cameras import combine_observation_cfgs
    from isaaclab_arena.utils.configclass import make_configclass

    ordinary_type = make_configclass("PolicyObservations", [("value", int, 1)], bases=(ObservationGroupCfg,))
    base = make_configclass("RobotObservations", [("policy", ordinary_type, ordinary_type())])()
    override = make_configclass("TaskObservations", [("policy", ordinary_type, ordinary_type(value=2))])()
    cameras = []
    for name in ("left_image", "right_image"):
        group = make_configclass(
            "CameraGroup",
            [(name, ObservationTermCfg, ObservationTermCfg(func=lambda env: None))],
            bases=(ObservationGroupCfg,),
        )()
        cameras.append(make_configclass("Cameras", [("camera_obs", type(group), group)])())
    for count in range(3):
        # A later ordinary group overrides an earlier one whatever the number of camera groups.
        combined = combine_observation_cfgs(base, *cameras[:count], override)
        assert combined.policy.value == 2
        assert base.policy.value == 1
        if count == 2:
            assert hasattr(combined.camera_obs, "left_image") and hasattr(combined.camera_obs, "right_image")
        # Two robots must not contribute the same ordinary group.
        definition = make_two_robot_definition()
        for index, robot in enumerate(definition.embodiments):
            robot_observations = combine_observation_cfgs(base, cameras[index] if index < count else None)
            robot.get_observation_cfg = lambda cfg=robot_observations: cfg
        with pytest.raises(AssertionError, match="same observation groups"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(solve_relations=False)).compose_manager_cfg()
    cameras[1].camera_obs.enable_corruption = not cameras[0].camera_obs.enable_corruption
    with pytest.raises(AssertionError, match="Camera groups disagree"):
        combine_observation_cfgs(*cameras)
    return True


def test_observation_overrides_do_not_depend_on_camera_count():
    assert run_function_with_persistent_simulation_app(_test_observation_precedence)


def _test_multiple_robots_accept_independent_composite_tasks(simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
    from isaaclab_arena.tasks.composite_task_base import CompositeTaskBase
    from isaaclab_arena.tasks.no_task import NoTask
    from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

    class ObservedTask(NoTask):
        def get_termination_cfg(self):
            return TaskTerminationCfg(
                timeout_s=10.0, success=[CompletionCriteria(name="done", predicate_sequence=[lambda env: True])]
            )

        def get_metrics(self):
            return []

    class BodySpecificTask(ObservedTask):
        def configure_for_embodiment(self, embodiment):
            raise AssertionError("An ambiguous robot must never reach this hook")

    class BodySpecificComposite(CompositeTaskBase):
        def configure_for_embodiment(self, embodiment):
            raise AssertionError("An ambiguous robot must never reach this hook")

    definition = make_two_robot_definition()
    definition.task = CompositeTaskBase([ObservedTask(), ObservedTask()])
    cfg, _ = ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).compose_manager_cfg()
    assert cfg.scene.left is not None and cfg.scene.right is not None
    for task in (
        BodySpecificTask(),
        CompositeTaskBase([BodySpecificTask()]),
        BodySpecificComposite([ObservedTask()]),
    ):
        definition.task = task
        with pytest.raises(AssertionError, match="embodiment-specific configuration require one robot"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).compose_manager_cfg()
    return True


def test_multiple_robots_accept_independent_composite_tasks():
    assert run_function_with_persistent_simulation_app(_test_multiple_robots_accept_independent_composite_tasks)
