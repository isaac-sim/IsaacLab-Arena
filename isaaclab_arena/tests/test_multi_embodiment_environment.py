# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Exercise robot composition, recorded identity, and single-robot compatibility."""

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def make_two_robot_definition(enable_cameras=False, mixed=False, relations=False):
    """Build two spaced robots with optional cameras and placement relations."""
    from isaaclab_arena.assets.registries import AssetRegistry
    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.relations import AtPosition, IsAnchor
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    registry = AssetRegistry()
    robots = [FrankaJointPosEmbodiment(instance_key="left", enable_cameras=enable_cameras)]
    robots.append(
        registry.get_asset_by_name("g1_wbc_joint")()
        if mixed
        else FrankaJointPosEmbodiment(instance_key="right", enable_cameras=enable_cameras)
    )
    for index, robot in enumerate(robots):
        x = -2.0 if index == 0 else 2.0
        if relations:
            robot.add_relation(AtPosition(x=x, y=0.0, z=0.0))
        else:
            robot.set_initial_pose(
                Pose(
                    position_xyz=(x, 0.0, 0.78 if mixed and index else 0.0),
                    rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
                )
            )
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
        placer_params=ObjectPlacerParams(min_unique_layouts_per_env=1),
    )


def _test_two_robots(simulation_app, output_dir, cameras=False, mixed=False):
    import h5py
    import json
    import torch
    from dataclasses import fields
    from unittest.mock import patch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.relation_solver_interface import solve_and_apply_relation_placement
    from isaaclab_arena.terms.recorders import TrajectoryRecorderTermsBaseCfg

    definition = make_two_robot_definition(enable_cameras=cameras, mixed=mixed, relations=cameras)
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
        if mixed:
            assert env.unwrapped.cfg.events.reset_all.params["asset_cfg"].name == "robot"
            assert "asset_cfg" not in definition.embodiments[1].event_config.reset_all.params
            left = env.unwrapped.scene["left"]
            assert not torch.equal(left.data.joint_pos.torch, left.data.default_joint_pos.torch)
        if cameras:
            for key, x in (("left", -2.0), ("right", 2.0)):
                position = env.unwrapped.scene[key].data.root_pos_w.torch - env.unwrapped.scene.env_origins
                assert torch.allclose(position[:, 0], torch.full_like(position[:, 0], x), atol=0.02)
            assert set(observations["camera_obs"]) == {"left_wrist_cam_rgb", "right_wrist_cam_rgb"}
            assert env.unwrapped.cfg.demo_recorder_config is not None
            assert {"left", "right"} <= set(builder.get_all_variations())
            assert (
                env.unwrapped.cfg.events.left_wrist_cam_extrinsics_variation.params["asset_cfg"].name
                == "left_wrist_cam"
            )
            assert (
                env.unwrapped.cfg.events.right_wrist_cam_extrinsics_variation.params["asset_cfg"].name
                == "right_wrist_cam"
            )
        recorders = env.unwrapped.cfg.recorders
        for field in fields(TrajectoryRecorderTermsBaseCfg):
            assert getattr(recorders, field.name) is not None
            assert not hasattr(recorders, f"left_{field.name}")
            assert not hasattr(recorders, f"right_{field.name}")
        assert recorders.left_record_end_effector_poses_0.asset_name == "left"
        if not mixed:
            assert recorders.right_record_end_effector_poses_0.asset_name == "right"
        assert env.action_space.shape[-1] == sum(env.unwrapped.action_manager.action_term_dim)
        widths = {}
        for key in keys:
            widths[key] = sum(
                width
                for name, width in zip(
                    env.unwrapped.action_manager.active_terms, env.unwrapped.action_manager.action_term_dim, strict=True
                )
                if env.unwrapped.action_manager.get_term(name).cfg.asset_name == key
            )
        assert all(width > 0 for width in widths.values())
        if not mixed:
            assert widths == {"left": 8, "right": 8}
        for _ in range(2):
            observations, _, _, _, _ = env.step(torch.zeros(env.action_space.shape, device=env.unwrapped.device))
        if mixed:
            assert observations["policy"]["actions"].shape[-1] == widths["robot"]
        env.reset()
    finally:
        env.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    expected = {robot.get_scene_key(): robot.embodiment_type for robot in definition.embodiments}
    assert all(record["embodiments"] == expected for record in records)
    if not mixed:
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


def test_keyed_franka_with_unkeyed_g1(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_two_robots, mixed=True, output_dir=tmp_path)


def _test_single_compatibility(simulation_app):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.tests.utils.legacy_single_embodiment_builder import LegacySingleEmbodimentBuilder
    from isaaclab_arena_environments.cube_goal_pose_environment import (
        CubeGoalPoseEnvironment,
        CubeGoalPoseEnvironmentCfg,
    )

    cfg = ArenaEnvBuilderCfg(
        num_envs=2, solve_relations=False, record_trajectories=True, recorder_dataset_filename="single_reference"
    )
    factory = CubeGoalPoseEnvironment()
    before, _ = LegacySingleEmbodimentBuilder(
        factory.build(CubeGoalPoseEnvironmentCfg(enable_cameras=True)), cfg
    ).compose_manager_cfg()
    after, _ = ArenaEnvBuilder(
        factory.build(CubeGoalPoseEnvironmentCfg(enable_cameras=True)), cfg
    ).compose_manager_cfg()
    assert after.to_dict() == before.to_dict()
    assert after.episode_recorders.core.params["embodiments"] == {"robot": "franka_ik"}
    return True


def test_single_robot_matches_prechange_assembly():
    assert run_function_with_persistent_simulation_app(_test_single_compatibility)


def _test_invalid_compositions(simulation_app):
    from types import SimpleNamespace
    from unittest.mock import patch

    from isaaclab_arena.embodiments.franka.franka import FrankaJointPosEmbodiment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    for robots in (
        [FrankaJointPosEmbodiment(), FrankaJointPosEmbodiment()],
        [FrankaJointPosEmbodiment(instance_key="arm"), FrankaJointPosEmbodiment(instance_key="arm")],
    ):
        with pytest.raises(AssertionError, match="unique|unkeyed"):
            IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiments=robots)
    with pytest.raises(AssertionError, match="not both"):
        IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiment=FrankaJointPosEmbodiment(), embodiments=[])
    unkeyed = [FrankaJointPosEmbodiment(), FrankaJointPosEmbodiment()]
    unkeyed[1].get_scene_key = lambda: "custom_articulation"
    with pytest.raises(AssertionError, match="unkeyed"):
        IsaacLabArenaEnvironment("invalid", Scene(assets=[]), embodiments=unkeyed)
    definition = make_two_robot_definition()
    with pytest.raises(AssertionError, match="Use embodiments"):
        _ = definition.embodiment
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
        with pytest.raises(AssertionError, match="XR requires exactly one"):
            ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    definition.embodiments.append(definition.embodiments[0])
    with pytest.raises(AssertionError, match="unique"):
        ArenaEnvBuilder(definition, ArenaEnvBuilderCfg()).compose_manager_cfg()
    definition.embodiment = FrankaJointPosEmbodiment()
    assert definition.embodiments == [definition.embodiment]
    definition.embodiment = None
    assert definition.embodiments == []
    return True


def test_invalid_compositions_fail_before_building():
    assert run_function_with_persistent_simulation_app(_test_invalid_compositions)
