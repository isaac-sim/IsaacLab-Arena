# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Offline ClutterOn settling and companion-cache replay."""


import pytest

from isaaclab_arena.tests.clutter.test_settled_scene import SOURCE, _arguments
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app


def _test_companion_cache_round_trip(simulation_app, tmp_path):
    import torch
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena.scripts.generate_clutter_scene import generate_scene

    data = yaml.safe_load(SOURCE.read_text())
    _, assets = build_arena_env_with_assets_from_graph_spec(ArenaEnvGraphSpec.from_yaml(SOURCE))
    support = assets["table"].get_world_bounding_box()
    cube = assets["cube_3"].get_bounding_box()
    passive_position = [
        float(support.max_point[0, 0]) - 0.1,
        float(support.max_point[0, 1]) - 0.1,
        float(support.max_point[0, 2] - cube.min_point[0, 2]),
    ]
    data["objects"][3]["params"] = {"initial_pose": {"position_xyz": passive_position}}
    data["relations"] = [relation for relation in data["relations"] if relation["subject"] != "cube_3"]
    data["relations"].append({"kind": "is_anchor", "subject": "cube_3"})
    data["placement_validators"] = {
        "enabled_checks": ["no_overlap", "on_relation"],
        "required_checks": ["no_overlap", "on_relation"],
    }
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    path = tmp_path / "poses.jsonl"
    args = _arguments(path)
    args.env_spec = source
    args.num_layouts = 4

    generate_scene(args)
    cache = PlacementLayouts.from_episode_jsonl(path)
    assert "cube_3" in cache.poses
    assert cache.num_layouts == 4
    assert cache.poses["cube_0"][0] != cache.poses["cube_0"][1]
    spec = ArenaEnvGraphSpec.from_yaml(source)
    with (
        patch.object(RelationSolver, "solve", side_effect=AssertionError("Cached replay must not solve")),
        patch.object(
            ObjectPlacer, "_validate_candidates", side_effect=AssertionError("Cached replay must not revalidate")
        ),
    ):
        arena_env = spec.to_arena_env(placement_layouts_path=path)
        assert arena_env.scene.assets["cube_3"].has_pose_reset_event()
        arena_env.embodiment.set_initial_pose(arena_env.embodiment.get_initial_pose())
        env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=3)).make_registered()
        try:
            scene = env.unwrapped.scene
            world = env.unwrapped.arena_world
            device = env.unwrapped.device
            cached_assets = {
                asset.get_scene_key(): asset
                for asset in arena_env.scene.assets.values()
                if asset.get_scene_key() in cache.poses
            }
            assert cached_assets.keys() == cache.poses.keys()
            assert all(not asset.has_pose_reset_event() for asset in cached_assets.values())
            robot = scene.articulations["robot"]
            robot_pose = robot.data.root_pose_w.torch.clone()
            for iteration in range(4):
                moved = robot_pose.clone()
                moved[:, 0] += 0.25
                robot.write_root_pose_to_sim(moved)
                for body in scene.rigid_objects.values():
                    displaced = body.data.root_pose_w.torch.clone()
                    displaced[:, 2] += 1.0
                    body.write_root_pose_to_sim(displaced)
                    body.write_root_velocity_to_sim(torch.ones_like(body.data.root_vel_w.torch))
                env.reset()
                torch.testing.assert_close(robot.data.root_pose_w.torch, robot_pose, atol=2e-5, rtol=0)
                for name, poses in cache.poses.items():
                    expected = torch.stack(
                        [poses[(iteration * 3 + i) % cache.num_layouts].to_tensor(device) for i in range(3)]
                    )
                    torch.testing.assert_close(world.get_pose_e(name), expected, atol=2e-5, rtol=0)
            before = {name: world.get_pose_e(name) for name in cache.poses}
            env_ids = torch.tensor([1], device=device)
            for name in cache.poses:
                displaced = scene[name].data.root_pose_w.torch[env_ids].clone()
                displaced[:, 2] += 1.0
                scene[name].write_root_pose_to_sim(displaced, env_ids=env_ids)
                scene[name].write_root_velocity_to_sim(torch.ones((1, 6), device=device), env_ids=env_ids)
            env.unwrapped._reset_idx(env_ids)
            for name, poses in cache.poses.items():
                actual = world.get_pose_e(name)
                torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
                torch.testing.assert_close(actual[1], poses[0].to_tensor(device), atol=2e-5, rtol=0)
                torch.testing.assert_close(
                    scene[name].data.root_vel_w.torch[env_ids],
                    torch.zeros((1, 6), device=device),
                    atol=0,
                    rtol=0,
                )
            before = {name: world.get_pose_e(name) for name in cache.poses}
            for _ in range(200):
                scene.write_data_to_sim()
                env.unwrapped.sim.step(render=False)
                scene.update(env.unwrapped.sim.get_physics_dt())
            for name in cache.poses:
                assert float((world.get_pose_e(name)[:, :3] - before[name][:, :3]).norm(dim=-1).max()) < 0.02
        finally:
            env.close()
    return True


def test_companion_cache_round_trip(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_companion_cache_round_trip, tmp_path=tmp_path)


def _make_cached_env():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    arena_env.placement_layouts = PlacementLayouts({f"cube_{i}": [Pose((0, 0, 1))] for i in range(4)})
    return arena_env


def _test_cache_rejects_conflicting_configuration(simulation_app, conflict, expected_error):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.relations import RandomAroundSolution, RotateAroundSolution
    from isaaclab_arena.utils.pose import PoseRange
    from isaaclab_arena.utils.velocity import Velocity

    arena_env = _make_cached_env()
    cfg = ArenaEnvBuilderCfg()
    cube = arena_env.scene.assets["cube_0"]
    if conflict == "random-reset":
        cube.add_relation(RandomAroundSolution())
    elif conflict == "pose-range":
        cube.set_initial_pose(PoseRange(position_xyz_max=(1.0, 1.0, 1.0)))
    elif conflict == "disabled-reset":
        cube.disable_reset_pose()
    elif conflict == "initial-velocity":
        cube.set_initial_velocity(Velocity(linear_xyz=(1.0, 0.0, 0.0)))
    elif conflict == "missing-robot":
        arena_env.embodiment.add_relation(RotateAroundSolution(yaw_rad=0.5))
    elif conflict == "placement-seed":
        cfg.placement_seed = 42
    elif conflict == "fixed-layout":
        cfg.resolve_on_reset = False
    elif conflict == "fixed-layout-default":
        arena_env.placer_params.resolve_on_reset = False
    with pytest.raises(AssertionError, match=expected_error):
        ArenaEnvBuilder(arena_env, cfg).compose_manager_cfg()
    return True


@pytest.mark.parametrize(
    "conflict, expected_error",
    [
        ("random-reset", "cannot randomize"),
        ("pose-range", "non-fixed pose-reset"),
        ("disabled-reset", "pose resets disabled"),
        ("initial-velocity", "nonzero initial velocity"),
        ("missing-robot", "missing placed objects.*robot"),
        ("placement-seed", "placement_seed applies to solving"),
        ("fixed-layout", "requires resolve_on_reset=True"),
        ("fixed-layout-default", "requires resolve_on_reset=True"),
    ],
)
def test_cache_rejects_conflicting_configuration(conflict, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_cache_rejects_conflicting_configuration, conflict=conflict, expected_error=expected_error
    )


def _test_python_integer_layouts_reset_objects_and_robot(simulation_app):
    import torch

    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_cached_env()
    poses = {f"cube_{i}": [Pose((i, 0, 2), (0, 0, 0, 1)), Pose((i, 1, 2), (0, 0, 0, 1))] for i in range(4)}
    poses["robot"] = [Pose((-1, 0, 0), (0, 0, 0, 1)), Pose((-1, 1, 0), (0, 0, 0, 1))]
    arena_env.placement_layouts = PlacementLayouts(poses)
    arena_env.embodiment.set_initial_pose(arena_env.embodiment.get_initial_pose())
    assert arena_env.embodiment.has_pose_reset_event()
    arena_env.placer_params.resolve_on_reset = False
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, resolve_on_reset=True)).make_registered()
    try:
        assert not arena_env.embodiment.has_pose_reset_event()
        # Equal counts consume and wrap the whole queue on every full reset.
        for _ in range(3):
            for name in poses:
                body = env.unwrapped.scene[name]
                displaced = body.data.root_pose_w.torch.clone()
                displaced[:, 0] += 0.25
                displaced[:, 3:] = torch.tensor([0, 0, 1, 0], device=env.unwrapped.device)
                body.write_root_pose_to_sim(displaced)
                body.write_root_velocity_to_sim(torch.ones_like(body.data.root_vel_w.torch))
            env.reset()
            for name, layouts in poses.items():
                expected = torch.tensor(
                    [layouts[i % 2].position_xyz + layouts[i % 2].rotation_xyzw for i in range(2)],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )
                torch.testing.assert_close(env.unwrapped.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)
                velocity = env.unwrapped.scene[name].data.root_vel_w.torch
                torch.testing.assert_close(velocity, torch.zeros_like(velocity), atol=0, rtol=0)
    finally:
        env.close()
    return True


def test_python_integer_layouts_reset_objects_and_robot():
    assert run_function_with_persistent_simulation_app(_test_python_integer_layouts_reset_objects_and_robot)
