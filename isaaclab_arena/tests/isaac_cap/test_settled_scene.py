# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Offline ClutterOn settling and companion-cache replay."""

from argparse import Namespace
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parents[3] / "isaaclab_arena_environments/isaac_cap/clutter/clutter_scene.yaml"


def _arguments(output):
    return Namespace(
        env_spec=SOURCE,
        output=output,
        num_envs=2,
        num_layouts=None,
        seed=42,
        device="cuda:0",
        presets=None,
        register=[],
        attempts=3,
        timeout_s=10.0,
        poll_interval_s=0.4,
        move_thresh_m=0.002,
        turn_thresh_deg=2.0,
        required_quiet_windows=2,
        fall_through_tolerance_m=0.01,
        containment_margin_m=0.0,
        passive_move_thresh_m=0.002,
        passive_turn_thresh_deg=2.0,
    )


def _assert_scene_state_equal(actual, expected):
    import torch

    for kind, states in expected.items():
        for name, state in states.items():
            for field, value in state.items():
                torch.testing.assert_close(actual[kind][name][field], value, atol=1e-6, rtol=0)


def _test_companion_cache_round_trip(simulation_app, tmp_path):
    import torch
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.placement_events import write_scene_poses_to_sim
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.relations.relation_solver import RelationSolver
    from isaaclab_arena_environments.isaac_cap.clutter.generate_clutter_scene import generate_scene
    from isaaclab_arena_environments.isaac_cap.clutter.settle import settle_clutter

    data = yaml.safe_load(SOURCE.read_text())
    data["placement_validators"] = {
        "enabled_checks": ["no_overlap", "on_relation"],
        "required_checks": ["no_overlap", "on_relation"],
    }
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    original = source.read_bytes()
    path = tmp_path / "poses.yaml"
    args = _arguments(path)
    args.env_spec = source
    args.num_layouts = 4
    with patch("isaaclab_arena_environments.isaac_cap.clutter.settle.settle_clutter", wraps=settle_clutter) as settle:
        assert generate_scene(args) == path
    # Each batch reserves seeds for 2 environments, 10 candidates and 3 trials.
    assert [call.kwargs["seed"] for call in settle.call_args_list] == [42, 102]
    assert source.read_bytes() == original
    cache = PlacementLayouts.from_yaml(path)
    assert cache.num_layouts == 4
    assert cache.poses["cube_0"][0] != cache.poses["cube_0"][1]
    spec = ArenaEnvGraphSpec.from_yaml(source)
    with (
        patch.object(RelationSolver, "solve", side_effect=AssertionError("Cached replay must not solve")),
        patch.object(
            ObjectPlacer, "_validate_candidates", side_effect=AssertionError("Cached replay must not revalidate")
        ),
    ):
        arena_env = spec.to_arena_env(placement_layouts=path)
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
                        [poses[(iteration + i) % cache.num_layouts].to_tensor(device) for i in range(3)]
                    )
                    torch.testing.assert_close(world.get_pose_e(name), expected, atol=2e-5, rtol=0)
            before = {name: world.get_pose_e(name) for name in cache.poses}
            env_ids = torch.tensor([1], device=device)
            selected = {name: pose.to_tensor(device).unsqueeze(0) for name, pose in cache.get_layout(1).items()}
            for name in selected:
                scene[name].write_root_velocity_to_sim(torch.ones((1, 6), device=device), env_ids=env_ids)
            write_scene_poses_to_sim(env.unwrapped, env_ids, selected)
            for name, expected in selected.items():
                actual = world.get_pose_e(name)
                torch.testing.assert_close(actual[env_ids], expected, atol=2e-5, rtol=0)
                torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
                torch.testing.assert_close(
                    scene[name].data.root_vel_w.torch[env_ids],
                    torch.zeros((1, 6), device=device),
                    atol=0,
                    rtol=0,
                )
            env.unwrapped._reset_idx(env_ids)
            for name, poses in cache.poses.items():
                actual = world.get_pose_e(name)
                torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
                torch.testing.assert_close(actual[1], poses[1].to_tensor(device), atol=2e-5, rtol=0)
            before = {name: world.get_pose_e(name) for name in cache.poses}
            for _ in range(200):
                scene.write_data_to_sim()
                env.unwrapped.sim.step(render=False)
                scene.update(env.unwrapped.sim.get_physics_dt())
            for name in cache.poses:
                assert float((world.get_pose_e(name)[:, :3] - before[name][:, :3]).norm(dim=-1).max()) < 0.02
        finally:
            env.close()
    assert source.read_bytes() == original
    return True


def test_companion_cache_round_trip(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_companion_cache_round_trip, tmp_path=tmp_path)


def _test_uncached_clutter_drops_at_simulation_start(simulation_app):
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    arena_env.placer_params.min_unique_layouts_per_env = 1
    arena_env.placer_params.max_placement_attempts = 1
    arena_env.placer_params.resolve_on_reset = False
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, placement_seed=42)).make_registered()
    try:
        env.reset()
        before = env.unwrapped.arena_world.get_pose_e("cube_0")[:, 2].clone()
        for _ in range(200):
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
        after = env.unwrapped.arena_world.get_pose_e("cube_0")[:, 2]
        assert float((before - after).min()) > 0.005
    finally:
        env.close()
    return True


def test_uncached_clutter_drops_at_simulation_start():
    assert run_function_with_persistent_simulation_app(_test_uncached_clutter_drops_at_simulation_start)


def _test_settling_restores_scene_and_retries_only_rejected_layouts(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena_environments.isaac_cap.clutter.settle import _release_objects, settle_clutter
    from isaaclab_arena_environments.isaac_cap.clutter.validation import ClutterRestVerdict, check_resting_poses

    arena_env, assets = build_arena_env_with_assets_from_graph_spec(ArenaEnvGraphSpec.from_yaml(SOURCE))
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    checked = []

    def reject_second_once(positions, *args):
        checked.append(positions.clone())
        return ClutterRestVerdict(fell_off=[0]) if len(checked) == 2 else check_resting_poses(positions, *args)

    try:
        env.reset()
        initial = env.unwrapped.scene.get_state()
        with (
            patch(
                "isaaclab_arena_environments.isaac_cap.clutter.settle._release_objects", wraps=_release_objects
            ) as release,
            patch(
                "isaaclab_arena_environments.isaac_cap.clutter.settle.check_resting_poses",
                side_effect=reject_second_once,
            ),
        ):
            layouts = settle_clutter(env, list(assets.values()), attempts=3)
        assert [call.args[1] for call in release.call_args_list] == [0, 1, 1]
        for i, check_index in ((0, 0), (1, 2)):
            actual = torch.tensor([layouts[i][f"cube_{j}"].position_xyz for j in range(4)])
            torch.testing.assert_close(actual, checked[check_index], atol=0, rtol=0)
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        with patch(
            "isaaclab_arena_environments.isaac_cap.clutter.settle.check_resting_poses",
            return_value=ClutterRestVerdict(fell_off=[0]),
        ):
            with pytest.raises(AssertionError, match="fell off: cube_0"):
                settle_clutter(env, list(assets.values()), attempts=1)
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
    finally:
        env.close()
    return True


def test_settling_restores_scene_and_retries_only_rejected_layouts():
    assert run_function_with_persistent_simulation_app(_test_settling_restores_scene_and_retries_only_rejected_layouts)


def _test_displaced_passive_neighbor_prevents_cache_output(simulation_app):
    import tempfile
    import yaml

    from isaaclab_arena_environments.isaac_cap.clutter.generate_clutter_scene import generate_scene

    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "source.yaml"
        data = yaml.safe_load(SOURCE.read_text())
        data["relations"] = [relation for relation in data["relations"] if relation["subject"] != "cube_3"]
        data["objects"][3]["params"] = {"initial_pose": {"position_xyz": [0.5, 0.0, 3.0]}}
        source.write_text(yaml.safe_dump(data))
        output = Path(directory) / "poses.yaml"
        args = _arguments(output)
        args.env_spec, args.num_envs, args.attempts = source, 1, 1
        with pytest.raises(AssertionError, match="cube_3: passive drift"):
            generate_scene(args)
        assert not output.exists()
    return True


def test_displaced_passive_neighbor_prevents_cache_output():
    assert run_function_with_persistent_simulation_app(_test_displaced_passive_neighbor_prevents_cache_output)


def _test_release_failures_retry_without_releasing_invalid_layouts(simulation_app):
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_conversion_utils import (
        build_arena_env_with_assets_from_graph_spec,
    )
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_validation import PlacementCheck
    from isaaclab_arena_environments.isaac_cap.clutter.settle import _release_objects, settle_clutter

    arena_env, assets = build_arena_env_with_assets_from_graph_spec(ArenaEnvGraphSpec.from_yaml(SOURCE))
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    params = ObjectPlacerParams(max_placement_attempts=2)
    validate = ObjectPlacer._validate_candidates
    try:
        env.reset()
        initial = env.unwrapped.scene.get_state()
        for exhaust in (False, True):
            rejected_envs = iter(({0, 1}, {0, 1}, {0, 1}) if exhaust else ({0, 1}, {1}, {0}))
            seeds = []

            def reject_candidates(placer, *args):
                seeds.append(placer.params.placement_seed)
                results = validate(placer, *args)
                for env_id in next(rejected_envs):
                    first = env_id * params.max_placement_attempts
                    for result in results[first : first + params.max_placement_attempts]:
                        result.validation_results[PlacementCheck.ON_RELATION] = False
                return results

            with (
                patch.object(ObjectPlacer, "_validate_candidates", reject_candidates),
                patch(
                    "isaaclab_arena_environments.isaac_cap.clutter.settle._release_objects", wraps=_release_objects
                ) as release,
            ):
                if exhaust:
                    with pytest.raises(AssertionError, match="No settled layout after 3.*release placement failed"):
                        settle_clutter(env, list(assets.values()), attempts=3, placer_params=params)
                    release.assert_not_called()
                else:
                    layouts = settle_clutter(env, list(assets.values()), attempts=3, placer_params=params)
                    assert len(layouts) == 2
                    assert [call.args[1] for call in release.call_args_list] == [0, 1]
                    assert all(call.args[2].success for call in release.call_args_list)
            assert seeds == [42, 46, 50]
            _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        with patch.object(ObjectPlacer, "_validate_candidates", side_effect=AssertionError("invalid configuration")):
            with pytest.raises(AssertionError, match="invalid configuration"):
                settle_clutter(env, list(assets.values()), attempts=3, placer_params=params)
    finally:
        env.close()
    return True


def test_release_failures_retry_without_releasing_invalid_layouts():
    assert run_function_with_persistent_simulation_app(_test_release_failures_retry_without_releasing_invalid_layouts)


def _make_cached_env():
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    arena_env.placement_layouts = PlacementLayouts({f"cube_{i}": [Pose((0, 0, 1))] for i in range(4)})
    return arena_env


def _assert_cache_rejected(arena_env, expected_error):
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    with pytest.raises(AssertionError, match=expected_error):
        ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg()).compose_manager_cfg()


def _test_python_cache_rejects_invalid_object_names(simulation_app, cached_names, expected_error):
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_cached_env()
    arena_env.placement_layouts = PlacementLayouts({name: [Pose()] for name in cached_names})
    _assert_cache_rejected(arena_env, expected_error)
    return True


@pytest.mark.parametrize(
    "cached_names, expected_error",
    [
        pytest.param(["cube_0", "cube_1", "cube_2"], "missing placed", id="missing-object"),
        pytest.param(["cube_0", "cube_1", "cube_2", "cube_3", "unknown"], "Unknown cached", id="unknown-object"),
    ],
)
def test_python_cache_rejects_invalid_object_names(cached_names, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_python_cache_rejects_invalid_object_names, cached_names=cached_names, expected_error=expected_error
    )


def _test_python_cache_rejects_conflicting_resets(simulation_app, randomize, expected_error):
    from isaaclab_arena.relations.relations import RandomAroundSolution
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_cached_env()
    cube = arena_env.scene.assets["cube_0"]
    if randomize:
        cube.add_relation(RandomAroundSolution())
    else:
        cube.set_initial_pose(Pose())
    _assert_cache_rejected(arena_env, expected_error)
    return True


@pytest.mark.parametrize(
    "randomize, expected_error",
    [
        pytest.param(True, "cannot randomize", id="random-reset"),
        pytest.param(False, "explicit pose-reset", id="fixed-reset"),
    ],
)
def test_python_cache_rejects_conflicting_resets(randomize, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_python_cache_rejects_conflicting_resets, randomize=randomize, expected_error=expected_error
    )


def _test_python_cache_revalidates_mutated_poses(simulation_app, position, count, expected_error):
    from isaaclab_arena.utils.pose import Pose

    arena_env = _make_cached_env()
    arena_env.placement_layouts.poses["cube_0"] = [Pose(position) for _ in range(count)]
    _assert_cache_rejected(arena_env, expected_error)
    return True


@pytest.mark.parametrize(
    "position, count, expected_error",
    [
        pytest.param((float("nan"), 0, 0), 1, "finite numbers", id="nonfinite-pose"),
        pytest.param((0, 0, 1), 2, "same nonzero", id="unequal-counts"),
    ],
)
def test_python_cache_revalidates_mutated_poses(position, count, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_python_cache_revalidates_mutated_poses, position=position, count=count, expected_error=expected_error
    )


def _test_python_cache_rejects_duplicate_scene_keys(simulation_app):
    arena_env = _make_cached_env()
    arena_env.scene.assets["cube_1"].name = "cube_0"
    _assert_cache_rejected(arena_env, "distinct scene keys")
    return True


def test_python_cache_rejects_duplicate_scene_keys():
    assert run_function_with_persistent_simulation_app(_test_python_cache_rejects_duplicate_scene_keys)


def _test_generation_honors_requested_validators(simulation_app, tmp_path, check, required, available, expected_error):
    import yaml
    from contextlib import nullcontext
    from unittest.mock import patch

    from isaaclab_arena.relations.placement_validator_registry import PlacementValidatorRegistry
    from isaaclab_arena.relations.placement_validators import PlacementValidator
    from isaaclab_arena_environments.isaac_cap.clutter.generate_clutter_scene import generate_scene
    from isaaclab_arena_environments.isaac_cap.clutter.settle import _release_objects

    validated_batches = []

    class RejectRelease(PlacementValidator):
        check = "reject_release"

        def validate_batch(self, positions, orientations, bboxes, collision_objects):
            validated_batches.append(len(positions))
            return [False] * len(positions)

    data = yaml.safe_load(SOURCE.read_text())
    data["placement_validators"] = {
        "enabled_checks": [check],
        "required_checks": [check] if required else [],
    }
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    output = tmp_path / "poses.yaml"
    args = _arguments(output)
    args.env_spec = source
    args.num_envs = 1
    args.attempts = 1
    registry = PlacementValidatorRegistry()
    with (
        patch.dict(registry._components, reject_release=RejectRelease),
        patch.object(RejectRelease, "is_available", return_value=available),
        patch(
            "isaaclab_arena_environments.isaac_cap.clutter.settle._release_objects", wraps=_release_objects
        ) as release,
    ):
        with pytest.raises(AssertionError, match=expected_error) if expected_error else nullcontext():
            assert generate_scene(args) == output
        if expected_error:
            release.assert_not_called()
            assert not output.exists()
        else:
            release.assert_called_once()
            assert output.exists()
    assert bool(validated_batches) == (check == RejectRelease.check and available)
    return True


@pytest.mark.parametrize(
    "check, required, available, expected_error",
    [
        pytest.param("reject_release", True, True, "release placement failed", id="required"),
        pytest.param("reject_release", False, True, None, id="optional"),
        pytest.param("missing_validator", True, True, "validators did not run", id="unknown"),
        pytest.param("reject_release", True, False, "validators did not run", id="unavailable"),
    ],
)
def test_generation_honors_requested_validators(tmp_path, check, required, available, expected_error):
    assert run_function_with_persistent_simulation_app(
        _test_generation_honors_requested_validators,
        tmp_path=tmp_path,
        check=check,
        required=required,
        available=available,
        expected_error=expected_error,
    )
