# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Offline ClutterOn settling and pose-file generation."""

from argparse import Namespace
from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parents[3] / "isaaclab_arena_examples/relations/clutter/clutter_scene.yaml"


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


def _test_generation_writes_complete_layouts(simulation_app, tmp_path):
    import yaml
    from unittest.mock import patch

    from isaaclab_arena.relations.clutter.settle import settle_clutter
    from isaaclab_arena.scripts.generate_clutter_scene import generate_scene
    from isaaclab_arena.utils.pose import Pose

    data = yaml.safe_load(SOURCE.read_text())
    data["placement_validators"] = {
        "enabled_checks": ["no_overlap", "on_relation"],
        "required_checks": ["no_overlap", "on_relation"],
    }
    data["external_yaml"] = "physics.yaml"
    physics = tmp_path / "physics.yaml"
    physics.write_text(
        yaml.safe_dump({
            "default_physics_backend": "physx",
            "env_cfg_override": {"sim": {"dt": 0.01}},
        })
    )
    source = tmp_path / "scene.yaml"
    source.write_text(yaml.safe_dump(data))
    original = source.read_bytes()
    path = tmp_path / "poses.yaml"
    args = _arguments(path)
    args.env_spec = source
    args.num_layouts = 4

    def settle_with_graph_settings(env, *args, **kwargs):
        assert env.unwrapped.sim.get_physics_dt() == pytest.approx(0.01)
        return settle_clutter(env, *args, **kwargs)

    with patch("isaaclab_arena.relations.clutter.settle.settle_clutter", wraps=settle_with_graph_settings) as settle:
        assert generate_scene(args) == path
    # Each batch reserves seeds for 2 environments, 10 candidates and 3 trials.
    assert [call.kwargs["seed"] for call in settle.call_args_list] == [42, 102]
    assert source.read_bytes() == original
    poses = yaml.safe_load(path.read_text())
    assert set(poses) == {f"cube_{i}" for i in range(4)}
    assert all(len(values) == 4 for values in poses.values())
    assert poses["cube_0"][0] != poses["cube_0"][1]
    for values in poses.values():
        for value in values:
            Pose.from_dict(value)
    with pytest.raises(AssertionError, match="Output already exists"):
        generate_scene(args)
    assert yaml.safe_load(path.read_text()) == poses
    return True


def test_generation_writes_complete_layouts(tmp_path):
    assert run_function_with_persistent_simulation_app(_test_generation_writes_complete_layouts, tmp_path=tmp_path)


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
    from isaaclab_arena.relations.clutter.settle import _release_objects, settle_clutter
    from isaaclab_arena.relations.clutter.validation import ClutterRestVerdict, check_resting_poses

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
            patch("isaaclab_arena.relations.clutter.settle._release_objects", wraps=_release_objects) as release,
            patch(
                "isaaclab_arena.relations.clutter.settle.check_resting_poses",
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
            "isaaclab_arena.relations.clutter.settle.check_resting_poses",
            return_value=ClutterRestVerdict(fell_off=[0]),
        ):
            with pytest.raises(AssertionError, match="fell off: cube_0"):
                settle_clutter(env, list(assets.values()), attempts=1)
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        world = env.unwrapped.arena_world
        by_scene_key = {asset.get_scene_key(): asset for asset in assets.values()}
        for env_id, layout in enumerate(layouts):
            for name, pose in layout.items():
                by_scene_key[name].write_layout_pose_to_sim(env.unwrapped, env_id, pose)
        before = {name: world.get_pose_e(name).clone() for name in layouts[0]}
        for _ in range(200):
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
        for name, pose in before.items():
            torch.testing.assert_close(world.get_pose_e(name), pose, atol=0.005, rtol=0)
    finally:
        env.close()
    return True


def test_settling_restores_scene_and_retries_only_rejected_layouts():
    assert run_function_with_persistent_simulation_app(_test_settling_restores_scene_and_retries_only_rejected_layouts)


def _test_displaced_passive_neighbor_prevents_cache_output(simulation_app):
    import tempfile
    import yaml

    from isaaclab_arena.scripts.generate_clutter_scene import generate_scene

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


def _test_generation_honors_requested_validators(simulation_app, tmp_path, check, required, available, expected_error):
    import yaml
    from contextlib import nullcontext
    from unittest.mock import patch

    from isaaclab_arena.relations.clutter.settle import _release_objects
    from isaaclab_arena.relations.placement_validator_registry import PlacementValidatorRegistry
    from isaaclab_arena.relations.placement_validators import PlacementValidator
    from isaaclab_arena.scripts.generate_clutter_scene import generate_scene

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
        patch("isaaclab_arena.relations.clutter.settle._release_objects", wraps=_release_objects) as release,
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
