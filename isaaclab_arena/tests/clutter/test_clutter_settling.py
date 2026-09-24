# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


"""Offline clutter settling, acceptance and candidate isolation."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parent / "data/clutter_cubes.yaml"


def _assert_scene_state_equal(actual, expected):
    import torch

    for kind, states in expected.items():
        for name, state in states.items():
            for field, value in state.items():
                torch.testing.assert_close(actual[kind][name][field], value, atol=1e-6, rtol=0)


def _replace_cube_with_variants(arena_env):
    from isaaclab_arena.assets.object_library import DexCube
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.relations.relations import ClutterOn
    from isaaclab_arena.utils.pose import Pose

    variants = RigidObjectSet(
        name="cube_0",
        objects=[DexCube(scale=(0.6, 0.6, 0.6)), DexCube(scale=(2.0, 2.0, 2.0))],
        random_choice=True,
        initial_pose=Pose((0.0, 0.0, 1.0)),
    )
    variants.add_relation(ClutterOn(arena_env.scene.assets["office_table_background"]))
    arena_env.scene.assets["cube_0"] = variants
    return variants


def _test_settling_rejects_unfixed_variants_without_mutation(simulation_app, assigned_envs):
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_settling import settle_clutter

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    variants = _replace_cube_with_variants(arena_env)
    if assigned_envs is not None:
        variants.assign_variants(assigned_envs, variant_seed=42)
    assignments = variants.variant_indices_by_env
    spawn_paths = list(variants.object_cfg.spawn.usd_path)
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=4, solve_relations=False)).make_registered()
    try:
        env.reset()
        initial = env.unwrapped.scene.get_state()
        with pytest.raises(AssertionError, match="cube_0.*fixed variants for 4 environments"):
            settle_clutter(env, arena_env.get_placement_assets())
        assert variants.variant_indices_by_env is assignments
        assert variants.object_cfg.spawn.usd_path == spawn_paths
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
    finally:
        env.close()
    return True


@pytest.mark.parametrize("assigned_envs", [None, 2])
def test_settling_rejects_unfixed_variants_without_mutation(assigned_envs):
    assert run_function_with_persistent_simulation_app(
        _test_settling_rejects_unfixed_variants_without_mutation, assigned_envs=assigned_envs
    )


def _test_uncached_clutter_drops_at_simulation_start(simulation_app):
    import torch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.relations import RotateAroundSolution
    from isaaclab_arena.utils.pose import Pose

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    marker = RotateAroundSolution(roll_rad=0.4, pitch_rad=0.3, yaw_rad=0.7)
    arena_env.scene.assets["cube_0"].add_relation(marker)
    arena_env.placer_params.min_unique_layouts_per_env = 1
    arena_env.placer_params.max_placement_attempts = 1
    arena_env.placer_params.resolve_on_reset = False
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=1, placement_seed=42)).make_registered()
    try:
        env.reset()
        pose = env.unwrapped.arena_world.get_pose_e("cube_0")[0].cpu()
        rotation = Pose(rotation_xyzw=tuple(pose[3:].tolist())).to_transform_matrix("cpu")[:3, :3]
        base = Pose(rotation_xyzw=marker.get_rotation_xyzw()).to_transform_matrix("cpu")[:3, :3]
        # Random world-Z yaw must preserve the authored tilt.
        torch.testing.assert_close((rotation @ base.T)[2], torch.tensor([0.0, 0.0, 1.0]), atol=1e-6, rtol=0)
        for _ in range(200):
            env.unwrapped.scene.write_data_to_sim()
            env.unwrapped.sim.step(render=False)
            env.unwrapped.scene.update(env.unwrapped.sim.get_physics_dt())
        after = env.unwrapped.arena_world.get_pose_e("cube_0")[0, 2]
        assert float(pose[2] - after.cpu()) > 0.005
    finally:
        env.close()
    return True


def test_uncached_clutter_drops_at_simulation_start():
    assert run_function_with_persistent_simulation_app(_test_uncached_clutter_drops_at_simulation_start)


def _test_settling_restores_scene_and_retries_only_rejected_layouts(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.offline_placement.clutter_settling import _release_objects, settle_clutter
    from isaaclab_arena.offline_placement.clutter_validators import (
        RestValidator,
        SupportContainmentValidator,
        build_post_physics_validators,
        default_post_physics_validators,
    )
    from isaaclab_arena.utils.pose import Pose

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    variants = _replace_cube_with_variants(arena_env)
    variants.assign_variants(2, variant_seed=42)
    assignments = list(variants.variant_indices_by_env)
    assert set(assignments) == {0, 1}
    assets = arena_env.get_placement_assets()
    arena_env.scene.assets["office_table_background"].set_initial_pose(
        Pose((1.0, 0.0, 0.0), (0.0, 0.0, 2**-0.5, 2**-0.5))
    )
    env = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, solve_relations=False)).make_registered()
    checked = []

    class RejectSecondContainment(SupportContainmentValidator):
        def validate(self, data):
            checked.append(data.poses)
            return self.report("fell off: cube_0") if len(checked) == 2 else super().validate(data)

    containment = RejectSecondContainment()
    configurations = default_post_physics_validators()
    configurations["rest"]["move_thresh_m"] = 0.001
    validators = build_post_physics_validators(configurations)
    validators[1] = containment

    try:
        env.reset()
        initial = env.unwrapped.scene.get_state()
        spawned_bounds = env.unwrapped.arena_world.get_aabb_in_local_frame("cube_0")
        expected_sizes = torch.tensor([0.036, 0.12], device=spawned_bounds.size.device)[assignments]
        torch.testing.assert_close(spawned_bounds.size, expected_sizes[:, None].expand(-1, 3))

        class StrictRest(RestValidator):
            check = "strict_rest"

        with pytest.raises(AssertionError, match="At most one enabled RestValidator"):
            settle_clutter(env, assets, validators=[RestValidator(), StrictRest(move_thresh_m=0.0001)])
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        validators.append(StrictRest(enabled=False))

        with patch(
            "isaaclab_arena.offline_placement.clutter_settling._release_objects", wraps=_release_objects
        ) as release:
            results = settle_clutter(env, assets, attempts=3, validators=validators)
            layouts = [result.poses for result in results]
        # Only the rejected environment receives a second release.
        assert [call.args[1] for call in release.call_args_list] == [0, 1, 1]
        assert layouts == [checked[0], checked[2]]
        assert variants.variant_indices_by_env == assignments
        for result in results:
            assert result.validation["pre_physics"]["clutter_on_relation"]
            reports = {report["check"]: report for report in result.validation["post_physics"]}
            assert all(reports[name]["passed"] is True for name in configurations)
            assert reports["rest"]["configuration"]["move_thresh_m"] == 0.001
            assert reports["strict_rest"]["passed"] is None
            assert reports["strict_rest"]["reason"] == "disabled by configuration"
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        with patch.object(containment, "validate", return_value=containment.report("fell off: cube_0")):
            with pytest.raises(AssertionError, match="fell off: cube_0"):
                settle_clutter(env, assets, attempts=1, validators=validators)
        _assert_scene_state_equal(env.unwrapped.scene.get_state(), initial)
        world = env.unwrapped.arena_world
        by_scene_key = {asset.get_scene_key(): asset for asset in assets}
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
