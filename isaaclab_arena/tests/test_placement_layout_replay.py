# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Relation-placement replay through episode conditions."""

from pathlib import Path

import pytest

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parent / "test_data/placement_replay.yaml"
LAYOUTS = SOURCE.with_suffix(".jsonl")
CLI_SOURCE = Path(__file__).parent / "test_data/pick_and_place_maple_table_env_graph.yaml"
OBJECT_SET_SOURCE = Path(__file__).parent / "test_data/object_set_maple_table_env_graph.yaml"


def _test_condition_replay_applies_complete_layouts(simulation_app, resolve_on_reset):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_layouts import PlacementLayouts

    layouts = PlacementLayouts.from_episode_jsonl(LAYOUTS)
    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    assert arena_env.placement_asset_identities["cube_0"] == "dex_cube"
    arena_env.placer_params.resolve_on_reset = resolve_on_reset
    builder = ArenaEnvBuilder(
        arena_env,
        ArenaEnvBuilderCfg(num_envs=3, episode_conditions_path=str(LAYOUTS)),
        hydra_overrides=["cube_0.mass.enabled=true"],
    )
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    event_names = list(vars(env_cfg.events))
    assert event_names.index("cube_0_mass_variation") < event_names.index("scene_relation_placement")
    assert event_names[-1] == "scene_relation_placement"
    placement = builder._scene_variations[0]
    with patch.object(
        placement.placement_pool,
        "sample_for_envs",
        side_effect=AssertionError("Condition replay must not sample the live placement pool"),
    ):
        env = builder.make_registered(env_cfg, env_kwargs)
        try:
            base = env.unwrapped
            env.reset()
            scheduler = base.condition_scheduler
            assert scheduler is not None
            for env_id in range(base.num_envs):
                condition = scheduler.condition_for_env(env_id)
                layout_index = int(condition.condition_id.rsplit("_", 1)[1])
                assert (
                    base.variation_recorder["scene.relation_placement"].sample_for_episode(
                        env_id, base.get_episode_index(env_id)
                    )
                    == condition.runtime_variations["scene.relation_placement"]
                )
                for name, poses in layouts.poses.items():
                    torch.testing.assert_close(
                        base.arena_world.get_pose_e(name)[env_id],
                        poses[layout_index].to_tensor(base.device),
                        atol=2e-5,
                        rtol=0,
                    )

            before = {name: base.arena_world.get_pose_e(name) for name in layouts.poses}
            robot = base.scene.articulations["robot"]
            default_joint_pos = robot.data.default_joint_pos.torch.clone()
            moved_joint_pos = default_joint_pos + 0.1
            robot.write_joint_position_to_sim_index(position=moved_joint_pos)
            base._reset_idx(torch.tensor([1], device=base.device))
            condition = scheduler.condition_for_env(1)
            layout_index = int(condition.condition_id.rsplit("_", 1)[1])
            for name, poses in layouts.poses.items():
                actual = base.arena_world.get_pose_e(name)
                torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
                torch.testing.assert_close(
                    actual[1],
                    poses[layout_index].to_tensor(base.device),
                    atol=2e-5,
                    rtol=0,
                )
            torch.testing.assert_close(robot.data.joint_pos.torch[0], moved_joint_pos[0], atol=2e-5, rtol=0)
            assert not torch.allclose(robot.data.joint_pos.torch[1], moved_joint_pos[1], atol=2e-5, rtol=0)
        finally:
            env.close()
    return True


@pytest.mark.parametrize("resolve_on_reset", [True, False])
def test_condition_replay_applies_complete_layouts(resolve_on_reset):
    assert run_function_with_persistent_simulation_app(
        _test_condition_replay_applies_complete_layouts,
        resolve_on_reset=resolve_on_reset,
    )


def _test_static_relation_placement_records_fixed_samples(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    arena_env.placer_params.resolve_on_reset = False
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    placement = builder._scene_variations[0]
    with patch.object(
        placement.placement_pool,
        "sample_for_envs",
        side_effect=AssertionError("Fixed placement must not sample the pool during reset"),
    ):
        env = builder.make_registered(env_cfg, env_kwargs)
        try:
            base = env.unwrapped
            env.reset()
            before = {name: base.arena_world.get_pose_e(name) for name in ("cube_0", "cube_1", "cube_2", "cube_3")}
            previous_sample = base.variation_recorder["scene.relation_placement"].sample_for_episode(
                1, base.get_episode_index(1)
            )
            env_id = torch.tensor([1], device=base.device)
            for name in before:
                moved = base.scene[name].data.root_pose_w.torch[env_id].clone()
                moved[:, 2] += 1.0
                base.scene[name].write_root_pose_to_sim(moved, env_ids=env_id)
            base._reset_idx(env_id)
            next_sample = base.variation_recorder["scene.relation_placement"].sample_for_episode(
                1, base.get_episode_index(1)
            )
            assert next_sample == previous_sample
            for name, pose in before.items():
                torch.testing.assert_close(base.arena_world.get_pose_e(name), pose, atol=2e-5, rtol=0)
        finally:
            env.close()
    return True


def test_static_relation_placement_records_fixed_samples():
    assert run_function_with_persistent_simulation_app(_test_static_relation_placement_records_fixed_samples)


def _test_live_relation_placement_respects_partial_resets(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2))
    env_cfg, env_kwargs = builder.compose_manager_cfg()
    placement = builder._scene_variations[0]
    with patch.object(
        placement.placement_pool,
        "sample_for_envs",
        wraps=placement.placement_pool.sample_for_envs,
    ) as sample:
        env = builder.make_registered(env_cfg, env_kwargs)
        try:
            base = env.unwrapped
            env.reset()
            sample.reset_mock()
            before = base.arena_world.get_pose_e("cube_0")
            base._reset_idx(torch.tensor([1], device=base.device))

            sample.assert_called_once_with([1])
            torch.testing.assert_close(base.arena_world.get_pose_e("cube_0")[0], before[0], atol=2e-5, rtol=0)
            assert placement.last_results.keys() == {1}
            assert (
                base.variation_recorder["scene.relation_placement"].sample_for_episode(1, base.get_episode_index(1))
                is not None
            )
        finally:
            env.close()
    return True


def test_live_relation_placement_respects_partial_resets():
    assert run_function_with_persistent_simulation_app(_test_live_relation_placement_respects_partial_resets)


def test_graph_cli_swap_preserves_key_and_changes_concrete_identity():
    from argparse import Namespace

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec

    spec = ArenaEnvGraphSpec.from_yaml(CLI_SOURCE)
    override = next(item for item in spec.cli_override_specs if item.arg == "object")
    original_key = override.target_node_id
    original_identity = spec._asset_by_id(original_key).registry_name
    replacement = next(obj.registry_name for obj in spec.objects if obj.registry_name != original_identity)
    spec.apply_cli_override_args(Namespace(**{override.dest: replacement}))

    assert spec._asset_by_id(original_key).id == original_key
    assert spec._asset_by_id(original_key).registry_name == replacement


def _test_object_set_replay_is_rejected_before_environment_construction(simulation_app):
    from isaaclab_arena.assets.object_library import DexCube, DomeLight
    from isaaclab_arena.assets.object_set import RigidObjectSet
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene

    graph_environment = ArenaEnvGraphSpec.from_yaml(OBJECT_SET_SOURCE).to_arena_env()
    programmatic_environment = IsaacLabArenaEnvironment(
        name="programmatic_object_set_replay",
        scene=Scene(
            assets=[
                RigidObjectSet(name="cubes", objects=[DexCube()]),
                DomeLight(),
            ]
        ),
    )
    for arena_env in (graph_environment, programmatic_environment):
        builder = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(episode_conditions_path=str(LAYOUTS)),
        )
        with pytest.raises(AssertionError, match="does not support RigidObjectSet"):
            builder.compose_manager_cfg()
    return True


def test_object_set_replay_is_rejected_before_environment_construction():
    assert run_function_with_persistent_simulation_app(
        _test_object_set_replay_is_rejected_before_environment_construction
    )
