# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end recorded placement replay coverage."""

from pathlib import Path

from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app

SOURCE = Path(__file__).parent / "test_data/placement_replay.yaml"
LAYOUTS = SOURCE.with_suffix(".jsonl")


def _test_recorded_placement_replay_applies_complete_layouts(simulation_app):
    import torch
    from unittest.mock import patch

    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from isaaclab_arena.relations.placement_sampler import placement_samples_to_pose_columns, read_placement_samples
    from isaaclab_arena.relations.relation_solver import RelationSolver

    poses_by_root = placement_samples_to_pose_columns(read_placement_samples(LAYOUTS))
    arena_env = ArenaEnvGraphSpec.from_yaml(SOURCE).to_arena_env()
    with patch.object(RelationSolver, "solve", side_effect=AssertionError("Recorded replay must not solve")):
        env = ArenaEnvBuilder(
            arena_env,
            ArenaEnvBuilderCfg(
                num_envs=3,
                solve_relations=False,
                resolve_on_reset=False,
                recorded_variation_samples_path=str(LAYOUTS),
            ),
        ).make_registered()
    try:
        base = env.unwrapped
        env.reset()
        for name, poses in poses_by_root.items():
            expected = torch.stack([pose.to_tensor(base.device) for pose in poses[:3]])
            torch.testing.assert_close(base.arena_world.get_pose_e(name), expected, atol=2e-5, rtol=0)

        # The initial reset consumed source rows 0, 1, and 2. A partial reset
        # receives row 3 without disturbing surviving env slots.
        before = {name: base.arena_world.get_pose_e(name).clone() for name in poses_by_root}
        base.reset(env_ids=torch.tensor([1], device=base.device))
        for name, poses in poses_by_root.items():
            actual = base.arena_world.get_pose_e(name)
            torch.testing.assert_close(actual[[0, 2]], before[name][[0, 2]], atol=2e-5, rtol=0)
            torch.testing.assert_close(actual[1], poses[3].to_tensor(base.device), atol=2e-5, rtol=0)

        assert base.variation_recorder.placement_record is not None
        assert (
            base.variation_recorder.placement_record.sample_for_episode(1, base.get_episode_index(1))["layout_id"]
            == "layout_000003"
        )
    finally:
        env.close()
    return True


def test_recorded_placement_replay_applies_complete_layouts():
    assert run_function_with_persistent_simulation_app(_test_recorded_placement_replay_applies_complete_layouts)
