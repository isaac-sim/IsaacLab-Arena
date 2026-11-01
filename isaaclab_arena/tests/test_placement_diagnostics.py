# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the diagnostics that the validation runner, the placement pool and the environment builder record."""

from pathlib import Path

import pytest

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.pooled_object_placer import PooledObjectPlacer
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relations import IsAnchor, On
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.tests.utils.constants import TestConstants
from isaaclab_arena.tests.utils.persistent_simulation_app import run_function_with_persistent_simulation_app
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose

_GRAPH = Path(TestConstants.test_data_dir) / "pick_and_place_maple_table_env_graph.yaml"


def _test_a_failed_rebuild_clears_the_placement_diagnostics(simulation_app) -> bool:
    from isaaclab_arena.environment_spec.arena_env_graph_spec import ArenaEnvGraphSpec
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg

    arena_env = ArenaEnvGraphSpec.from_yaml(_GRAPH).to_arena_env()
    # The assertions only need one filled pool, not a converged layout.
    arena_env.placer_params = ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=20, verbose=False),
        max_placement_attempts=1,
        min_unique_layouts_per_env=1,
    )
    builder = ArenaEnvBuilder(arena_env, ArenaEnvBuilderCfg(num_envs=2, placement_seed=0))
    builder.build_env_cfg()
    diagnostics = builder.placement_diagnostics
    assert diagnostics is not None and diagnostics.batches and diagnostics.batches[0].checks

    def reject_env_cfg(env_cfg):
        raise RuntimeError("env_cfg rejected")

    # Fail after placement has filled the pool, so the failed build has diagnostics of its own.
    arena_env.env_cfg_callback = reject_env_cfg
    with pytest.raises(RuntimeError, match="env_cfg rejected"):
        builder.build_env_cfg()
    # Neither the earlier build's diagnostics nor the failed build's own may be reported.
    assert builder.placement_diagnostics is None
    return True


def test_a_failed_rebuild_clears_the_placement_diagnostics():
    """A build records its placement diagnostics, and a later build that fails after placement leaves none behind."""
    assert run_function_with_persistent_simulation_app(_test_a_failed_rebuild_clears_the_placement_diagnostics)


def test_pool_diagnostics_record_a_fixed_size_first_batch_and_its_checks():
    """The first batch holds pool_size * max_placement_attempts candidates."""
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=0.2), _placer_params(), pool_size=2 * 3, num_envs=2)

    [first_batch, *_] = pool.last_fill_diagnostics.batches
    assert first_batch.num_candidates == 2 * 3 * 2
    assert 0 < first_batch.num_valid <= first_batch.num_candidates
    on_relation = first_batch.checks["on_relation"]
    assert on_relation.required and on_relation.num_evaluated == first_batch.num_candidates
    assert not pool.last_fill_diagnostics.fallback_envs


def test_pool_diagnostics_describe_only_the_most_recent_fill():
    """A refill records its batches in new diagnostics and leaves the earlier fill's diagnostics unchanged."""
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=0.2), _placer_params(), pool_size=2 * 3, num_envs=2)
    first_fill = pool.last_fill_diagnostics
    num_first_fill_batches = len(first_fill.batches)

    # Taking every stored layout makes the next sample refill the pool.
    pool.sample_without_replacement(2 * 3)
    pool.sample_without_replacement(2)

    assert pool.last_fill_diagnostics is not first_fill and pool.last_fill_diagnostics.batches
    assert len(first_fill.batches) == num_first_fill_batches


def test_pool_diagnostics_record_fallbacks_and_explain_fill_failures():
    """A box wider than its desk is never valid: the pool falls back, or names the failing check when it can't."""
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=1.5), _placer_params(), pool_size=2, num_envs=2)

    diagnostics = pool.last_fill_diagnostics
    # Every batch rejects every candidate, so the pool solves all max_placement_attempts batches, then falls back.
    assert [batch.num_valid for batch in diagnostics.batches] == [0, 0]
    assert diagnostics.batches[0].checks["on_relation"].num_rejected == diagnostics.batches[0].num_candidates
    assert diagnostics.fallback_envs == [0, 1] and pool.had_fallbacks

    with pytest.raises(RuntimeError, match=r"batch 0: 0/4 valid; passed per check: .*on_relation=0/4"):
        PooledObjectPlacer(
            _desk_and_box(box_width_m=1.5),
            _placer_params(allow_best_loss_fallbacks=False),
            pool_size=2,
            num_envs=2,
        )


def _desk_and_box(box_width_m: float) -> list[DummyObject]:
    desk = DummyObject(name="desk", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (1.0, 1.0, 0.1)))
    desk.set_initial_pose(Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))
    desk.add_relation(IsAnchor())
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (box_width_m, box_width_m, 0.2)))
    box.add_relation(On(desk))
    return [desk, box]


def _placer_params(allow_best_loss_fallbacks: bool = True) -> ObjectPlacerParams:
    return ObjectPlacerParams(
        solver_params=RelationSolverParams(max_iters=20, verbose=False),
        max_placement_attempts=2,
        placement_seed=0,
        allow_best_loss_fallbacks=allow_best_loss_fallbacks,
    )
