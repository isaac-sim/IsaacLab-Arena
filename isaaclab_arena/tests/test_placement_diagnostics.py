# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the diagnostics that the validation runner, the placement pool and the environment builder record."""

from pathlib import Path

import pytest

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.placement_diagnostics import (
    AggregatedPlacementDiagnostics,
    BatchValidationDiagnostics,
    PlacementDiagnostics,
    Rejection,
    ValidationCheckDiagnostics,
)
from isaaclab_arena.relations.pooled_object_placer import PlacementPoolFillError, PooledObjectPlacer
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


def test_aggregated_diagnostics_sum_fills_by_batch_position():
    """Fills sum batch by batch and check by check; a later batch sums only the fills that validated one."""
    overlap = Rejection("'a' and 'b': bounding boxes intersect")
    first_batch = BatchValidationDiagnostics(
        100, 60, {"no_overlap": ValidationCheckDiagnostics(100, 70, required=True, rejections=[overlap] * 30)}
    )
    later_batch = BatchValidationDiagnostics(
        20,
        15,
        {
            "no_overlap": ValidationCheckDiagnostics(20, 16, required=True),
            "ik_reachable": ValidationCheckDiagnostics(16, 15, required=False),
        },
    )

    aggregated = AggregatedPlacementDiagnostics([
        PlacementDiagnostics(10, [first_batch, later_batch], fallback_envs=[3]),
        PlacementDiagnostics(10, [first_batch]),
        PlacementDiagnostics(5, [first_batch]),
    ])

    summed_first_batch = BatchValidationDiagnostics(
        300,
        180,
        {"no_overlap": ValidationCheckDiagnostics(300, 210, required=True, rejections=[overlap] * 90)},
    )
    assert (aggregated.num_builds, aggregated.num_envs, aggregated.num_fallback_envs, aggregated.num_batches) == (
        3,
        25,
        1,
        4,
    )
    assert aggregated.batches == [summed_first_batch, later_batch]
    assert summed_first_batch.checks["no_overlap"].get_num_rejected_by_cause() == {overlap.cause: 90}
    # The fills' own diagnostics are left unchanged.
    assert first_batch == BatchValidationDiagnostics(
        100, 60, {"no_overlap": ValidationCheckDiagnostics(100, 70, required=True, rejections=[overlap] * 30)}
    )


def test_aggregated_pool_diagnostics_file_round_trips(tmp_path):
    """A fill's diagnostics, with its batches, checks, rejections and fallbacks, load back from their file unchanged."""
    pool = PooledObjectPlacer(_desk_and_box(box_width_m=1.5), _placer_params(), pool_size=2, num_envs=2)
    aggregated = AggregatedPlacementDiagnostics([pool.last_fill_diagnostics])
    path = tmp_path / "placement_diagnostics.json"

    aggregated.save(path)

    assert AggregatedPlacementDiagnostics.load(path) == aggregated
    assert aggregated.num_envs == 2
    assert aggregated.num_fallback_envs > 0
    assert AggregatedPlacementDiagnostics.load(path).fills[0].batches[0].checks["on_relation"].rejections


def test_fill_failure_names_the_failing_relation_and_object():
    """A fill failure names the relation and object that rejected candidates, and carries the fill's diagnostics."""
    objects = _desk_and_box(box_width_m=1.5)
    pool = PooledObjectPlacer(objects, _placer_params(), pool_size=2, num_envs=2)

    num_rejected_by_cause = pool.last_fill_diagnostics.get_num_rejected_by_cause_per_check()["on_relation"]
    num_candidates = sum(batch.num_candidates for batch in pool.last_fill_diagnostics.batches)
    assert num_rejected_by_cause == {"'box' on(desk): edge margin wider than the free space": num_candidates}

    with pytest.raises(PlacementPoolFillError) as error:
        PooledObjectPlacer(
            _desk_and_box(box_width_m=1.5),
            _placer_params(allow_best_loss_fallbacks=False),
            pool_size=2,
            num_envs=2,
        )
    message = str(error.value)
    assert error.value.diagnostics.get_num_rejected_by_cause_per_check() == {
        "on_relation": {"'box' on(desk): edge margin wider than the free space": num_candidates}
    }
    assert "on_relation: 'box' on(desk): edge margin wider than the free space x" in message


def test_next_to_validator_records_the_relation_and_its_violation():
    """A NextTo rejection is recorded under the subject, reference and side, with the measured violation."""
    from isaaclab_arena.relations.relations import NextTo, Side
    from isaaclab_arena.relations.validation.pre_physics import NextToValidator
    from isaaclab_arena.tests.dummy_object import make_candidate_batch

    desk, box = _desk_and_box(box_width_m=0.2)
    plate = DummyObject(name="plate", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.1, 0.1, 0.02)))
    plate.add_relation(NextTo(box, side=Side.POSITIVE_X, distance_m=0.05, tolerance_m=0.01))
    bounds = {obj: obj.get_bounding_box() for obj in (desk, box, plate)}
    on_side = {desk: (0.0, 0.0, 0.0), box: (0.4, 0.4, 0.1), plate: (0.65, 0.4, 0.1)}
    wrong_side = {desk: (0.0, 0.0, 0.0), box: (0.4, 0.4, 0.1), plate: (0.1, 0.4, 0.1)}
    validator = NextToValidator(_placer_params())
    rejections: list[Rejection] = []

    verdicts = validator.validate_batch(
        make_candidate_batch([on_side, wrong_side], [{}, {}], [bounds, bounds]), [], rejections=rejections
    )

    assert verdicts == [True, False]
    assert rejections == [Rejection("'plate' next_to(box, positive_x): on the wrong side", detail="by 0.500 m")]


def test_on_validator_records_an_edge_margin_wider_than_the_free_space():
    """An On edge margin wider than the support leaves around the child is a rejection of its own, with both margins."""
    from isaaclab_arena.relations.validation.pre_physics import OnRelationValidator
    from isaaclab_arena.tests.dummy_object import make_candidate_batch

    desk, _ = _desk_and_box(box_width_m=0.9)
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.9, 0.9, 0.2)))
    box.add_relation(On(desk, edge_margin_m=0.1))
    bounds = {obj: obj.get_bounding_box() for obj in (desk, box)}
    layout = {desk: (0.0, 0.0, 0.0), box: (0.05, 0.05, 0.1)}
    validator = OnRelationValidator(_placer_params())
    rejections: list[Rejection] = []

    verdicts = validator.validate_batch(make_candidate_batch([layout], [{}], [bounds]), [], rejections=rejections)

    assert verdicts == [False]
    assert rejections == [
        Rejection(
            "'box' on(desk): edge margin wider than the free space",
            detail="edge_margin_m 0.100 m, feasible margin 0.050 m",
        )
    ]


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
