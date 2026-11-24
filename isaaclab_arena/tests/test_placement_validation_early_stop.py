# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Tests for the rank-ordered, early-stopping expensive checks of the placement validation runner."""

from __future__ import annotations

from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
from isaaclab_arena.relations.optimization.placement_candidate_batch import PlacementCandidate, PlacementCandidateBatch
from isaaclab_arena.relations.validation.placement_validation_runner import PlacementValidationRunner
from isaaclab_arena.relations.validation.pre_physics import PrePhysicsPlacementValidator
from isaaclab_arena.tests.dummy_object import DummyObject
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


class _CheapAlwaysPasses(PrePhysicsPlacementValidator):
    check = "cheap"

    def _validate_batch(self, batch, collision_objects, rejections):
        return [True] * len(batch)


class _ExpensiveRecording(PrePhysicsPlacementValidator):
    """Expensive check that records the candidates it evaluates and fails the ones in ``infeasible``."""

    check = "expensive"
    run_after_inexpensive_checks = True

    def __init__(self, params, visualizer=None, infeasible=()):
        super().__init__(params, visualizer)
        self.evaluated_rounds: list[list[int]] = []
        self._infeasible = set(infeasible)

    def _validate_batch(self, batch, collision_objects, rejections):
        ids = [candidate.candidate_id for candidate in batch.candidates]
        self.evaluated_rounds.append(ids)
        return [candidate_id not in self._infeasible for candidate_id in ids]


def _batch(losses_per_env: dict[int, list[float]]) -> PlacementCandidateBatch:
    """One candidate per loss, numbered in the order given, with env ids as keyed."""
    box = DummyObject(name="box", bounding_box=AxisAlignedBoundingBox((0.0, 0.0, 0.0), (0.1, 0.1, 0.1)))
    candidates = []
    for env_id, losses in losses_per_env.items():
        for loss in losses:
            candidate = PlacementCandidate(
                env_id, len(candidates), {box: (0.0, 0.0, 0.0)}, {box: 0.0}, {box: box.get_bounding_box()}
            )
            candidate.loss = loss
            candidates.append(candidate)
    return PlacementCandidateBatch(candidates)


def _runner(expensive: _ExpensiveRecording) -> PlacementValidationRunner:
    params = ObjectPlacerParams(enabled_checks={"cheap", "expensive"})
    return PlacementValidationRunner(params, [_CheapAlwaysPasses(params), expensive])


def test_expensive_checks_stop_once_every_env_has_enough_valid_layouts():
    expensive = _ExpensiveRecording(ObjectPlacerParams())
    # env 0: candidates 0..3 with losses 0.4, 0.1, 0.3, 0.2 ; env 1: candidates 4..6 with losses 0.2, 0.1, 0.3
    batch = _batch({0: [0.4, 0.1, 0.3, 0.2], 1: [0.2, 0.1, 0.3]})

    _runner(expensive).validate_candidates(batch, [], required_valid_per_env=2)

    assert expensive.evaluated_rounds == [[1, 3, 5, 4]], "the two lowest-loss candidates of each env, in one round"
    verdicts = {
        candidate.candidate_id: candidate.validation.do_all_required_validation_checks_pass()
        for candidate in batch.candidates
    }
    assert verdicts == {0: False, 1: True, 2: False, 3: True, 4: True, 5: True, 6: False}, "unchecked candidates fail"


def test_expensive_checks_ask_for_more_candidates_when_a_round_falls_short():
    # Candidate 1 (env 0's best) is infeasible, so env 0 needs a second round; env 1 is satisfied after the first.
    expensive = _ExpensiveRecording(ObjectPlacerParams(), infeasible={1})
    batch = _batch({0: [0.4, 0.1, 0.3, 0.2], 1: [0.2, 0.1, 0.3]})

    _runner(expensive).validate_candidates(batch, [], required_valid_per_env=2)

    assert expensive.evaluated_rounds == [[1, 3, 5, 4], [2]], "env 0 fetches its next candidate by loss, env 1 is done"
    assert [c.candidate_id for c in batch.candidates if c.validation.do_all_required_validation_checks_pass()] == [
        2,
        3,
        4,
        5,
    ]


def test_expensive_checks_run_on_every_candidate_without_a_target():
    expensive = _ExpensiveRecording(ObjectPlacerParams())
    batch = _batch({0: [0.4, 0.1], 1: [0.2]})

    _runner(expensive).validate_candidates(batch, [])

    assert expensive.evaluated_rounds == [[0, 1, 2]]
