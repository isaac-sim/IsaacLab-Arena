# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Solve movable clutter supports before their release layouts."""

from __future__ import annotations

import hashlib
from copy import copy
from itertools import chain
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import (
    build_per_env_bounding_boxes,
    has_heterogeneous_objects,
    update_candidate_bounds,
)
from isaaclab_arena.relations.placement_candidate_batch import PlacementCandidate, PlacementCandidateBatch
from isaaclab_arena.relations.placement_events import get_pose_from_layout, get_rotation_xyzw
from isaaclab_arena.relations.placement_result import PlacementResult
from isaaclab_arena.relations.placement_validation_runner import PlacementValidationRunner
from isaaclab_arena.relations.relations import ClutterOn, FaceTo, IsAnchor, RandomAroundSolution, Relation, get_relation
from isaaclab_arena.relations.validation.types import PlacementValidationResults
from isaaclab_arena.utils.bounding_box import quaternion_to_90_deg_z_quarters

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.object_placer import ObjectPlacer
    from isaaclab_arena.relations.object_placer_params import ObjectPlacerParams
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


def place_staged_clutter(
    placer: ObjectPlacer,
    objects: list[PlaceableAsset],
    num_envs: int,
    results_per_env: int,
    collision_objects: list[CollisionObject],
) -> list[list[PlacementResult]]:
    """Solve movable clutter supports first; otherwise retain the joint solve.

    Args:
        placer: Owner of the reusable solver, validators and placement settings.
        objects: Placement assets, including any ClutterOn supports as non-clutter fixtures.
        num_envs: Number of independent environment queues.
        results_per_env: Number of complete candidates to return for each queue.
        collision_objects: Fixed obstacles to avoid in both passes.

    Returns:
        Complete results keyed by the original assets, ranked by failed checks then combined loss.
        When two passes are needed, both contribute losses, attempts and required-check failures.
        Tie-breaking favors variation across retained fixture layouts.
    """
    clutter = [obj for obj in objects if get_relation(obj, ClutterOn) is not None]
    if not clutter or all(get_relation(obj, ClutterOn).parent.is_anchor for obj in clutter):
        return placer._place_ranked_per_env(objects, num_envs, results_per_env, collision_objects)

    assert not has_heterogeneous_objects(objects), "Resolve object sets before staged clutter placement"
    fixtures = [obj for obj in objects if obj not in clutter]
    _validate_fixture_dependencies(fixtures, clutter, placer.params)
    stage_validation = PlacementValidationRunner(
        placer.params,
        [validator for validator in placer._validators if not validator.run_after_inexpensive_checks],
        placer._visualizer,
    )
    fixture_results = placer._place_ranked_per_env(
        fixtures, num_envs, results_per_env, collision_objects, validation=stage_validation
    )
    fixture_groups = _place_clutter_for_fixtures(
        placer, objects, clutter, fixture_results, collision_objects, stage_validation
    )
    results = [list(chain.from_iterable(groups)) for groups in fixture_groups]
    _validate_complete_layouts(placer, objects, results, collision_objects)
    for env_id, groups in enumerate(fixture_groups):
        for layouts in groups:
            layouts.sort(key=_layout_rank)
        # Each fixture has the same restart budget. Interleave after all checks so tied
        # restarts from one fixture do not crowd out other equally ranked fixture poses.
        interleaved = chain.from_iterable(zip(*groups, strict=True))
        results[env_id] = sorted(interleaved, key=_layout_rank)[:results_per_env]
    return results


def _validate_fixture_dependencies(
    fixtures: list[PlaceableAsset], clutter: list[PlaceableAsset], params: ObjectPlacerParams
) -> None:
    """Require independent fixtures and orientations representable as fixed solver geometry."""
    fixture_set = set(fixtures)
    for obj in fixtures:
        for relation in obj.get_relations():
            if isinstance(relation, (Relation, FaceTo)):
                assert (
                    relation.parent in fixture_set
                ), f"Fixture '{obj.name}' must reference another participating fixture, not clutter"
        if obj.is_anchor:
            continue
        assert (
            not params.random_yaw_init and get_relation(obj, FaceTo) is None
        ), "Staged fixtures require fixed quarter-turn rotations; disable random_yaw_init and FaceTo"
        assert get_relation(obj, RandomAroundSolution) is None, "Staged fixtures cannot randomize after solving"
        quaternion_to_90_deg_z_quarters(get_rotation_xyzw(obj))
    for obj in clutter:
        assert get_relation(obj, ClutterOn).parent in fixture_set, (
            f"ClutterOn parent for '{obj.name}' must be a participating non-clutter fixture; nested clutter is"
            " unsupported"
        )


def _place_clutter_for_fixtures(
    placer: ObjectPlacer,
    objects: list[PlaceableAsset],
    clutter: list[PlaceableAsset],
    fixture_results: list[list[PlacementResult]],
    collision_objects: list[CollisionObject],
    validation: PlacementValidationRunner,
) -> list[list[list[PlacementResult]]]:
    """Keep all clutter restarts grouped by environment, then source fixture layout."""
    results = []
    for env_id, layouts in enumerate(fixture_results):
        fixture_groups = []
        for layout_index, layout in enumerate(layouts):
            copies = _freeze_fixtures(objects, clutter, layout)
            clutter_seed = None
            if placer.params.placement_seed is not None:
                # Keep the release-pass stream independent of fixture sampling.
                seed_key = f"staged-placement:{placer.params.placement_seed}:1:{env_id}:{layout_index}"
                clutter_seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little") & (
                    (1 << 63) - 1
                )
            # Solver anchors share one pose across rows, so extend one complete candidate at a time.
            clutter_layouts = placer._place_ranked_per_env(
                list(copies.values()),
                1,
                1,
                collision_objects,
                placement_seed=clutter_seed,
                validation=validation,
                return_all_candidates=True,
            )[0]
            fixture_groups.append([_combine_layouts(layout, current, clutter, copies) for current in clutter_layouts])
        results.append(fixture_groups)
    return results


def _layout_rank(layout: PlacementResult) -> tuple[int, int, float]:
    """Rank complete layouts by accumulated check failures, then loss."""
    return (*layout.validation_results.get_number_of_required_and_optional_failures, layout.final_loss)


def _freeze_fixtures(
    objects: list[PlaceableAsset], clutter: list[PlaceableAsset], layout: PlacementResult
) -> dict[PlaceableAsset, PlaceableAsset]:
    """Build a solver-only graph, sharing geometry but not poses or relations."""
    copies = {obj: copy(obj) for obj in objects}
    for obj, clone in copies.items():
        if obj not in clutter:
            if not obj.is_anchor:
                pose = get_pose_from_layout(obj, layout)
                quaternion_to_90_deg_z_quarters(pose.rotation_xyzw)
                # A setter would also change the shared spawn config. Only the solver reads this copy.
                clone.initial_pose = pose
            clone.relations = [IsAnchor()]
        else:
            clone.relations = []
            for relation in obj.get_relations():
                relation_copy = copy(relation)
                if isinstance(relation_copy, (Relation, FaceTo)):
                    assert relation_copy.parent in copies, "Relation parent must participate in placement"
                    relation_copy.parent = copies[relation_copy.parent]
                clone.relations.append(relation_copy)
    return copies


def _combine_layouts(
    previous: PlacementResult,
    current: PlacementResult,
    clutter: list[PlaceableAsset],
    copies: dict[PlaceableAsset, PlaceableAsset],
) -> PlacementResult:
    """Preserve earlier verdicts and poses when restoring original asset identities."""
    positions = dict(previous.positions)
    orientations = dict(previous.orientations)
    for obj in clutter:
        clone = copies[obj]
        positions[obj] = current.positions[clone]
        if clone in current.orientations:
            orientations[obj] = current.orientations[clone]
    return PlacementResult(
        validation_results=_merge_validation(previous.validation_results, current.validation_results),
        positions=positions,
        orientations=orientations,
        final_loss=previous.final_loss + current.final_loss,
        attempts=previous.attempts + current.attempts,
    )


def _merge_validation(*reports: PlacementValidationResults) -> PlacementValidationResults:
    """Retain every failed check and the required-check policy across validation passes."""
    checks = {}
    required: set[str] | None = set()
    for report in reports:
        for check, passed in report.validation_results.items():
            checks[check] = checks.get(check, True) and passed
        if report.required_checks is None:
            # Keep the default policy so checks added later are also required.
            required = None
        elif required is not None:
            required.update(report.required_checks)
    return PlacementValidationResults(checks, required)


def _validate_complete_layouts(
    placer: ObjectPlacer,
    objects: list[PlaceableAsset],
    results: list[list[PlacementResult]],
    collision_objects: list[CollisionObject],
) -> None:
    """Run deferred checks such as IK against complete layouts with original asset identities."""
    validators = [validator for validator in placer._validators if validator.run_after_inexpensive_checks]
    if not validators:
        return
    env_bboxes = build_per_env_bounding_boxes(objects, len(results)).get_bounding_boxes_for_all_envs()
    candidates = []
    eligible_layouts = []
    for env_id, layouts in enumerate(results):
        for candidate_id, layout in enumerate(layouts):
            if layout.success:
                candidates.append(
                    PlacementCandidate(
                        env_id,
                        candidate_id,
                        layout.positions,
                        layout.orientations,
                        env_bboxes[env_id],
                        loss=layout.final_loss,
                        validation=layout.validation_results,
                    )
                )
                eligible_layouts.append(layout)
            else:
                skipped = PlacementValidationResults(
                    {validator.check: False for validator in validators}, placer.params.required_checks
                )
                layout.validation_results = _merge_validation(layout.validation_results, skipped)
    if candidates:
        batch = PlacementCandidateBatch(candidates)
        update_candidate_bounds(batch, env_bboxes)
        PlacementValidationRunner(placer.params, validators, placer._visualizer).validate_candidates(
            batch, collision_objects
        )
        for layout, candidate in zip(eligible_layouts, candidates, strict=True):
            layout.validation_results = _merge_validation(layout.validation_results, candidate.validation)
