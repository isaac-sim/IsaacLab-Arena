# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Place non-clutter objects first when ClutterOn depends on their poses."""

from __future__ import annotations

import hashlib
from copy import copy
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
    """Place non-clutter objects first, then solve clutter with their poses fixed.

    Args:
        placer: Owner of the reusable solver, validators and placement settings.
        objects: All participating objects, including anchors and clutter supports.
        num_envs: Number of environments to solve.
        results_per_env: Number of complete candidates to return per environment.
        collision_objects: Fixed obstacles to avoid in both passes.

    Returns:
        Complete results keyed by the original assets, ranked by required failures, optional
        failures, then combined loss.
        Both passes contribute losses, attempts and required-check failures.
        Tie-breaking favors variation across retained non-clutter layouts.
    """
    clutter_objects = [obj for obj in objects if get_relation(obj, ClutterOn) is not None]
    assert not has_heterogeneous_objects(objects), "Resolve object sets before staged clutter placement"
    non_clutter_objects = [obj for obj in objects if obj not in clutter_objects]
    _validate_non_clutter_objects(non_clutter_objects, clutter_objects, placer.params)
    inexpensive_validation = PlacementValidationRunner(
        placer.params,
        [validator for validator in placer._validators if not validator.run_after_inexpensive_checks],
        placer._visualizer,
    )
    non_clutter_layouts_per_env = placer._place_ranked_per_env(
        non_clutter_objects, num_envs, results_per_env, collision_objects, validation=inexpensive_validation
    )
    candidates_by_env_and_layout = _place_clutter_for_layouts(
        placer, objects, clutter_objects, non_clutter_layouts_per_env, collision_objects, inexpensive_validation
    )
    _run_deferred_checks(placer, objects, candidates_by_env_and_layout, collision_objects)
    return [
        _select_layouts_for_env(candidates_by_layout, results_per_env)
        for candidates_by_layout in candidates_by_env_and_layout
    ]


def _validate_non_clutter_objects(
    non_clutter_objects: list[PlaceableAsset], clutter_objects: list[PlaceableAsset], params: ObjectPlacerParams
) -> None:
    """Require non-clutter objects to be independent of clutter and have fixed rotations."""
    non_clutter_set = set(non_clutter_objects)
    for obj in non_clutter_objects:
        for relation in obj.get_relations():
            if isinstance(relation, (Relation, FaceTo)):
                assert (
                    relation.parent in non_clutter_set
                ), f"Non-clutter object '{obj.name}' must reference another participating non-clutter object"
        if obj.is_anchor:
            continue
        assert (
            not params.random_yaw_init and get_relation(obj, FaceTo) is None
        ), "Non-clutter objects require fixed quarter-turn rotations; disable random_yaw_init and FaceTo"
        assert get_relation(obj, RandomAroundSolution) is None, "Non-clutter objects cannot randomize after solving"
        quaternion_to_90_deg_z_quarters(get_rotation_xyzw(obj))
    for obj in clutter_objects:
        assert get_relation(obj, ClutterOn).parent in non_clutter_set, (
            f"ClutterOn parent for '{obj.name}' must be a participating non-clutter object; nested clutter is"
            " unsupported"
        )


def _place_clutter_for_layouts(
    placer: ObjectPlacer,
    objects: list[PlaceableAsset],
    clutter_objects: list[PlaceableAsset],
    non_clutter_layouts_per_env: list[list[PlacementResult]],
    collision_objects: list[CollisionObject],
    validation: PlacementValidationRunner,
) -> list[list[list[PlacementResult]]]:
    """Return complete candidates indexed by environment, non-clutter layout, then clutter attempt."""
    # NOTE: Anchors share one pose across rows. This costs num_envs * results_per_env
    # serial clutter solves after the batched non-clutter pass.
    candidates_by_env_and_layout = []
    for env_id, non_clutter_layouts in enumerate(non_clutter_layouts_per_env):
        candidates_by_layout = []
        for layout_index, non_clutter_layout in enumerate(non_clutter_layouts):
            object_copies = _copy_objects_with_fixed_poses(objects, clutter_objects, non_clutter_layout)
            clutter_seed = None
            if placer.params.placement_seed is not None:
                # Keep clutter sampling independent of non-clutter sampling.
                seed_key = f"staged-placement:{placer.params.placement_seed}:1:{env_id}:{layout_index}"
                clutter_seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "little") & (
                    (1 << 63) - 1
                )
            clutter_layouts = placer._place_ranked_per_env(
                list(object_copies.values()),
                num_envs=1,
                results_per_env=1,
                collision_objects=collision_objects,
                placement_seed=clutter_seed,
                validation=validation,
                return_all_candidates=True,
            )[0]
            complete_candidates = [
                _combine_layouts(non_clutter_layout, clutter_layout, clutter_objects, object_copies)
                for clutter_layout in clutter_layouts
            ]
            candidates_by_layout.append(complete_candidates)
        candidates_by_env_and_layout.append(candidates_by_layout)
    return candidates_by_env_and_layout


def _select_layouts_for_env(
    candidates_by_layout: list[list[PlacementResult]], results_per_env: int
) -> list[PlacementResult]:
    """Select complete layouts after validation, favoring different non-clutter poses on ties."""
    for candidates in candidates_by_layout:
        candidates.sort(key=_layout_rank)

    # Each non-clutter layout has the same number of clutter attempts. Consider each
    # layout's best candidate before its next best, so ties favor different support poses.
    interleaved_candidates = []
    for candidates_at_rank in zip(*candidates_by_layout, strict=True):
        interleaved_candidates.extend(candidates_at_rank)
    interleaved_candidates.sort(key=_layout_rank)
    return interleaved_candidates[:results_per_env]


def _layout_rank(layout: PlacementResult) -> tuple[int, int, float]:
    """Rank by required failures, then optional failures, then combined loss."""
    return (*layout.validation_results.get_number_of_required_and_optional_failures, layout.final_loss)


def _copy_objects_with_fixed_poses(
    objects: list[PlaceableAsset], clutter_objects: list[PlaceableAsset], non_clutter_layout: PlacementResult
) -> dict[PlaceableAsset, PlaceableAsset]:
    """Copy the placement graph with non-clutter poses fixed, leaving the original objects unchanged."""
    object_copies = {obj: copy(obj) for obj in objects}
    for obj, clone in object_copies.items():
        if obj not in clutter_objects:
            if not obj.is_anchor:
                pose = get_pose_from_layout(obj, non_clutter_layout)
                quaternion_to_90_deg_z_quarters(pose.rotation_xyzw)
                # A setter would also change the shared spawn config. Only the solver reads this copy.
                clone.initial_pose = pose
            clone.relations = [IsAnchor()]
        else:
            clone.relations = []
            for relation in obj.get_relations():
                relation_copy = copy(relation)
                if isinstance(relation_copy, (Relation, FaceTo)):
                    assert relation_copy.parent in object_copies, "Relation parent must participate in placement"
                    relation_copy.parent = object_copies[relation_copy.parent]
                clone.relations.append(relation_copy)
    return object_copies


def _combine_layouts(
    non_clutter_layout: PlacementResult,
    clutter_layout: PlacementResult,
    clutter_objects: list[PlaceableAsset],
    object_copies: dict[PlaceableAsset, PlaceableAsset],
) -> PlacementResult:
    """Preserve earlier verdicts and poses when restoring original asset identities."""
    positions = dict(non_clutter_layout.positions)
    orientations = dict(non_clutter_layout.orientations)
    for obj in clutter_objects:
        clone = object_copies[obj]
        positions[obj] = clutter_layout.positions[clone]
        if clone in clutter_layout.orientations:
            orientations[obj] = clutter_layout.orientations[clone]
    return PlacementResult(
        validation_results=_merge_validation(non_clutter_layout.validation_results, clutter_layout.validation_results),
        positions=positions,
        orientations=orientations,
        final_loss=non_clutter_layout.final_loss + clutter_layout.final_loss,
        attempts=non_clutter_layout.attempts + clutter_layout.attempts,
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


def _run_deferred_checks(
    placer: ObjectPlacer,
    objects: list[PlaceableAsset],
    candidates_by_env_and_layout: list[list[list[PlacementResult]]],
    collision_objects: list[CollisionObject],
) -> None:
    """Update complete layout reports in place with deferred checks, preserving original asset identities."""
    validators = [validator for validator in placer._validators if validator.run_after_inexpensive_checks]
    if not validators:
        return
    env_bboxes = build_per_env_bounding_boxes(
        objects, len(candidates_by_env_and_layout)
    ).get_bounding_boxes_for_all_envs()
    candidates = []
    eligible_layouts = []
    for env_id, candidates_by_layout in enumerate(candidates_by_env_and_layout):
        complete_layouts = []
        for clutter_candidates in candidates_by_layout:
            complete_layouts.extend(clutter_candidates)
        for candidate_id, layout in enumerate(complete_layouts):
            if layout.success:
                candidates.append(
                    PlacementCandidate(
                        env_id=env_id,
                        candidate_id=candidate_id,
                        positions=layout.positions,
                        orientations=layout.orientations,
                        bboxes=env_bboxes[env_id],
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
