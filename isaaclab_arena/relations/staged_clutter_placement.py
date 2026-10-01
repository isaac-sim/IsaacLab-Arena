# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Solve fixtures before clutter while returning one complete layout per candidate."""

from __future__ import annotations

import hashlib
from copy import copy
from typing import TYPE_CHECKING

from isaaclab_arena.relations.bounding_box_helpers import has_heterogeneous_objects
from isaaclab_arena.relations.placement_events import get_pose_from_layout, get_rotation_xyzw
from isaaclab_arena.relations.placement_result import PlacementResult
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
    """Rank complete fixture/clutter layouts without changing the input asset graph.

    Args:
        placer: Owner of the reusable solver, validators and placement settings.
        objects: Concrete placement assets, with clutter depending only on ordinary fixtures.
        num_envs: Number of independent environment queues.
        results_per_env: Number of complete candidates to return for each queue.
        collision_objects: Fixed obstacles to avoid in both stages.

    Returns:
        Results keyed by the original assets, ranked by failed checks then combined loss.
        Losses and attempt counts are summed across the two stages.
    """
    params = placer.params
    clutter = [obj for obj in objects if get_relation(obj, ClutterOn) is not None]
    if not clutter:
        return placer._place_ranked_per_env(objects, num_envs, results_per_env, collision_objects)

    assert not has_heterogeneous_objects(objects), "Resolve object sets before staged clutter placement"
    fixtures = [obj for obj in objects if obj not in clutter]
    _validate_fixture_dependencies(fixtures, clutter, params)
    fixture_results = placer._place_ranked_per_env(fixtures, num_envs, results_per_env, collision_objects)
    results = []
    for env_id, layouts in enumerate(fixture_results):
        complete_layouts = []
        for layout_index, fixture_layout in enumerate(layouts):
            copies = _freeze_fixtures(objects, fixtures, fixture_layout)
            clutter_seed = None
            if params.placement_seed is not None:
                # Separate streams without consuming the pool's fixture-candidate seed range.
                seed_key = f"staged-clutter:{params.placement_seed}:{env_id}:{layout_index}".encode()
                clutter_seed = int.from_bytes(hashlib.sha256(seed_key).digest()[:8], "little") & ((1 << 63) - 1)
            # Current solver anchors have one pose shared by all rows, so solve one fixture layout at a time.
            clutter_layout = placer._place_ranked_per_env(
                list(copies.values()), 1, 1, collision_objects, placement_seed=clutter_seed
            )[0][0]
            complete_layouts.append(_combine_layouts(fixture_layout, clutter_layout, clutter, copies))
        complete_layouts.sort(
            key=lambda layout: (
                *layout.validation_results.get_number_of_required_and_optional_failures,
                layout.final_loss,
            )
        )
        results.append(complete_layouts)
    return results


def _validate_fixture_dependencies(
    fixtures: list[PlaceableAsset], clutter: list[PlaceableAsset], params: ObjectPlacerParams
) -> None:
    """Require a two-stage graph and orientations supported by fixed solver geometry."""
    for obj in fixtures:
        for relation in obj.get_relations():
            if isinstance(relation, (Relation, FaceTo)):
                assert relation.parent not in clutter, "Non-clutter fixtures cannot depend on clutter objects"
        if not obj.is_anchor:
            assert (
                not params.random_yaw_init and get_relation(obj, FaceTo) is None
            ), "Staged fixtures require fixed quarter-turn rotations; disable random_yaw_init and FaceTo"
            assert get_relation(obj, RandomAroundSolution) is None, "Staged fixtures cannot randomize after solving"
            quaternion_to_90_deg_z_quarters(get_rotation_xyzw(obj))
    for obj in clutter:
        parent = get_relation(obj, ClutterOn).parent
        assert parent in fixtures, "ClutterOn must reference a first-stage fixture; nested clutter is unsupported"


def _freeze_fixtures(
    objects: list[PlaceableAsset], fixtures: list[PlaceableAsset], layout: PlacementResult
) -> dict[PlaceableAsset, PlaceableAsset]:
    """Build a solver-only graph, sharing geometry but not poses or relations."""
    copies = {obj: copy(obj) for obj in objects}
    for obj, clone in copies.items():
        if obj in fixtures:
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
    fixtures: PlacementResult,
    clutter_layout: PlacementResult,
    clutter: list[PlaceableAsset],
    copies: dict[PlaceableAsset, PlaceableAsset],
) -> PlacementResult:
    """Preserve fixture verdicts and orientations when restoring original asset identities."""
    positions = dict(fixtures.positions)
    orientations = dict(fixtures.orientations)
    for obj in clutter:
        clone = copies[obj]
        positions[obj] = clutter_layout.positions[clone]
        if clone in clutter_layout.orientations:
            orientations[obj] = clutter_layout.orientations[clone]
    checks = {}
    required = set()
    for layout in (fixtures, clutter_layout):
        validation = layout.validation_results
        for check, passed in validation.validation_results.items():
            checks[check] = checks.get(check, True) and passed
        required.update(
            validation.validation_results if validation.required_checks is None else validation.required_checks
        )
    return PlacementResult(
        validation_results=PlacementValidationResults(checks, required),
        positions=positions,
        orientations=orientations,
        final_loss=fixtures.final_loss + clutter_layout.final_loss,
        attempts=fixtures.attempts + clutter_layout.attempts,
    )
