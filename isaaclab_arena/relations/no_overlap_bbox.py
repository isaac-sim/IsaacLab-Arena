# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounding-box no-overlap collision loss computation."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On
from isaaclab_arena.utils.bounding_box import CachedOverlapVolumeUpperBound

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

_MIN_LOSS_HALF_EXTENT_M = 1e-3
"""Smallest box half-extent in the oriented loss, so a zero-thickness box has an overlap volume to reduce."""

_MIN_VOLUME_SUM_M3 = 1e-12
"""Floor on the sum of the two face-axis volumes, so the harmonic mean stays finite when both are zero."""


@dataclass(frozen=True)
class NoOverlapPair:
    """One directed penalty, with gradients applied only to the subject."""

    subject: PlaceableAsset
    """Asset whose position receives this penalty's gradient."""
    obstacle: PlaceableAsset | CollisionObject
    """Object treated as fixed for this directed penalty."""


@dataclass(frozen=True)
class NoOverlapPairCache:
    """Fixed pair routing and overlap-volume upper bound for one solve and collision configuration."""

    box_pairs: list[NoOverlapPair]
    """Pairs whose collision geometry matches box validation, in directed penalty order."""
    mesh_fallback_pairs: list[NoOverlapPair]
    """Pairs using enclosing-box mesh proxies, in directed penalty order."""
    overlap_bound: CachedOverlapVolumeUpperBound | None
    """Upper bound prepared for box_pairs, or None when there are no box pairs or every collision box is an
    AxisAlignedBoundingBox."""


def compute_no_overlap_loss_bbox(
    state: RelationSolverState,
    cache: NoOverlapPairCache,
    no_collision_strategy: NoCollisionLossStrategy,
    clearance_m: float,
    debug: bool = False,
) -> tuple[torch.Tensor, int]:
    """Sum overlap-volume penalties over the cached directed pairs, returning (B,) losses and pair count.

    Args:
        state: Current positions and fixed collision geometry for one solve.
        cache: Pair routing and upper bound prepared for state.
        no_collision_strategy: Penalty scale and aligned-box interval loss.
        clearance_m: Nonnegative clearance kept around each directed pair's obstacle.
        debug: Print the mean loss for each directed pair.

    Returns:
        Per-candidate losses of shape (B,) and the number of directed pairs.
    """
    total = torch.zeros(state.batch_size, device=state.device)
    for group, enclosing_bounds in ((cache.box_pairs, False), (cache.mesh_fallback_pairs, True)):
        if not group:
            continue
        if enclosing_bounds or state.collision_boxes_axis_aligned:
            pair_loss = _compute_aabb_pair_losses(state, group, no_collision_strategy, clearance_m, enclosing_bounds)
        else:
            pair_loss = _compute_oriented_pair_losses(
                state, group, cache.overlap_bound, no_collision_strategy, clearance_m
            )
        total = total + pair_loss.sum(dim=0)
        if debug:
            for pair, loss in zip(group, pair_loss):
                print(f"  [NoOverlap] {pair.subject.name} vs {pair.obstacle.name}: loss={loss.mean().item():.6f}")
    return total, len(cache.box_pairs) + len(cache.mesh_fallback_pairs)


def get_on_linked_pairs(objects: list[PlaceableAsset]) -> set[tuple[int, int]]:
    """Return (id(a), id(b)) for both orders of every On-linked object pair."""
    on_pairs = set()
    for obj in objects:
        for relation in obj.get_relations():
            if isinstance(relation, On):
                on_pairs.add((id(obj), id(relation.parent)))
                on_pairs.add((id(relation.parent), id(obj)))
    return on_pairs


def prepare_no_overlap_pair_cache(
    state: RelationSolverState,
    on_pairs: set[tuple[int, int]],
    mesh_manager: WarpMeshAndSphereCache | None = None,
    default_collision_mode: CollisionMode = CollisionMode.BBOX,
) -> NoOverlapPairCache:
    """Prepare pair routing and the overlap-volume upper bound; a mesh_manager excludes pairs the mesh loss covers.

    Args:
        state: Solver state with fixed collision bounds.
        on_pairs: On-linked pairs to skip.
        mesh_manager: Collision mesh cache when the mesh loss runs, otherwise None.
        default_collision_mode: Mode for objects without an explicit override.

    Returns:
        Ordered pair groups and the prepared upper bound for this solve.
    """
    pairs = _list_no_overlap_pairs(state, on_pairs, mesh_manager, default_collision_mode)
    box_pairs, mesh_fallback_pairs = _split_mesh_fallback_pairs(state, pairs, mesh_manager, default_collision_mode)
    overlap_bound = None
    if box_pairs and not state.collision_boxes_axis_aligned:
        overlap_bound = _build_overlap_bound(state, box_pairs)
    return NoOverlapPairCache(box_pairs, mesh_fallback_pairs, overlap_bound)


def _list_no_overlap_pairs(
    state: RelationSolverState,
    on_pairs: set[tuple[int, int]],
    mesh_manager: WarpMeshAndSphereCache | None,
    default_collision_mode: CollisionMode,
) -> list[NoOverlapPair]:
    """List directed movable-vs-fixed then movable-vs-movable pairs, skipping On and mesh-loss pairs."""
    movable = state.optimizable_objects
    fixed = list(state.anchor_objects) + state.collision_objects
    pairs = []
    for child in movable:
        for obstacle in fixed:
            if (id(child), id(obstacle)) in on_pairs:
                continue
            if mesh_manager is not None and _fixed_pair_is_covered_by_mesh_collision(
                state, child, obstacle, mesh_manager, default_collision_mode
            ):
                continue
            pairs.append(NoOverlapPair(child, obstacle))
    for index, child in enumerate(movable):
        for other in movable[index + 1 :]:
            if (id(child), id(other)) in on_pairs:
                continue
            if mesh_manager is not None and _dynamic_pair_is_covered_by_mesh_collision(
                state, child, other, mesh_manager, default_collision_mode
            ):
                continue
            pairs.extend((NoOverlapPair(child, other), NoOverlapPair(other, child)))
    return pairs


def _split_mesh_fallback_pairs(
    state: RelationSolverState,
    pairs: list[NoOverlapPair],
    mesh_manager: WarpMeshAndSphereCache | None,
    default_collision_mode: CollisionMode,
) -> tuple[list[NoOverlapPair], list[NoOverlapPair]]:
    """Split pairs into box pairs and pairs that mesh validation checks with enclosing-box proxies."""
    if mesh_manager is None:
        return pairs, []
    mesh_objects = set()
    for obj in [*state.optimizable_objects, *state.anchor_objects, *state.collision_objects]:
        if object_uses_mesh_collision(obj, default_collision_mode) and mesh_manager.get_collision_mesh(obj) is not None:
            mesh_objects.add(obj)
    box_pairs, mesh_fallback_pairs = [], []
    background_objects = set(state.collision_objects)
    for pair in pairs:
        uses_mesh_validation = pair.obstacle in mesh_objects or (
            pair.subject in mesh_objects and pair.obstacle not in background_objects
        )
        if uses_mesh_validation:
            # Mesh validation uses enclosing-box sphere proxies for these unsupported pairs.
            mesh_fallback_pairs.append(pair)
        else:
            box_pairs.append(pair)
    return box_pairs, mesh_fallback_pairs


def _build_overlap_bound(state: RelationSolverState, box_pairs: list[NoOverlapPair]) -> CachedOverlapVolumeUpperBound:
    """Stack fixed half-extents and axes for box_pairs into an overlap-volume upper bound."""
    geometry = {}
    for pair in box_pairs:
        for obj in (pair.subject, pair.obstacle):
            if obj not in geometry:
                geometry[obj] = state.get_collision_geometry(obj)
    return CachedOverlapVolumeUpperBound(
        torch.stack([geometry[p.subject][1] for p in box_pairs]).clamp_min(_MIN_LOSS_HALF_EXTENT_M),
        torch.stack([geometry[p.subject][2] for p in box_pairs]),
        torch.stack([geometry[p.obstacle][1].detach() for p in box_pairs]).clamp_min(_MIN_LOSS_HALF_EXTENT_M),
        torch.stack([geometry[p.obstacle][2].detach() for p in box_pairs]),
    )


def _compute_aabb_pair_losses(
    state: RelationSolverState,
    pairs: list[NoOverlapPair],
    strategy: NoCollisionLossStrategy,
    clearance_m: float,
    enclosing_bounds: bool,
) -> torch.Tensor:
    """Return (P, B) interval-volume penalties for P directed pairs and B candidates, keeping obstacles detached."""
    extents = {}
    for obj in state.optimizable_objects:
        box = state.get_bbox(obj) if enclosing_bounds else state.get_collision_bbox(obj)
        origin_W = state.get_position(obj)
        extents[obj] = (
            (box.min_point + origin_W).expand(state.batch_size, 3),
            (box.max_point + origin_W).expand(state.batch_size, 3),
        )
    for obj in [*state.anchor_objects, *state.collision_objects]:
        box = state.get_fixed_obstacle_world_bbox(obj)
        extents[obj] = (box.min_point.expand(state.batch_size, 3), box.max_point.expand(state.batch_size, 3))
    return strategy.compute_loss_batched(
        clearance_m,
        torch.stack([extents[p.subject][0] for p in pairs]),
        torch.stack([extents[p.subject][1] for p in pairs]),
        torch.stack([extents[p.obstacle][0].detach() for p in pairs]),
        torch.stack([extents[p.obstacle][1].detach() for p in pairs]),
    )


def _compute_oriented_pair_losses(
    state: RelationSolverState,
    pairs: list[NoOverlapPair],
    overlap_bound: CachedOverlapVolumeUpperBound,
    strategy: NoCollisionLossStrategy,
    clearance_m: float,
) -> torch.Tensor:
    """Return (P, B) overlap-volume penalties for P directed pairs and B candidates at the current positions.

    The penalty is the harmonic mean of the interval-overlap volumes along the subject's and the obstacle's face axes.
    It is zero exactly when either volume is zero and equals the intersection volume for axis-aligned pairs. When only
    one box's projections sit inside the other's, the other box's volume still gives a gradient.
    """
    centers_W = {}
    for obj in [*state.optimizable_objects, *state.anchor_objects, *state.collision_objects]:
        centers_W[obj] = state.get_collision_center_W(obj)
    subject_centers_W = torch.stack([centers_W[p.subject] for p in pairs])
    obstacle_centers_W = torch.stack([centers_W[p.obstacle].detach() for p in pairs])
    volume_A, volume_B = overlap_bound.compute_face_axis_volumes(
        subject_centers_W, obstacle_centers_W, margin=clearance_m
    )
    return strategy.slope * 2 * volume_A * volume_B / (volume_A + volume_B).clamp_min(_MIN_VOLUME_SUM_M3)


def _fixed_pair_is_covered_by_mesh_collision(
    state: RelationSolverState,
    subject: PlaceableAsset,
    obstacle: CollisionObject,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss handles subject vs fixed obstacle."""
    obstacle_mesh = (
        mesh_manager.get_collision_mesh(obstacle)
        if object_uses_mesh_collision(obstacle, default_collision_mode)
        else None
    )
    return obstacle_mesh is not None and _has_mesh_or_invariant_bbox(
        state, subject, mesh_manager, default_collision_mode
    )


def _dynamic_pair_is_covered_by_mesh_collision(
    state: RelationSolverState,
    a: PlaceableAsset,
    b: PlaceableAsset,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss handles a non-anchor object pair."""
    a_mesh = mesh_manager.get_collision_mesh(a) if object_uses_mesh_collision(a, default_collision_mode) else None
    b_mesh = mesh_manager.get_collision_mesh(b) if object_uses_mesh_collision(b, default_collision_mode) else None
    if a_mesh is None and b_mesh is None:
        return False
    return _has_mesh_or_invariant_bbox(state, a, mesh_manager, default_collision_mode) and _has_mesh_or_invariant_bbox(
        state, b, mesh_manager, default_collision_mode
    )


def _has_mesh_or_invariant_bbox(
    state: RelationSolverState,
    obj: PlaceableAsset,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss can represent obj as mesh or one bbox proxy."""
    mesh = mesh_manager.get_collision_mesh(obj) if object_uses_mesh_collision(obj, default_collision_mode) else None
    if mesh is not None:
        return True
    return state.get_bbox(obj).is_batch_invariant()
