# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""OBB-based no-overlap collision loss computation."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


@dataclass(frozen=True)
class NoOverlapPair:
    """One directed overlap pair."""

    subject: OrientedBoundingBox
    """Subject world boxes; center and half-extents have shape (N, 3)."""

    obstacle: OrientedBoundingBox
    """Detached obstacle world boxes; center and half-extents have shape (N, 3)."""

    tie_break_sign: float = 1.0
    """Directed sign used when coincident boxes have equally deep escape directions."""


def compute_no_overlap_loss_obb(
    state: RelationSolverState,
    no_collision_strategy: NoCollisionLossStrategy,
    clearance_m: float,
    mesh_manager: WarpMeshAndSphereCache | None,
    default_collision_mode: CollisionMode = CollisionMode.BBOX,
    skip_mesh_pairs: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, int]:
    """OBB collision loss summed over all directed pairs, returned per environment.

    - Non-anchor vs fixed obstacle (anchor or collision object): gradient flows to the non-anchor only.
    - Non-anchor vs non-anchor: both objects accumulate gradient (two directed passes).

    Args:
        state: Solver state with positions and batch info.
        no_collision_strategy: Loss strategy for scoring overlap.
        clearance_m: Minimum clearance between objects.
        mesh_manager: Warp mesh cache (for skip_mesh_pairs filtering).
        default_collision_mode: Collision mode used by objects without a per-object override.
        skip_mesh_pairs: Skip pairs handled by mesh or mixed mesh/AABB collision.
        debug: Print per-pair loss when True.

    Returns:
        Tuple of (per-env loss tensor shaped (B,), number of directed pairs scored).
    """
    device = state.device
    batch_size = state.batch_size
    zero_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

    non_anchor_objects = state.optimizable_objects
    anchor_objects = list(state.anchor_objects)
    fixed_obstacles = anchor_objects + list(state.collision_objects)

    on_pairs: set[tuple[int, int]] = set()
    for obj in [*non_anchor_objects, *anchor_objects]:
        for rel in obj.get_relations():
            if isinstance(rel, On):
                on_pairs.add((id(obj), id(rel.parent)))
                on_pairs.add((id(rel.parent), id(obj)))

    world_boxes: dict[PlaceableAsset | CollisionObject, OrientedBoundingBox] = {}
    for obj in non_anchor_objects:
        pos = state.get_position(obj)
        bbox = state.get_bbox(obj)
        world_boxes[obj] = bbox.translated(pos)
    for obstacle in fixed_obstacles:
        bbox = state.get_fixed_obstacle_world_bbox(obstacle)
        world_boxes[obstacle] = OrientedBoundingBox.from_tensors_unchecked(
            bbox.center.expand(batch_size, 3),
            bbox.half_extents.expand(batch_size, 3),
            bbox.rotation_xyzw.expand(batch_size, 4),
        )
    detached_world_boxes = {obj: _detached(bbox) for obj, bbox in world_boxes.items()}

    pairs: list[NoOverlapPair] = []
    pair_names: list[tuple[str, str]] = []

    for child in non_anchor_objects:
        for obstacle in fixed_obstacles:
            if (id(child), id(obstacle)) in on_pairs:
                continue
            if (
                skip_mesh_pairs
                and mesh_manager is not None
                and _fixed_pair_is_covered_by_mesh_collision(
                    state, child, obstacle, mesh_manager, default_collision_mode
                )
            ):
                continue
            pairs.append(NoOverlapPair(world_boxes[child], detached_world_boxes[obstacle]))
            pair_names.append((child.name, obstacle.name))

    for i, child in enumerate(non_anchor_objects):
        for j in range(i + 1, len(non_anchor_objects)):
            other = non_anchor_objects[j]
            if (id(child), id(other)) in on_pairs:
                continue
            if (
                skip_mesh_pairs
                and mesh_manager is not None
                and _dynamic_pair_is_covered_by_mesh_collision(
                    state, child, other, mesh_manager, default_collision_mode
                )
            ):
                continue
            pairs.append(NoOverlapPair(world_boxes[child], detached_world_boxes[other], tie_break_sign=1.0))
            pair_names.append((child.name, other.name))
            pairs.append(NoOverlapPair(world_boxes[other], detached_world_boxes[child], tie_break_sign=-1.0))
            pair_names.append((other.name, child.name))

    num_pairs = len(pairs)
    if not pairs:
        return zero_loss, 0

    subjects = _concatenate_boxes([pair.subject for pair in pairs])
    obstacles = _concatenate_boxes([pair.obstacle for pair in pairs])
    tie_break_signs = torch.tensor(
        [pair.tie_break_sign for pair in pairs],
        dtype=subjects.center.dtype,
        device=device,
    ).repeat_interleave(batch_size)
    penetration = subjects.penetration(
        obstacles,
        clearance_m,
        tie_break_sign=tie_break_signs,
    ).reshape(num_pairs, batch_size)
    pair_loss = no_collision_strategy.compute_loss_batched(penetration)

    if debug:
        for (subject_name, obstacle_name), loss in zip(pair_names, pair_loss):
            print(f"  [NoOverlap] {subject_name} vs {obstacle_name}: loss_per_env={loss.detach().tolist()}")

    return pair_loss.sum(dim=0), num_pairs


def _detached(bbox: OrientedBoundingBox) -> OrientedBoundingBox:
    """Return a box detached from the optimization graph."""
    return OrientedBoundingBox.from_tensors_unchecked(
        bbox.center.detach(),
        bbox.half_extents.detach(),
        bbox.rotation_xyzw.detach(),
    )


def _concatenate_boxes(boxes: list[OrientedBoundingBox]) -> OrientedBoundingBox:
    """Concatenate pair-major box batches without revalidating their tensors."""
    return OrientedBoundingBox.from_tensors_unchecked(
        torch.cat([bbox.center for bbox in boxes], dim=0),
        torch.cat([bbox.half_extents for bbox in boxes], dim=0),
        torch.cat([bbox.rotation_xyzw for bbox in boxes], dim=0),
    )


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
    return state.get_base_bbox(obj).is_batch_invariant()
