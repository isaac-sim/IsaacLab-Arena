# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""AABB-based no-overlap collision loss computation."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


@dataclass(frozen=True)
class NoOverlapPair:
    """One directed overlap penalty: the subject box is pushed off the (detached) obstacle box.

    Dimensions: B = batch_size (num envs).
    """

    subject_min: torch.Tensor
    """(B, 3) world-space min corner of the subject box."""

    subject_max: torch.Tensor
    """(B, 3) world-space max corner of the subject box."""

    obstacle_min: torch.Tensor
    """(B, 3) world-space min corner of the obstacle box."""

    obstacle_max: torch.Tensor
    """(B, 3) world-space max corner of the obstacle box."""


def compute_no_overlap_loss_aabb(
    state: RelationSolverState,
    no_collision_strategy: NoCollisionLossStrategy,
    clearance_m: float,
    mesh_manager: WarpMeshAndSphereCache | None,
    default_collision_mode: CollisionMode = CollisionMode.BBOX,
    skip_mesh_pairs: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, int]:
    """AABB collision loss summed over all directed pairs, returned per environment.

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

    extents: dict[ObjectBase | CollisionObject, tuple[torch.Tensor, torch.Tensor]] = {}
    for obj in non_anchor_objects:
        pos = state.get_position(obj)
        bbox = state.get_bbox(obj)
        extents[obj] = (pos + bbox.min_point, pos + bbox.max_point)
    for obstacle in fixed_obstacles:
        obstacle_world_bbox = state.get_fixed_obstacle_world_bbox(obstacle)
        extents[obstacle] = (
            obstacle_world_bbox.min_point.expand(batch_size, 3),
            obstacle_world_bbox.max_point.expand(batch_size, 3),
        )

    pairs: list[NoOverlapPair] = []
    pair_names: list[tuple[str, str]] = []

    for child in non_anchor_objects:
        child_min, child_max = extents[child]
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
            obstacle_min, obstacle_max = extents[obstacle]
            pairs.append(NoOverlapPair(child_min, child_max, obstacle_min, obstacle_max))
            pair_names.append((child.name, obstacle.name))

    for i, child in enumerate(non_anchor_objects):
        child_min, child_max = extents[child]
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
            other_min, other_max = extents[other]
            pairs.append(NoOverlapPair(child_min, child_max, other_min.detach(), other_max.detach()))
            pair_names.append((child.name, other.name))
            pairs.append(NoOverlapPair(other_min, other_max, child_min.detach(), child_max.detach()))
            pair_names.append((other.name, child.name))

    num_pairs = len(pairs)
    if not pairs:
        return zero_loss, 0

    subject_min = torch.stack([p.subject_min for p in pairs], dim=0)
    subject_max = torch.stack([p.subject_max for p in pairs], dim=0)
    obstacle_min = torch.stack([p.obstacle_min for p in pairs], dim=0)
    obstacle_max = torch.stack([p.obstacle_max for p in pairs], dim=0)

    pair_loss = no_collision_strategy.compute_loss_batched(
        clearance_m, subject_min, subject_max, obstacle_min, obstacle_max
    )

    if debug:
        for (subject_name, obstacle_name), loss in zip(pair_names, pair_loss):
            print(f"  [NoOverlap] {subject_name} vs {obstacle_name}: loss={loss.mean().item():.6f}")

    return pair_loss.sum(dim=0), num_pairs


def _fixed_pair_is_covered_by_mesh_collision(
    state: RelationSolverState,
    subject: ObjectBase,
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
    a: ObjectBase,
    b: ObjectBase,
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
    obj: ObjectBase,
    mesh_manager: WarpMeshAndSphereCache,
    default_collision_mode: CollisionMode,
) -> bool:
    """Return True when MESH loss can represent obj as mesh or one bbox proxy."""
    mesh = mesh_manager.get_collision_mesh(obj) if object_uses_mesh_collision(obj, default_collision_mode) else None
    if mesh is not None:
        return True
    return state.get_bbox(obj).is_batch_invariant()
