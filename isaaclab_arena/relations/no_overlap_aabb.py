# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""AABB-based no-overlap collision loss computation."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relation_loss_strategies import NoCollisionLossStrategy
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


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
    skip_mesh_pairs: bool = False,
    debug: bool = False,
) -> tuple[torch.Tensor, int]:
    """AABB collision loss summed over all directed pairs, returned per environment.

    - Non-anchor vs anchor: gradient flows to the non-anchor only.
    - Non-anchor vs non-anchor: both objects accumulate gradient (two directed passes).

    When skip_mesh_pairs=True, only processes pairs where at least one object lacks a collision mesh.

    Args:
        state: Solver state with positions and batch info.
        no_collision_strategy: Loss strategy for scoring overlap.
        clearance_m: Minimum clearance between objects.
        mesh_manager: Warp mesh cache (for skip_mesh_pairs filtering).
        skip_mesh_pairs: Skip pairs where both objects have meshes.
        debug: Print per-pair loss when True.

    Returns:
        Tuple of (per-env loss tensor shaped (B,), number of directed pairs scored).
    """
    device = state.device
    batch_size = state.batch_size
    zero_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

    non_anchor_objects = state.optimizable_objects
    anchor_objects = list(state.anchor_objects)

    on_pairs: set[tuple[int, int]] = set()
    for obj in [*non_anchor_objects, *anchor_objects]:
        for rel in obj.get_relations():
            if isinstance(rel, On):
                on_pairs.add((id(obj), id(rel.parent)))
                on_pairs.add((id(rel.parent), id(obj)))

    extents: dict[ObjectBase, tuple[torch.Tensor, torch.Tensor]] = {}
    for obj in non_anchor_objects:
        pos = state.get_position(obj)
        bbox = state.get_bbox(obj)
        extents[obj] = (pos + bbox.min_point, pos + bbox.max_point)
    for anchor in anchor_objects:
        anchor_world_bbox = anchor.get_world_bounding_box().to(device)
        extents[anchor] = (
            anchor_world_bbox.min_point.expand(batch_size, 3),
            anchor_world_bbox.max_point.expand(batch_size, 3),
        )

    pairs: list[NoOverlapPair] = []
    pair_names: list[tuple[str, str]] = []

    for child in non_anchor_objects:
        child_min, child_max = extents[child]
        for anchor in anchor_objects:
            if (id(child), id(anchor)) in on_pairs:
                continue
            if (
                skip_mesh_pairs
                and mesh_manager is not None
                and mesh_manager.get_collision_mesh(child) is not None
                and mesh_manager.get_collision_mesh(anchor) is not None
            ):
                continue
            anchor_min, anchor_max = extents[anchor]
            pairs.append(NoOverlapPair(child_min, child_max, anchor_min, anchor_max))
            pair_names.append((child.name, anchor.name))

    for i, child in enumerate(non_anchor_objects):
        child_min, child_max = extents[child]
        for j in range(i + 1, len(non_anchor_objects)):
            other = non_anchor_objects[j]
            if (id(child), id(other)) in on_pairs:
                continue
            if (
                skip_mesh_pairs
                and mesh_manager is not None
                and mesh_manager.get_collision_mesh(child) is not None
                and mesh_manager.get_collision_mesh(other) is not None
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
