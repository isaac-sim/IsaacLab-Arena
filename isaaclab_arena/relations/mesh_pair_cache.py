# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed container for precomputed mesh-collision pair data."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import warp as wp

    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass(slots=True)
class MeshPairCache:
    """Precomputed per-pair collision data for the vectorized multi-mesh kernel."""

    all_centers_local: torch.Tensor
    """(S, 3) sphere centers in each child's local frame, concatenated across pairs."""

    all_radii: torch.Tensor
    """(S,) sphere radii, concatenated across pairs."""

    pair_child_objs: list[ObjectBase]
    """Per-pair child object reference."""

    pair_parent_objs: list[ObjectBase]
    """Per-pair parent/target object reference."""

    pair_is_anchor: list[bool]
    """Per-pair flag: True if parent is a static anchor."""

    pair_anchor_pos: list[torch.Tensor | None]
    """Per-pair world position of anchors (None for non-anchor parents)."""

    pair_anchor_yaw: list[float]
    """Per-pair anchor yaw in radians (0.0 for non-anchor parents)."""

    pair_c_bbox_min: torch.Tensor
    """(P, B, 3) child bbox min corners for broadphase."""

    pair_c_bbox_max: torch.Tensor
    """(P, B, 3) child bbox max corners for broadphase."""

    pair_p_bbox_min: torch.Tensor
    """(P, B, 3) parent bbox min corners for broadphase."""

    pair_p_bbox_max: torch.Tensor
    """(P, B, 3) parent bbox max corners for broadphase."""

    pair_max_r: torch.Tensor
    """(P,) max sphere radius per pair (broadphase margin)."""

    sphere_pair_id: torch.Tensor
    """(S,) maps each sphere to its pair index for segment reduction."""

    sphere_mesh_idx: torch.Tensor
    """(S,) per-sphere index into mesh_id_array."""

    pair_sphere_count: torch.Tensor
    """(P,) number of spheres per pair (for mean reduction)."""

    mesh_id_array: wp.array
    """Warp uint64 array of mesh IDs for the multi-mesh kernel."""

    num_pairs: int
    """Total number of active object pairs."""

    total_spheres: int
    """Total number of sphere queries across all pairs."""

    def __post_init__(self) -> None:
        assert len(self.pair_child_objs) == self.num_pairs, "pair_child_objs length mismatch"
        assert len(self.pair_parent_objs) == self.num_pairs, "pair_parent_objs length mismatch"
        assert len(self.pair_is_anchor) == self.num_pairs, "pair_is_anchor length mismatch"
        assert self.all_centers_local.shape[0] == self.total_spheres, "all_centers_local size mismatch"
        assert self.all_radii.shape[0] == self.total_spheres, "all_radii size mismatch"
        assert self.sphere_pair_id.shape[0] == self.total_spheres, "sphere_pair_id size mismatch"
        assert self.sphere_mesh_idx.shape[0] == self.total_spheres, "sphere_mesh_idx size mismatch"
        assert int(self.pair_sphere_count.sum().item()) == self.total_spheres, "pair_sphere_count sum mismatch"
        for i, (is_anchor, pos) in enumerate(zip(self.pair_is_anchor, self.pair_anchor_pos)):
            assert not is_anchor or pos is not None, f"pair {i}: is_anchor=True but anchor_pos is None"
