# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed container for precomputed mesh-collision pair data."""

from __future__ import annotations

import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import warp as wp

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


class MeshPairEntry(NamedTuple):
    """One directed sphere-to-mesh collision pair (subject spheres vs obstacle mesh).

    Dimensions: S = sphere count for this pair's subject, B = batch_size.
    """

    subject: ObjectBase
    """Subject (sphere source) object."""

    obstacle: ObjectBase
    """Obstacle (mesh target) object."""

    is_anchor: bool
    """True when obstacle is a static anchor."""

    anchor_pos: torch.Tensor | None
    """(3,) world-frame position of the anchor obstacle; None for non-anchors."""

    anchor_yaw: float
    """Anchor Z-yaw in radians (0.0 for non-anchors)."""

    centers_local: torch.Tensor
    """(S, 3) sphere centers in subject-local frame."""

    radii: torch.Tensor
    """(S,) sphere radii."""

    subject_bbox_min: torch.Tensor
    """(B, 3) subject bounding box min corners."""

    subject_bbox_max: torch.Tensor
    """(B, 3) subject bounding box max corners."""

    obstacle_bbox_min: torch.Tensor
    """(B, 3) obstacle bounding box min corners."""

    obstacle_bbox_max: torch.Tensor
    """(B, 3) obstacle bounding box max corners."""

    warp_mesh: wp.Mesh
    """Warp mesh asset for the obstacle."""


@dataclass(slots=True)
class MeshPairCache:
    """Precomputed per-pair collision data for the vectorized multi-mesh kernel.

    Dimensions: P = num_pairs (ordered subject/obstacle pairs), B = batch_size (num envs),
    S = total_spheres (sum of sphere counts across all P pairs; each subject object is decomposed
    into multiple covering spheres via greedy_sphere_decomposition),
    M = num_unique_meshes (distinct collision meshes referenced by the pairs).
    """

    all_centers_local: torch.Tensor
    """(S, 3) sphere centers in each subject's local frame, concatenated across pairs."""

    all_radii: torch.Tensor
    """(S,) sphere radii, concatenated across pairs."""

    pair_subject_objs: list[ObjectBase]
    """(P,) subject (sphere source) object reference per pair."""

    pair_obstacle_objs: list[ObjectBase]
    """(P,) obstacle (mesh target) object reference per pair."""

    pair_is_anchor: list[bool]
    """(P,) True if the obstacle is a static anchor."""

    pair_anchor_pos: list[torch.Tensor | None]
    """(P,) world position for anchor obstacles (None for non-anchors)."""

    pair_anchor_yaw: list[float]
    """(P,) anchor yaw in radians (0.0 for non-anchors)."""

    pair_subject_bbox_min: torch.Tensor
    """(P, B, 3) subject bounding box min corners."""

    pair_subject_bbox_max: torch.Tensor
    """(P, B, 3) subject bounding box max corners."""

    pair_obstacle_bbox_min: torch.Tensor
    """(P, B, 3) obstacle bounding box min corners."""

    pair_obstacle_bbox_max: torch.Tensor
    """(P, B, 3) obstacle bounding box max corners."""

    pair_max_radius: torch.Tensor
    """(P,) maximum sphere radius across all spheres in each pair."""

    sphere_pair_id: torch.Tensor
    """(S,) pair index for each sphere."""

    sphere_mesh_idx: torch.Tensor
    """(S,) per-sphere index into mesh_id_array."""

    pair_sphere_count: torch.Tensor
    """(P,) number of spheres for each pair."""

    mesh_id_array: wp.array
    """(M,) Warp uint64 array of unique collision-mesh IDs."""

    num_pairs: int
    """Total number of active object pairs."""

    total_spheres: int
    """Total number of sphere queries across all pairs."""

    def __post_init__(self) -> None:
        assert len(self.pair_subject_objs) == self.num_pairs, "pair_subject_objs length mismatch"
        assert len(self.pair_obstacle_objs) == self.num_pairs, "pair_obstacle_objs length mismatch"
        assert len(self.pair_is_anchor) == self.num_pairs, "pair_is_anchor length mismatch"
        assert self.all_centers_local.shape[0] == self.total_spheres, "all_centers_local size mismatch"
        assert self.all_radii.shape[0] == self.total_spheres, "all_radii size mismatch"
        assert self.sphere_pair_id.shape[0] == self.total_spheres, "sphere_pair_id size mismatch"
        assert self.sphere_mesh_idx.shape[0] == self.total_spheres, "sphere_mesh_idx size mismatch"
        assert int(self.pair_sphere_count.sum().item()) == self.total_spheres, "pair_sphere_count sum mismatch"
        for i, (is_anchor, pos) in enumerate(zip(self.pair_is_anchor, self.pair_anchor_pos)):
            assert not is_anchor or pos is not None, f"pair {i}: is_anchor=True but anchor_pos is None"
