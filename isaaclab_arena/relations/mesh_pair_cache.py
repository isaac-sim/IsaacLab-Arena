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
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset


class MeshPairEntry(NamedTuple):
    """One directed sphere-to-mesh collision pair."""

    subject: PlaceableAsset
    """Subject (sphere source) object."""

    obstacle: PlaceableAsset | CollisionObject
    """Obstacle (mesh target) object."""

    obstacle_is_fixed: bool
    """True when obstacle is fixed in world coordinates."""

    fixed_obstacle_pos: torch.Tensor | None
    """(3,) world-frame position of the fixed obstacle; None for non-fixed obstacles."""

    fixed_obstacle_rotation: torch.Tensor | None
    """(4,) world-frame xyzw rotation of the fixed obstacle; None for non-fixed obstacles."""

    centers_local: torch.Tensor
    """(S, 3) sphere centers in subject-local frame."""

    radii: torch.Tensor
    """(S,) asset-local sphere radii."""

    warp_mesh: wp.Mesh
    """Warp mesh asset for the obstacle."""


@dataclass(slots=True)
class MeshPairCache:
    """Precomputed data for P directed pairs, S spheres, and M target meshes."""

    all_centers_local: torch.Tensor
    """(S, 3) sphere centers in each subject's local frame, concatenated across pairs."""

    all_radii: torch.Tensor
    """(S,) sphere radii, concatenated across pairs."""

    pair_subject_objs: list[PlaceableAsset]
    """(P,) subject (sphere source) object reference per pair."""

    pair_obstacle_objs: list[PlaceableAsset | CollisionObject]
    """(P,) obstacle (mesh target) object reference per pair."""

    pair_obstacle_is_fixed: list[bool]
    """(P,) True if the obstacle is fixed in world coordinates."""

    pair_fixed_obstacle_pos: list[torch.Tensor | None]
    """(P,) world position for fixed obstacles (None for non-fixed obstacles)."""

    pair_fixed_obstacle_rotation: list[torch.Tensor | None]
    """(P,) fixed-obstacle world xyzw rotations, each (4,); None for dynamic obstacles."""

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
        assert len(self.pair_obstacle_is_fixed) == self.num_pairs, "pair_obstacle_is_fixed length mismatch"
        assert len(self.pair_fixed_obstacle_pos) == self.num_pairs, "pair_fixed_obstacle_pos length mismatch"
        assert len(self.pair_fixed_obstacle_rotation) == self.num_pairs, "pair_fixed_obstacle_rotation length mismatch"
        assert self.all_centers_local.shape[0] == self.total_spheres, "all_centers_local size mismatch"
        assert self.all_radii.shape[0] == self.total_spheres, "all_radii size mismatch"
        assert self.sphere_pair_id.shape[0] == self.total_spheres, "sphere_pair_id size mismatch"
        assert self.sphere_mesh_idx.shape[0] == self.total_spheres, "sphere_mesh_idx size mismatch"
        assert int(self.pair_sphere_count.sum().item()) == self.total_spheres, "pair_sphere_count sum mismatch"
        for i, (is_fixed, pos, rotation) in enumerate(
            zip(
                self.pair_obstacle_is_fixed,
                self.pair_fixed_obstacle_pos,
                self.pair_fixed_obstacle_rotation,
                strict=True,
            )
        ):
            assert not is_fixed or (
                pos is not None and rotation is not None
            ), f"pair {i}: fixed obstacles require position and rotation"
