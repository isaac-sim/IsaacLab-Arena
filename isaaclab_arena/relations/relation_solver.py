# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import time
import torch
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, cast

import warp as wp
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.relations.collision_mode import CollisionMode
from isaaclab_arena.relations.mesh_pair_cache import MeshPairCache
from isaaclab_arena.relations.relation_loss_strategies import (
    NoCollisionLossStrategy,
    RelationLossStrategy,
    UnaryRelationLossStrategy,
)
from isaaclab_arena.relations.relation_solver_params import RelationSolverParams
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import On, Relation, RelationBase, UnaryRelation
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
from isaaclab_arena.relations.warp_sdf_kernels import clamp_sdf_sentinel, multi_mesh_sdf
from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox
from isaaclab_arena.utils.pose import Pose, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


@dataclass(frozen=True)
class NoOverlapPair:
    """One directed overlap penalty: the subject box is pushed off the (detached) obstacle box.

    Each extent tensor is shaped (batch_size, 3).
    """

    subject_min: torch.Tensor
    subject_max: torch.Tensor
    obstacle_min: torch.Tensor
    obstacle_max: torch.Tensor


class MeshPairEntry(NamedTuple):
    """One directed sphere-to-mesh collision pair (subject spheres vs obstacle mesh).

    Dimensions: S = sphere count for this pair's subject, B = batch_size.
    """

    subject: ObjectBase
    obstacle: ObjectBase
    is_anchor: bool
    anchor_pos: torch.Tensor | None  # (3,) world-frame position of the anchor obstacle
    anchor_yaw: float
    centers_local: torch.Tensor  # (S, 3) sphere centers in subject-local frame
    radii: torch.Tensor  # (S,)
    subject_bbox_min: torch.Tensor  # (B, 3) subject bbox min corners
    subject_bbox_max: torch.Tensor  # (B, 3)
    obstacle_bbox_min: torch.Tensor  # (B, 3) obstacle bbox min corners
    obstacle_bbox_max: torch.Tensor  # (B, 3)
    warp_mesh: object  # wp.Mesh (untyped to avoid import at runtime)


class RelationSolver:
    """Differentiable solver for 3D spatial relations of IsaacLab Arena Objects.

    Uses the Strategy pattern for loss computation: each Relation type has a
    corresponding RelationLossStrategy that handles the actual loss calculation.
    """

    POSITION_HISTORY_SAVE_INTERVAL = 10
    """Save position snapshot every N iterations (when save_position_history is enabled)."""

    def __init__(
        self,
        params: RelationSolverParams | None = None,
    ):
        """
        Args:
            params: Solver configuration parameters. If None, uses defaults.
        """
        self.params = params or RelationSolverParams()
        # High slope (vs 10-100 for relation strategies) so overlap avoidance dominates.
        self._no_collision_strategy = NoCollisionLossStrategy(slope=10000.0)
        self._last_loss_history: list[float] = []
        self._last_position_history: list = []
        self._last_loss_per_env: torch.Tensor | None = None
        self._last_no_overlap_pair_count: int = 0
        self._mesh_orientations: list[dict[ObjectBase, float]] | None = None
        self._warned_no_mesh: set[str] = set()
        self._mesh_manager: WarpMeshAndSphereCache | None = None
        self._mesh_cache_forward: MeshPairCache | None = None
        self._mesh_cache_reverse: MeshPairCache | None = None

    def _get_strategy(self, relation: RelationBase) -> RelationLossStrategy | UnaryRelationLossStrategy:
        """Look up the loss strategy for a relation type.

        Args:
            relation: The relation to find a strategy for.
        """
        strategy = self.params.strategies.get(type(relation))
        if strategy is None:
            raise ValueError(
                f"No loss strategy registered for {type(relation).__name__}. "
                f"Available strategies: {list(self.params.strategies.keys())}"
            )
        return strategy

    def _compute_total_loss(
        self,
        state: RelationSolverState,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute total loss from all relations using registered strategies.

        Args:
            state: Current optimization state with object positions and
                optional per-env bounding boxes (accessed via state.get_bbox).
            debug: If True, print detailed loss breakdown.

        Returns:
            Scalar loss tensor (mean over environments).
        """
        batch_size = state.batch_size
        device = state.device
        total_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

        for obj in state.optimizable_objects:
            for relation in obj.get_spatial_relations():
                child_pos = state.get_position(obj)
                strategy = self._get_strategy(relation)
                child_bbox = state.get_bbox(obj)

                if isinstance(relation, UnaryRelation):
                    unary_strategy = cast(UnaryRelationLossStrategy, strategy)
                    loss = unary_strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=child_bbox,
                    )
                    if debug:
                        _print_unary_relation_debug(obj, relation, child_pos[0], loss.mean())
                # Binary relation (On, NextTo, etc.)
                elif isinstance(relation, Relation):
                    relation_strategy = cast(RelationLossStrategy, strategy)
                    parent = relation.parent
                    if parent in state.anchor_objects:
                        parent_world_bbox = parent.get_world_bounding_box().to(device)
                    else:
                        parent_pos = state.get_position(parent)
                        parent_bbox = state.get_bbox(parent)
                        parent_world_bbox = parent_bbox.translated(parent_pos)
                    loss = relation_strategy.compute_loss(
                        relation=relation,
                        child_pos=child_pos,
                        child_bbox=child_bbox,
                        parent_world_bbox=parent_world_bbox,
                    )
                    if debug:
                        parent_pos = state.get_position(parent)
                        _print_relation_debug(obj, relation, child_pos[0], parent_pos[0], loss.mean())
                else:
                    raise ValueError(f"Unknown relation type: {type(relation).__name__}")

                total_loss = total_loss + loss

        # Add built-in no-overlap loss between all object pairs
        total_loss = total_loss + self._compute_no_overlap_loss(state, debug)

        self._last_loss_per_env = total_loss.detach().clone()
        return total_loss.mean()

    def _compute_no_overlap_loss(
        self,
        state: RelationSolverState,
        debug: bool = False,
    ) -> torch.Tensor:
        """Compute pairwise no-overlap loss, skipping On-linked pairs."""
        if self.params.collision_mode == CollisionMode.MESH:
            mesh_loss = self._compute_no_overlap_loss_mesh(state, debug)
            aabb_loss = self._compute_no_overlap_loss_aabb(state, debug, skip_mesh_pairs=True)
            return mesh_loss + aabb_loss
        else:
            return self._compute_no_overlap_loss_aabb(state, debug)

    def _compute_no_overlap_loss_aabb(
        self,
        state: RelationSolverState,
        debug: bool,
        skip_mesh_pairs: bool = False,
    ) -> torch.Tensor:
        """Per-pair AABB collision loss.

        - Non-anchor vs anchor: gradient flows to the non-anchor only.
        - Non-anchor vs non-anchor: both objects accumulate gradient (two directed passes).

        When skip_mesh_pairs=True, only processes pairs where at least one object
        lacks a collision mesh.
        """
        device = state.device
        batch_size = state.batch_size
        zero_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

        non_anchor_objects = state.optimizable_objects
        anchor_objects = list(state.anchor_objects)

        # Collect On-relation pairs to skip (stacked objects shouldn't repel each other).
        on_pairs: set[tuple[int, int]] = set()
        for obj in [*non_anchor_objects, *anchor_objects]:
            for rel in obj.get_relations():
                if isinstance(rel, On):
                    on_pairs.add((id(obj), id(rel.parent)))
                    on_pairs.add((id(rel.parent), id(obj)))

        # World-space (min, max) extents once per object, shape (batch, 3). Non-anchor
        # extents carry gradient through the object's position; anchor extents are constant.
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
        pair_names: list[tuple[str, str]] = []  # for the debug=True print

        # Non-anchor vs each anchor: one pass (anchor is constant, so no detach).
        for child in non_anchor_objects:
            child_min, child_max = extents[child]
            for anchor in anchor_objects:
                if (id(child), id(anchor)) in on_pairs:
                    continue
                if (
                    skip_mesh_pairs
                    and self._mesh_manager is not None
                    and self._mesh_manager.get_collision_mesh(child) is not None
                    and self._mesh_manager.get_collision_mesh(anchor) is not None
                ):
                    continue
                anchor_min, anchor_max = extents[anchor]
                pairs.append(NoOverlapPair(child_min, child_max, anchor_min, anchor_max))
                pair_names.append((child.name, anchor.name))

        # Non-anchor vs non-anchor: score both directions (detach the obstacle) so each gets gradient.
        for i, child in enumerate(non_anchor_objects):
            child_min, child_max = extents[child]
            for j in range(i + 1, len(non_anchor_objects)):
                other = non_anchor_objects[j]
                if (id(child), id(other)) in on_pairs:
                    continue
                if (
                    skip_mesh_pairs
                    and self._mesh_manager is not None
                    and self._mesh_manager.get_collision_mesh(child) is not None
                    and self._mesh_manager.get_collision_mesh(other) is not None
                ):
                    continue
                other_min, other_max = extents[other]
                pairs.append(NoOverlapPair(child_min, child_max, other_min.detach(), other_max.detach()))
                pair_names.append((child.name, other.name))
                pairs.append(NoOverlapPair(other_min, other_max, child_min.detach(), child_max.detach()))
                pair_names.append((other.name, child.name))

        self._last_no_overlap_pair_count = len(pairs)
        if not pairs:
            return zero_loss

        # Stack to (num_pairs, batch_size, 3) and score every pair in one op.
        subject_min = torch.stack([p.subject_min for p in pairs], dim=0)
        subject_max = torch.stack([p.subject_max for p in pairs], dim=0)
        obstacle_min = torch.stack([p.obstacle_min for p in pairs], dim=0)
        obstacle_max = torch.stack([p.obstacle_max for p in pairs], dim=0)

        pair_loss = self._no_collision_strategy.compute_loss_batched(
            self.params.clearance_m, subject_min, subject_max, obstacle_min, obstacle_max
        )

        if debug:
            for (subject_name, obstacle_name), loss in zip(pair_names, pair_loss):
                print(f"  [NoOverlap] {subject_name} vs {obstacle_name}: loss={loss.mean().item():.6f}")

        return pair_loss.sum(dim=0)

    def _prepare_mesh_collision_cache(
        self,
        state: RelationSolverState,
        on_pairs: set[tuple[int, int]],
    ) -> None:
        """Precompute static per-pair mesh collision data."""
        device = state.device
        device_str = str(device)
        if self._mesh_manager is None or self._mesh_manager.device != device_str:
            self._mesh_manager = WarpMeshAndSphereCache(num_spheres=self.params.num_spheres, device=device_str)
        manager = self._mesh_manager

        non_anchor_objects = state.optimizable_objects
        anchor_objects = list(state.anchor_objects)

        forward_pairs, reverse_pairs = self._collect_mesh_pairs(
            state, manager, non_anchor_objects, anchor_objects, on_pairs, device
        )
        self._mesh_cache_forward = self._finalize_mesh_cache(forward_pairs, device)
        self._mesh_cache_reverse = self._finalize_mesh_cache(reverse_pairs, device)

    def _collect_mesh_pairs(
        self,
        state: RelationSolverState,
        manager: WarpMeshAndSphereCache,
        non_anchor_objects: list,
        anchor_objects: list,
        on_pairs: set[tuple[int, int]],
        device: torch.device,
    ) -> tuple[list[MeshPairEntry], list[MeshPairEntry]]:
        """Collect forward and reverse mesh pairs."""
        forward_pairs: list[MeshPairEntry] = []
        reverse_pairs: list[MeshPairEntry] = []

        for i, child in enumerate(non_anchor_objects):
            child_mesh = manager.get_collision_mesh(child)
            if child_mesh is None:
                if child.name not in self._warned_no_mesh:
                    self._warned_no_mesh.add(child.name)
                    print(f"[NoCollision] '{child.name}' has no collision mesh; pair will use AABB fallback.")
                continue
            child_spheres = manager.get_query_spheres(child_mesh, obj=child).to(device)
            child_centers_local = child_spheres[:, :3]
            child_radii = child_spheres[:, 3]
            child_bbox = state.get_bbox(child)
            c_bbox_min = child_bbox.min_point.to(device).expand(state.batch_size, 3)
            c_bbox_max = child_bbox.max_point.to(device).expand(state.batch_size, 3)

            # Forward: child's spheres → anchor's mesh
            for anchor in anchor_objects:
                if (id(child), id(anchor)) in on_pairs:
                    continue
                anchor_mesh = manager.get_collision_mesh(anchor)
                if anchor_mesh is None:
                    if anchor.name not in self._warned_no_mesh:
                        self._warned_no_mesh.add(anchor.name)
                        print(f"[NoCollision] '{anchor.name}' has no collision mesh; pair will use AABB fallback.")
                    continue
                pose = anchor.get_initial_pose()
                assert pose is not None and isinstance(
                    pose, Pose
                ), f"MESH collision requires anchor '{anchor.name}' to have a fixed Pose initial_pose"
                assert abs(pose.rotation_xyzw[0]) < 1e-6 and abs(pose.rotation_xyzw[1]) < 1e-6, (
                    f"MESH collision requires anchor '{anchor.name}' to have identity or "
                    f"pure-Z rotation, got rotation_xyzw={pose.rotation_xyzw}. "
                    "Roll/pitch anchors are not supported in MESH mode."
                )
                anchor_bbox = state.get_bbox(anchor)
                forward_pairs.append(
                    MeshPairEntry(
                        subject=child,
                        obstacle=anchor,
                        is_anchor=True,
                        anchor_pos=torch.tensor(pose.position_xyz, dtype=torch.float32, device=device),
                        anchor_yaw=yaw_from_quat_xyzw(pose.rotation_xyzw),
                        centers_local=child_centers_local,
                        radii=child_radii,
                        subject_bbox_min=c_bbox_min,
                        subject_bbox_max=c_bbox_max,
                        obstacle_bbox_min=anchor_bbox.min_point.to(device).expand(state.batch_size, 3),
                        obstacle_bbox_max=anchor_bbox.max_point.to(device).expand(state.batch_size, 3),
                        warp_mesh=manager.get_warp_mesh(anchor_mesh, obj=anchor),
                    )
                )

            # Forward + Reverse: non-anchor pairs (bidirectional gradient)
            for j in range(i + 1, len(non_anchor_objects)):
                other = non_anchor_objects[j]
                if (id(child), id(other)) in on_pairs:
                    continue
                other_mesh = manager.get_collision_mesh(other)
                if other_mesh is None:
                    if other.name not in self._warned_no_mesh:
                        self._warned_no_mesh.add(other.name)
                        print(f"[NoCollision] '{other.name}' has no collision mesh; pair will use AABB fallback.")
                    continue
                other_bbox = state.get_bbox(other)
                o_bbox_min = other_bbox.min_point.to(device).expand(state.batch_size, 3)
                o_bbox_max = other_bbox.max_point.to(device).expand(state.batch_size, 3)

                # forward: child's spheres → other's mesh
                forward_pairs.append(
                    MeshPairEntry(
                        subject=child,
                        obstacle=other,
                        is_anchor=False,
                        anchor_pos=None,
                        anchor_yaw=0.0,
                        centers_local=child_centers_local,
                        radii=child_radii,
                        subject_bbox_min=c_bbox_min,
                        subject_bbox_max=c_bbox_max,
                        obstacle_bbox_min=o_bbox_min,
                        obstacle_bbox_max=o_bbox_max,
                        warp_mesh=manager.get_warp_mesh(other_mesh, obj=other),
                    )
                )

                # reverse: other's spheres → child's mesh
                other_spheres = manager.get_query_spheres(other_mesh, obj=other).to(device)
                reverse_pairs.append(
                    MeshPairEntry(
                        subject=other,
                        obstacle=child,
                        is_anchor=False,
                        anchor_pos=None,
                        anchor_yaw=0.0,
                        centers_local=other_spheres[:, :3],
                        radii=other_spheres[:, 3],
                        subject_bbox_min=o_bbox_min,
                        subject_bbox_max=o_bbox_max,
                        obstacle_bbox_min=c_bbox_min,
                        obstacle_bbox_max=c_bbox_max,
                        warp_mesh=manager.get_warp_mesh(child_mesh, obj=child),
                    )
                )

        return forward_pairs, reverse_pairs

    @staticmethod
    def _finalize_mesh_cache(entries: list[MeshPairEntry], device: torch.device) -> MeshPairCache | None:
        """Stack collected pair entries into a MeshPairCache; None when no pairs qualify."""
        if not entries:
            return None

        mesh_id_map: dict[int, int] = {}
        mesh_id_values: list[int] = []
        mesh_idx_per_sphere: list[int] = []
        pair_slices: list[tuple[int, int]] = []
        offset = 0

        for entry in entries:
            n_spheres = entry.centers_local.shape[0]
            mesh_key = id(entry.warp_mesh)
            if mesh_key not in mesh_id_map:
                mesh_id_map[mesh_key] = len(mesh_id_values)
                mesh_id_values.append(entry.warp_mesh.id)
            mesh_idx_per_sphere.extend([mesh_id_map[mesh_key]] * n_spheres)
            pair_slices.append((offset, offset + n_spheres))
            offset += n_spheres

        pair_sphere_count = torch.tensor([e - s for s, e in pair_slices], dtype=torch.float32, device=device)
        sphere_pair_id = torch.repeat_interleave(
            torch.arange(len(pair_slices), device=device), pair_sphere_count.long()
        )

        return MeshPairCache(
            all_centers_local=torch.cat([e.centers_local for e in entries], dim=0),
            all_radii=torch.cat([e.radii for e in entries], dim=0),
            pair_subject_objs=[e.subject for e in entries],
            pair_obstacle_objs=[e.obstacle for e in entries],
            pair_is_anchor=[e.is_anchor for e in entries],
            pair_anchor_pos=[e.anchor_pos for e in entries],
            pair_anchor_yaw=[e.anchor_yaw for e in entries],
            pair_subject_bbox_min=torch.stack([e.subject_bbox_min for e in entries]),
            pair_subject_bbox_max=torch.stack([e.subject_bbox_max for e in entries]),
            pair_obstacle_bbox_min=torch.stack([e.obstacle_bbox_min for e in entries]),
            pair_obstacle_bbox_max=torch.stack([e.obstacle_bbox_max for e in entries]),
            pair_max_radius=torch.tensor([e.radii.max().item() for e in entries], device=device),
            sphere_pair_id=sphere_pair_id,
            sphere_mesh_idx=torch.tensor(mesh_idx_per_sphere, dtype=torch.int32, device=device),
            pair_sphere_count=pair_sphere_count,
            mesh_id_array=wp.array(np.array(mesh_id_values, dtype=np.uint64), dtype=wp.uint64, device=str(device)),
            num_pairs=len(entries),
            total_spheres=offset,
        )

    def _compute_no_overlap_loss_mesh(
        self,
        state: RelationSolverState,
        debug: bool,
    ) -> torch.Tensor:
        """Per-env sphere-to-SDF penetration loss."""
        device = state.device
        total_loss = torch.zeros(state.batch_size, device=device, dtype=torch.float32)
        clearance_m = self.params.clearance_m
        slope = self._no_collision_strategy.slope

        # Per-env loop (not batched like AABB): per-env yaw and active-pair masking produce a different sphere subset per env.
        for b in range(state.batch_size):
            for cache in (self._mesh_cache_forward, self._mesh_cache_reverse):
                if cache is None:
                    continue

                num_pairs = cache.num_pairs

                subject_positions = torch.stack(
                    [state.get_position(cache.pair_subject_objs[p])[b] for p in range(num_pairs)]
                )
                obstacle_positions = torch.stack([
                    (
                        cache.pair_anchor_pos[p]
                        if cache.pair_is_anchor[p]
                        else state.get_position(cache.pair_obstacle_objs[p])[b].detach()
                    )
                    for p in range(num_pairs)
                ])

                anchor_yaws = cache.pair_anchor_yaw
                has_any_yaw = self._mesh_orientations is not None or any(y != 0.0 for y in anchor_yaws)
                if has_any_yaw:
                    ori_b = self._mesh_orientations[b] if self._mesh_orientations is not None else {}
                    subject_yaws = torch.tensor(
                        [ori_b.get(cache.pair_subject_objs[p], 0.0) for p in range(num_pairs)],
                        dtype=torch.float32,
                        device=device,
                    )
                    obstacle_yaws = torch.tensor(
                        [ori_b.get(cache.pair_obstacle_objs[p], anchor_yaws[p]) for p in range(num_pairs)],
                        dtype=torch.float32,
                        device=device,
                    )

                # AABB overlap filter (yaw-aware): skip separated pairs.
                margins = cache.pair_max_radius + clearance_m
                s_bbox_min = cache.pair_subject_bbox_min[:, b, :]
                s_bbox_max = cache.pair_subject_bbox_max[:, b, :]
                o_bbox_min = cache.pair_obstacle_bbox_min[:, b, :]
                o_bbox_max = cache.pair_obstacle_bbox_max[:, b, :]

                if has_any_yaw:
                    s_bbox_min, s_bbox_max = self._rotate_bbox_extents(s_bbox_min, s_bbox_max, subject_yaws)
                    o_bbox_min, o_bbox_max = self._rotate_bbox_extents(o_bbox_min, o_bbox_max, obstacle_yaws)

                subject_min = subject_positions + s_bbox_min
                subject_max = subject_positions + s_bbox_max
                obstacle_min = obstacle_positions + o_bbox_min
                obstacle_max = obstacle_positions + o_bbox_max

                sep_subject = (subject_min - margins.unsqueeze(1)) > obstacle_max
                sep_obstacle = (obstacle_min - margins.unsqueeze(1)) > subject_max
                separated = sep_subject.any(dim=1) | sep_obstacle.any(dim=1)
                active_pair = ~separated

                if not active_pair.any():
                    continue

                offsets = subject_positions - obstacle_positions
                sphere_active_mask = active_pair[cache.sphere_pair_id]
                active_idx = sphere_active_mask.nonzero(as_tuple=True)[0]

                active_sphere_pair_id = cache.sphere_pair_id[active_idx]
                local_centers = cache.all_centers_local[active_idx]

                # R(subject_yaw - obstacle_yaw) · local + R(-obstacle_yaw) · offset
                if has_any_yaw:
                    net_yaws = (subject_yaws - obstacle_yaws)[active_sphere_pair_id]
                    half_net = net_yaws / 2.0
                    q_net_z = torch.zeros(len(half_net), 4, device=device, dtype=local_centers.dtype)
                    q_net_z[:, 2] = torch.sin(half_net)
                    q_net_z[:, 3] = torch.cos(half_net)
                    local_centers = quat_apply(q_net_z, local_centers)

                    pair_offsets = offsets[active_sphere_pair_id]
                    obs_yaws = obstacle_yaws[active_sphere_pair_id]
                    half_o = obs_yaws / 2.0
                    q_obstacle_z = torch.zeros(len(half_o), 4, device=device, dtype=local_centers.dtype)
                    q_obstacle_z[:, 2] = torch.sin(half_o)
                    q_obstacle_z[:, 3] = torch.cos(half_o)
                    rotated_offsets = quat_apply_inverse(q_obstacle_z, pair_offsets)
                    active_centers = local_centers + rotated_offsets
                else:
                    active_centers = local_centers + offsets[active_sphere_pair_id]
                active_radii = cache.all_radii[active_idx]
                active_mesh_idx = cache.sphere_mesh_idx[active_idx].contiguous()

                active_mesh_indices_wp = wp.from_torch(active_mesh_idx, dtype=wp.int32)
                sdf_values = multi_mesh_sdf(active_centers, cache.mesh_id_array, active_mesh_indices_wp)
                self._mesh_manager.warn_sdf_sentinel(sdf_values)
                sdf_values = clamp_sdf_sentinel(sdf_values)
                penetration = torch.relu(active_radii + clearance_m - sdf_values)

                pair_sum = torch.zeros(num_pairs, device=device, dtype=penetration.dtype)
                pair_sum.index_add_(0, active_sphere_pair_id, penetration)
                pair_mean = pair_sum / cache.pair_sphere_count
                active_pair_idx = active_pair.nonzero(as_tuple=True)[0]
                total_loss[b] = total_loss[b] + slope * pair_mean[active_pair_idx].sum()

        if debug:
            print(f"  [NoOverlap MESH] total_loss={total_loss.tolist()}")

        return total_loss

    @staticmethod
    def _rotate_bbox_extents(
        bbox_min: torch.Tensor, bbox_max: torch.Tensor, yaws: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the AABB enclosing a Z-rotated bbox (bbox_min/max: (N,3), yaws: (N,))."""
        cos_y = torch.cos(yaws).abs().unsqueeze(1)
        sin_y = torch.sin(yaws).abs().unsqueeze(1)
        half = (bbox_max - bbox_min) / 2.0
        center = (bbox_max + bbox_min) / 2.0
        new_hx = half[:, 0:1] * cos_y + half[:, 1:2] * sin_y
        new_hy = half[:, 0:1] * sin_y + half[:, 1:2] * cos_y
        new_half = torch.cat([new_hx, new_hy, half[:, 2:3]], dim=1)
        return center - new_half, center + new_half

    def solve(
        self,
        objects: list[ObjectBase],
        initial_positions: list[dict[ObjectBase, tuple[float, float, float]]],
        env_bboxes: dict[ObjectBase, AxisAlignedBoundingBox] | None = None,
        orientations: list[dict[ObjectBase, float]] | None = None,
    ) -> list[dict[ObjectBase, tuple[float, float, float]]]:
        """Solve for optimal positions of all objects.

        Args:
            objects: List of ObjectBase instances. Must include at least one object
                marked with IsAnchor() which serves as a fixed reference.
            initial_positions: List of dicts (one per env). Use a single-element list
                for single-env placement.
            env_bboxes: Optional per-env bounding boxes keyed by object.
                ObjectPlacer always supplies these, with each
                AxisAlignedBoundingBox shaped (batch, 3). Direct solver calls
                may omit them to use each object's default get_bounding_box().
            orientations: Optional per-env yaw angles (radians about Z) per object.
                Used in MESH mode to rotate sphere centers before collision queries.

        Returns:
            List of dicts (one per env) mapping objects to their solved (x, y, z) positions.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state = RelationSolverState(objects, initial_positions, device=device, env_bboxes=env_bboxes)

        if self.params.verbose:
            anchor_names = [obj.name for obj in state.anchor_objects]
            optimizable_names = [obj.name for obj in state.optimizable_objects]
            print("=== RelationSolver ===")
            print(f"Anchors (fixed): {anchor_names}")
            print(f"Optimizable: {optimizable_names}")

        # Early return if nothing to optimize (all objects are anchors)
        if len(state.optimizable_objects) == 0:
            if self.params.verbose:
                print("No optimizable objects, skipping solver.")
            self._last_loss_history = [0.0]
            self._last_loss_per_env = torch.zeros(state.batch_size)
            self._last_position_history = [state.get_all_positions_snapshot()]
            return state.get_final_positions()

        if self.params.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_start = time.perf_counter()

        # Precompute mesh collision cache (once per solve, before opt loop)
        if self.params.collision_mode == CollisionMode.MESH:
            non_anchor_objects = state.optimizable_objects
            anchor_objects = list(state.anchor_objects)
            on_pairs: set[tuple[int, int]] = set()
            for obj in [*non_anchor_objects, *anchor_objects]:
                for rel in obj.get_relations():
                    if isinstance(rel, On):
                        on_pairs.add((id(obj), id(rel.parent)))
                        on_pairs.add((id(rel.parent), id(obj)))
            self._mesh_orientations = orientations
            self._prepare_mesh_collision_cache(state, on_pairs)
            self._mesh_manager.reset_sentinel_warning()

        # Setup optimizer (only for optimizable positions)
        optimizer = torch.optim.Adam([state.optimizable_positions], lr=self.params.lr)

        # Compute initial loss so _last_loss_per_env is always populated, even when max_iters=0.
        with torch.no_grad():
            self._compute_total_loss(state)

        # Optimization loop
        loss_history = []
        position_history = []  # Track positions for visualization

        for iter in range(self.params.max_iters):
            optimizer.zero_grad()

            if self.params.save_position_history and iter % self.POSITION_HISTORY_SAVE_INTERVAL == 0:
                position_history.append(state.get_all_positions_snapshot())

            loss = self._compute_total_loss(state)
            loss_history.append(loss.item())

            # Constant-zero loss has no grad_fn — skip backward when overlap filter culls all pairs.
            if loss.grad_fn is not None:
                loss.backward()
                optimizer.step()

            if self.params.verbose and iter % 100 == 0:
                print(f"Iter {iter}: loss = {loss.item():.6f}")

            # Check convergence
            if loss.item() < self.params.convergence_threshold:
                if self.params.verbose:
                    print(f"Converged at iteration {iter}")
                break

        if self.params.profile and torch.cuda.is_available():
            torch.cuda.synchronize()
        solve_elapsed_ms = (time.perf_counter() - solve_start) * 1e3

        if self.params.save_position_history:
            position_history.append(state.get_all_positions_snapshot())

        if self.params.verbose and loss_history:
            print(f"\nFinal loss: {loss_history[-1]:.6f}")
            print(f"Total iterations: {len(loss_history)}")

        if self.params.profile and loss_history:
            iters_run = len(loss_history)
            print(
                f"[RelationSolver] solve: {solve_elapsed_ms:.1f} ms"
                f" | batch={state.batch_size}"
                f" | objects={len(state.optimizable_objects)} optimizable + {len(state.anchor_objects)} anchors"
                f" | no-overlap pairs={self._last_no_overlap_pair_count}"
                f" | iters={iters_run} ({solve_elapsed_ms / iters_run:.2f} ms/iter)"
            )

        self._last_loss_history = loss_history
        self._last_position_history = position_history

        return state.get_final_positions()

    @property
    def last_loss_history(self) -> list[float]:
        """Loss values from the most recent solve() call."""
        return self._last_loss_history

    @property
    def last_loss_per_env(self) -> torch.Tensor | None:
        """Per-candidate loss tensor of shape (batch_size,) from the last solve() call."""
        return self._last_loss_per_env

    @property
    def last_position_history(self) -> list:
        """Position snapshots from the most recent solve() call."""
        return self._last_position_history

    def debug_losses(self, objects: list[ObjectBase]) -> None:
        """Print detailed loss breakdown for all relations using final positions.

        Call this after solve() to inspect why objects may not be correctly positioned.

        Args:
            objects: The same list of objects passed to solve().
        """
        print("\n" + "=" * 60)
        print("DEBUG: Final Loss Breakdown")
        print("=" * 60)

        final_positions_list = self.last_position_history[-1] if self.last_position_history else None
        if final_positions_list is None:
            print("No position history available. Run solve() first.")
            return

        final_positions = {obj: (pos[0], pos[1], pos[2]) for obj, pos in zip(objects, final_positions_list)}

        state = RelationSolverState(objects, [final_positions])
        self._compute_total_loss(state, debug=True)
        print("\n" + "=" * 60)


def _print_relation_debug(
    obj: ObjectBase,
    relation: Relation,
    child_pos: torch.Tensor,
    parent_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a single binary relation."""
    child_bbox = obj.get_bounding_box()
    parent_world_bbox = relation.parent.get_world_bounding_box()

    print(f"\n=== {obj.name} -> {type(relation).__name__}({relation.parent.name}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(
        f"  Child bbox: min={child_bbox.min_point[0].tolist()}, max={child_bbox.max_point[0].tolist()},"
        f" size={child_bbox.size[0].tolist()}"
    )
    print(f"  Parent pos: ({parent_pos[0].item():.4f}, {parent_pos[1].item():.4f}, {parent_pos[2].item():.4f})")
    print(
        f"  Parent world bbox: min={parent_world_bbox.min_point[0].tolist()},"
        f" max={parent_world_bbox.max_point[0].tolist()}, size={parent_world_bbox.size[0].tolist()}"
    )

    # Child world extents
    child_x_range = (
        child_pos[0].item() + child_bbox.min_point[0, 0].item(),
        child_pos[0].item() + child_bbox.max_point[0, 0].item(),
    )
    child_y_range = (
        child_pos[1].item() + child_bbox.min_point[0, 1].item(),
        child_pos[1].item() + child_bbox.max_point[0, 1].item(),
    )

    print(f"  Child world X: [{child_x_range[0]:.4f}, {child_x_range[1]:.4f}]")
    print(f"  Child world Y: [{child_y_range[0]:.4f}, {child_y_range[1]:.4f}]")
    print(
        f"  Parent world X: [{parent_world_bbox.min_point[0, 0].item():.4f},"
        f" {parent_world_bbox.max_point[0, 0].item():.4f}]"
    )
    print(
        f"  Parent world Y: [{parent_world_bbox.min_point[0, 1].item():.4f},"
        f" {parent_world_bbox.max_point[0, 1].item():.4f}]"
    )
    print(f"  Loss: {loss.item():.6f}")


def _print_unary_relation_debug(
    obj: ObjectBase,
    relation: RelationBase,
    child_pos: torch.Tensor,
    loss: torch.Tensor,
) -> None:
    """Print debug information for a unary relation (no parent)."""
    child_bbox = obj.get_bounding_box()

    params = {k: v for k, v in relation.__dict__.items() if v is not None and k != "relation_loss_weight"}
    param_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items())
    print(f"\n=== {obj.name} -> {type(relation).__name__}({param_str}) ===")
    print(f"  Child pos: ({child_pos[0].item():.4f}, {child_pos[1].item():.4f}, {child_pos[2].item():.4f})")
    print(
        f"  Child bbox: min={child_bbox.min_point[0].tolist()}, max={child_bbox.max_point[0].tolist()},"
        f" size={child_bbox.size[0].tolist()}"
    )
    print(f"  Loss: {loss.item():.6f}")
