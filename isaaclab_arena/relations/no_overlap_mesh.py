# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-based no-overlap collision loss computation."""

from __future__ import annotations

import numpy as np
import torch
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.relations.mesh_pair_cache import MeshPairCache, MeshPairEntry
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
from isaaclab_arena.relations.warp_sdf_kernels import clamp_sdf_sentinel, multi_mesh_sdf
from isaaclab_arena.utils.pose import Pose, rotate_points_by_yaw_batch, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def compute_no_overlap_loss_mesh(
    state: RelationSolverState,
    mesh_cache: MeshPairCache | None,
    mesh_manager: WarpMeshAndSphereCache,
    orientations: list[dict[ObjectBase, float]] | None,
    clearance_m: float,
    slope: float,
    debug: bool,
) -> torch.Tensor:
    """Per-env sphere-to-SDF penetration loss.

    Args:
        state: Current solver state with positions and batch info.
        mesh_cache: Precomputed collision pair data (None = no pairs).
        mesh_manager: Warp mesh/sphere cache (for sentinel warnings).
        orientations: Per-env yaw angles per object.
        clearance_m: Minimum clearance between objects.
        slope: Gradient magnitude for overlap loss.
        debug: Print per-pair loss when True.
    """
    device = state.device
    total_loss = torch.zeros(state.batch_size, device=device, dtype=torch.float32)

    if mesh_cache is None:
        return total_loss

    num_pairs = mesh_cache.num_pairs

    # Per-env loop: per-env yaw and active-pair masking produce a different sphere subset per env.
    for b in range(state.batch_size):
        subject_positions = torch.stack(
            [state.get_position(mesh_cache.pair_subject_objs[p])[b] for p in range(num_pairs)]
        )
        obstacle_positions = torch.stack([
            (
                mesh_cache.pair_anchor_pos[p]
                if mesh_cache.pair_is_anchor[p]
                else state.get_position(mesh_cache.pair_obstacle_objs[p])[b].detach()
            )
            for p in range(num_pairs)
        ])

        anchor_yaws = mesh_cache.pair_anchor_yaw
        has_any_yaw = orientations is not None or any(y != 0.0 for y in anchor_yaws)
        if has_any_yaw:
            ori_b = orientations[b] if orientations is not None else {}
            subject_yaws = torch.tensor(
                [ori_b.get(mesh_cache.pair_subject_objs[p], 0.0) for p in range(num_pairs)],
                dtype=torch.float32,
                device=device,
            )
            obstacle_yaws = torch.tensor(
                [ori_b.get(mesh_cache.pair_obstacle_objs[p], anchor_yaws[p]) for p in range(num_pairs)],
                dtype=torch.float32,
                device=device,
            )

        # AABB overlap filter (yaw-aware): skip separated pairs.
        margins = mesh_cache.pair_max_radius + clearance_m
        s_bbox_min = mesh_cache.pair_subject_bbox_min[:, b, :]
        s_bbox_max = mesh_cache.pair_subject_bbox_max[:, b, :]
        o_bbox_min = mesh_cache.pair_obstacle_bbox_min[:, b, :]
        o_bbox_max = mesh_cache.pair_obstacle_bbox_max[:, b, :]

        if has_any_yaw:
            s_bbox_min, s_bbox_max = _rotate_bbox_extents(s_bbox_min, s_bbox_max, subject_yaws)
            o_bbox_min, o_bbox_max = _rotate_bbox_extents(o_bbox_min, o_bbox_max, obstacle_yaws)

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
        sphere_active_mask = active_pair[mesh_cache.sphere_pair_id]
        active_idx = sphere_active_mask.nonzero(as_tuple=True)[0]

        active_sphere_pair_id = mesh_cache.sphere_pair_id[active_idx]
        local_centers = mesh_cache.all_centers_local[active_idx]

        # R(subject_yaw - obstacle_yaw) · local + R(-obstacle_yaw) · offset
        if has_any_yaw:
            net_yaws = (subject_yaws - obstacle_yaws)[active_sphere_pair_id]
            local_centers = rotate_points_by_yaw_batch(local_centers, net_yaws)

            pair_offsets = offsets[active_sphere_pair_id]
            obs_yaws = obstacle_yaws[active_sphere_pair_id]
            rotated_offsets = rotate_points_by_yaw_batch(pair_offsets, -obs_yaws)
            active_centers = local_centers + rotated_offsets
        else:
            active_centers = local_centers + offsets[active_sphere_pair_id]
        active_radii = mesh_cache.all_radii[active_idx]
        active_mesh_idx = mesh_cache.sphere_mesh_idx[active_idx].contiguous()

        active_mesh_indices_wp = wp.from_torch(active_mesh_idx, dtype=wp.int32)
        sdf_values = multi_mesh_sdf(active_centers, mesh_cache.mesh_id_array, active_mesh_indices_wp)
        mesh_manager.warn_sdf_sentinel(sdf_values)
        sdf_values = clamp_sdf_sentinel(sdf_values)
        penetration = torch.relu(active_radii + clearance_m - sdf_values)

        pair_sum = torch.zeros(num_pairs, device=device, dtype=penetration.dtype)
        pair_sum.index_add_(0, active_sphere_pair_id, penetration)
        pair_mean = pair_sum / mesh_cache.pair_sphere_count
        active_pair_idx = active_pair.nonzero(as_tuple=True)[0]
        total_loss[b] = total_loss[b] + slope * pair_mean[active_pair_idx].sum()

    if debug:
        print(f"  [NoOverlap MESH] total_loss={total_loss.tolist()}")

    return total_loss


def prepare_mesh_collision_cache(
    state: RelationSolverState,
    mesh_manager: WarpMeshAndSphereCache,
    on_pairs: set[tuple[int, int]],
    warned_no_mesh: set[str],
) -> MeshPairCache | None:
    """Precompute static per-pair mesh collision data.

    Args:
        state: Solver state with object info and batch size.
        mesh_manager: Warp mesh/sphere cache.
        on_pairs: Set of (id(a), id(b)) pairs linked by On relations (skipped).
        warned_no_mesh: Mutable set tracking which objects have already been warned about.

    Returns:
        Combined MeshPairCache for all directed pairs, or None if no pairs qualify.
    """
    device = state.device
    non_anchor_objects = state.optimizable_objects
    anchor_objects = list(state.anchor_objects)

    all_pairs = _collect_mesh_pairs(
        state, mesh_manager, non_anchor_objects, anchor_objects, on_pairs, device, warned_no_mesh
    )
    return _finalize_mesh_cache(all_pairs, device)


def _collect_mesh_pairs(
    state: RelationSolverState,
    manager: WarpMeshAndSphereCache,
    non_anchor_objects: list,
    anchor_objects: list,
    on_pairs: set[tuple[int, int]],
    device: torch.device,
    warned_no_mesh: set[str],
) -> list[MeshPairEntry]:
    """Collect all directed mesh pairs (forward + reverse)."""
    pairs: list[MeshPairEntry] = []

    for i, child in enumerate(non_anchor_objects):
        child_mesh = manager.get_collision_mesh(child)
        if child_mesh is None:
            if child.name not in warned_no_mesh:
                warned_no_mesh.add(child.name)
                print(f"[NoCollision] '{child.name}' has no collision mesh; pair will use AABB fallback.")
            continue
        child_spheres = manager.get_query_spheres(child_mesh, obj=child).to(device)
        child_centers_local = child_spheres[:, :3]
        child_radii = child_spheres[:, 3]
        # Use unrotated bbox: _rotate_bbox_extents handles yaw in the broadphase filter.
        child_bbox = child.get_bounding_box().to(device)
        c_bbox_min = child_bbox.min_point.expand(state.batch_size, 3)
        c_bbox_max = child_bbox.max_point.expand(state.batch_size, 3)

        # child's spheres → anchor's mesh
        for anchor in anchor_objects:
            if (id(child), id(anchor)) in on_pairs:
                continue
            anchor_mesh = manager.get_collision_mesh(anchor)
            if anchor_mesh is None:
                if anchor.name not in warned_no_mesh:
                    warned_no_mesh.add(anchor.name)
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
            pairs.append(
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

        # Non-anchor pairs (bidirectional): forward + reverse
        for j in range(i + 1, len(non_anchor_objects)):
            other = non_anchor_objects[j]
            if (id(child), id(other)) in on_pairs:
                continue
            other_mesh = manager.get_collision_mesh(other)
            if other_mesh is None:
                if other.name not in warned_no_mesh:
                    warned_no_mesh.add(other.name)
                    print(f"[NoCollision] '{other.name}' has no collision mesh; pair will use AABB fallback.")
                continue
            other_bbox = other.get_bounding_box().to(device)
            o_bbox_min = other_bbox.min_point.expand(state.batch_size, 3)
            o_bbox_max = other_bbox.max_point.expand(state.batch_size, 3)

            # forward: child's spheres → other's mesh
            pairs.append(
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
            pairs.append(
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

    return pairs


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
    sphere_pair_id = torch.repeat_interleave(torch.arange(len(pair_slices), device=device), pair_sphere_count.long())

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
