# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-based no-overlap collision loss computation."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from typing import TYPE_CHECKING

import warp as wp

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.mesh_pair_cache import MeshPairCache, MeshPairEntry
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.warp_sdf_kernels import clamp_sdf_sentinel, multi_mesh_sdf
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena.utils.yaw import rotate_points_by_yaw_batch, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.placement_asset import PlaceableAsset
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
    from isaaclab_arena.utils.bounding_box import AxisAlignedBoundingBox


def compute_no_overlap_loss_mesh(
    state: RelationSolverState,
    mesh_cache: MeshPairCache | None,
    mesh_manager: WarpMeshAndSphereCache,
    orientations: list[dict[PlaceableAsset, float]] | None,
    clearance_m: float,
    slope: float,
    debug: bool,
) -> torch.Tensor:
    """Compute per-environment sphere-to-SDF penetration loss in one batched query.

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
    batch_size = state.batch_size
    total_loss = torch.zeros(batch_size, device=device, dtype=torch.float32)

    if mesh_cache is None:
        return total_loss

    num_pairs = mesh_cache.num_pairs
    num_spheres = mesh_cache.total_spheres
    sphere_pair_id = mesh_cache.sphere_pair_id.long()

    subject_positions = torch.stack(
        [state.get_position(subject) for subject in mesh_cache.pair_subject_objs],
        dim=1,
    )
    obstacle_positions = torch.stack(
        [
            (
                mesh_cache.pair_fixed_obstacle_pos_tensor[p].expand(batch_size, 3)
                if mesh_cache.pair_obstacle_is_fixed[p]
                else state.get_position(mesh_cache.pair_obstacle_objs[p]).detach()
            )
            for p in range(num_pairs)
        ],
        dim=1,
    )
    subject_yaws, obstacle_yaws, has_any_yaw = _get_batched_pair_yaws(mesh_cache, orientations, batch_size, device)

    subject_bbox_min = mesh_cache.pair_subject_bbox_min.permute(1, 0, 2)
    subject_bbox_max = mesh_cache.pair_subject_bbox_max.permute(1, 0, 2)
    obstacle_bbox_min = mesh_cache.pair_obstacle_bbox_min.permute(1, 0, 2)
    obstacle_bbox_max = mesh_cache.pair_obstacle_bbox_max.permute(1, 0, 2)
    if has_any_yaw:
        subject_bbox_yaws = torch.where(
            mesh_cache.pair_subject_bbox_includes_yaw_tensor.unsqueeze(0),
            0.0,
            subject_yaws,
        )
        obstacle_bbox_yaws = torch.where(
            mesh_cache.pair_obstacle_bbox_includes_yaw_tensor.unsqueeze(0),
            0.0,
            obstacle_yaws,
        )
        subject_bbox_min, subject_bbox_max = _rotate_bbox_extents_batched(
            subject_bbox_min, subject_bbox_max, subject_bbox_yaws
        )
        obstacle_bbox_min, obstacle_bbox_max = _rotate_bbox_extents_batched(
            obstacle_bbox_min, obstacle_bbox_max, obstacle_bbox_yaws
        )

    margins = mesh_cache.pair_max_radius.view(1, num_pairs, 1) + clearance_m
    subject_min = subject_positions + subject_bbox_min
    subject_max = subject_positions + subject_bbox_max
    obstacle_min = obstacle_positions + obstacle_bbox_min
    obstacle_max = obstacle_positions + obstacle_bbox_max
    separated = ((subject_min - margins) > obstacle_max).any(dim=2) | ((obstacle_min - margins) > subject_max).any(
        dim=2
    )
    active_pair = ~separated

    offsets = subject_positions - obstacle_positions
    local_centers = mesh_cache.all_centers_local.unsqueeze(0).expand(batch_size, num_spheres, 3)
    sphere_offsets = offsets[:, sphere_pair_id, :]
    if has_any_yaw:
        subject_sphere_yaws = subject_yaws[:, sphere_pair_id]
        obstacle_sphere_yaws = obstacle_yaws[:, sphere_pair_id]
        local_centers = rotate_points_by_yaw_batch(
            local_centers.reshape(-1, 3),
            (subject_sphere_yaws - obstacle_sphere_yaws).reshape(-1),
        ).reshape(batch_size, num_spheres, 3)
        sphere_offsets = rotate_points_by_yaw_batch(
            sphere_offsets.reshape(-1, 3),
            -obstacle_sphere_yaws.reshape(-1),
        ).reshape(batch_size, num_spheres, 3)

    active_sphere = active_pair[:, sphere_pair_id]
    sdf_values = _query_component_union_sdf_batched(
        local_centers + sphere_offsets,
        active_sphere,
        mesh_cache,
        mesh_manager,
    )
    penetration = torch.relu(mesh_cache.all_radii.unsqueeze(0) + clearance_m - sdf_values)
    penetration = penetration * active_sphere

    pair_sum = torch.zeros(batch_size, num_pairs, device=device, dtype=penetration.dtype)
    pair_sum.scatter_add_(1, sphere_pair_id.unsqueeze(0).expand(batch_size, num_spheres), penetration)
    total_loss = slope * (pair_sum / mesh_cache.pair_sphere_count.unsqueeze(0)).sum(dim=1)

    if debug:
        print(f"  [NoOverlap MESH] total_loss={total_loss.tolist()}")

    return total_loss


def _query_component_union_sdf_batched(
    sphere_centers: torch.Tensor,
    active_sphere: torch.Tensor,
    mesh_cache: MeshPairCache,
    mesh_manager: WarpMeshAndSphereCache,
) -> torch.Tensor:
    """Query all candidate/component combinations and return their union SDF.

    Keeping overlapping closed components separate avoids the ambiguous closest-face sign produced
    by concatenating them into one topologically watertight mesh. Their pointwise minimum is the SDF
    of the component union.
    """
    batch_size, num_spheres, _ = sphere_centers.shape
    num_queries = len(mesh_cache.query_sphere_id)
    query_centers = sphere_centers[:, mesh_cache.query_sphere_id, :].reshape(batch_size * num_queries, 3)
    query_mesh_idx = mesh_cache.query_mesh_idx.unsqueeze(0).expand(batch_size, num_queries).reshape(-1).contiguous()
    query_active = active_sphere[:, mesh_cache.query_sphere_id].reshape(-1)
    query_sdf = multi_mesh_sdf(
        query_centers.contiguous(),
        mesh_cache.mesh_id_array,
        wp.from_torch(query_mesh_idx, dtype=wp.int32),
        query_active,
    )
    mesh_manager.warn_sdf_sentinel(query_sdf)
    query_sdf = clamp_sdf_sentinel(query_sdf)

    candidate_offsets = torch.arange(batch_size, device=sphere_centers.device).unsqueeze(1) * num_spheres
    flat_sphere_id = (mesh_cache.query_sphere_id.unsqueeze(0) + candidate_offsets).reshape(-1)
    flat_union_sdf = torch.full(
        (batch_size * num_spheres,),
        torch.inf,
        dtype=query_sdf.dtype,
        device=sphere_centers.device,
    )
    flat_union_sdf.scatter_reduce_(0, flat_sphere_id, query_sdf, reduce="amin", include_self=True)
    return flat_union_sdf.reshape(batch_size, num_spheres)


def _get_batched_pair_yaws(
    mesh_cache: MeshPairCache,
    orientations: list[dict[PlaceableAsset, float]] | None,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return subject and obstacle yaw tables shaped (B, P)."""
    num_pairs = mesh_cache.num_pairs
    has_any_yaw = orientations is not None or any(yaw != 0.0 for yaw in mesh_cache.pair_fixed_obstacle_yaw)
    if not has_any_yaw:
        zeros = torch.zeros(batch_size, num_pairs, dtype=torch.float32, device=device)
        return zeros, zeros, False

    if orientations is None:
        subject_yaws = torch.zeros(batch_size, num_pairs, dtype=torch.float32, device=device)
        obstacle_yaws = mesh_cache.pair_fixed_obstacle_yaw_tensor.unsqueeze(0).expand(batch_size, num_pairs)
        return subject_yaws, obstacle_yaws, True

    assert len(orientations) == batch_size, "Orientations must provide one mapping per candidate."
    subject_yaws = torch.tensor(
        [
            [
                orientation.get(mesh_cache.pair_subject_objs[p], 0.0) if mesh_cache.pair_subject_applies_yaw[p] else 0.0
                for p in range(num_pairs)
            ]
            for orientation in orientations
        ],
        dtype=torch.float32,
        device=device,
    )
    obstacle_yaws = torch.tensor(
        [
            [
                orientation.get(mesh_cache.pair_obstacle_objs[p], mesh_cache.pair_fixed_obstacle_yaw[p])
                for p in range(num_pairs)
            ]
            for orientation in orientations
        ],
        dtype=torch.float32,
        device=device,
    )
    return subject_yaws, obstacle_yaws, True


def prepare_mesh_collision_cache(
    state: RelationSolverState,
    mesh_manager: WarpMeshAndSphereCache,
    on_pairs: set[tuple[int, int]],
    warned_no_mesh: set[str],
    default_collision_mode: CollisionMode = CollisionMode.MESH,
    bboxes_include_yaw: bool = False,
) -> MeshPairCache | None:
    """Precompute static per-pair mesh collision data.

    Args:
        state: Solver state with object info and batch size.
        mesh_manager: Warp mesh/sphere cache.
        on_pairs: Set of (id(a), id(b)) pairs linked by On relations (skipped).
        warned_no_mesh: Mutable set tracking which objects have already been warned about.
        default_collision_mode: Collision mode used by objects without a per-object override.
        bboxes_include_yaw: True when state bboxes are already yaw-expanded.

    Returns:
        Combined MeshPairCache for all directed pairs, or None if no pairs qualify.
    """
    device = state.device
    non_anchor_objects = state.optimizable_objects
    anchor_objects = list(state.anchor_objects)
    fixed_obstacles = anchor_objects + list(state.collision_objects)

    all_pairs = _collect_mesh_pairs(
        state,
        mesh_manager,
        non_anchor_objects,
        fixed_obstacles,
        on_pairs,
        device,
        warned_no_mesh,
        default_collision_mode,
        bboxes_include_yaw,
    )
    return _finalize_mesh_cache(all_pairs, device)


def _collect_mesh_pairs(
    state: RelationSolverState,
    manager: WarpMeshAndSphereCache,
    non_anchor_objects: list,
    fixed_obstacles: list[PlaceableAsset | CollisionObject],
    on_pairs: set[tuple[int, int]],
    device: torch.device,
    warned_no_mesh: set[str],
    default_collision_mode: CollisionMode,
    bboxes_include_yaw: bool,
) -> list[MeshPairEntry]:
    """Collect all directed mesh pairs (forward + reverse)."""
    pairs: list[MeshPairEntry] = []

    for i, child in enumerate(non_anchor_objects):
        child_uses_mesh = object_uses_mesh_collision(child, default_collision_mode)
        child_meshes = manager.get_collision_meshes(child) if child_uses_mesh else ()
        child_bbox = state.get_bbox(child).to(device)
        child_bbox_is_invariant = child_bbox.is_batch_invariant()
        if child_uses_mesh and not child_meshes and child.name not in warned_no_mesh:
            warned_no_mesh.add(child.name)
            fallback = (
                "using an AABB-sphere approximation for mesh-obstacle pairs"
                if child_bbox_is_invariant
                else "pair will use AABB fallback for varying per-env bboxes"
            )
            print(f"[NoCollision] '{child.name}' has no collision mesh; {fallback}.")
        child_spheres = _get_subject_spheres(child_meshes, child_bbox, child, manager, device)
        child_applies_yaw = bool(child_meshes) or not bboxes_include_yaw
        c_bbox_min, c_bbox_max = _collision_bounds(child_meshes, child_bbox, state.batch_size, device)
        child_bbox_includes_yaw = bboxes_include_yaw and not child_meshes

        # child's spheres → fixed obstacle mesh (anchors plus passive background)
        for obstacle in fixed_obstacles:
            if (id(child), id(obstacle)) in on_pairs:
                continue
            obstacle_meshes = (
                manager.get_collision_meshes(obstacle)
                if object_uses_mesh_collision(obstacle, default_collision_mode)
                else ()
            )
            if not obstacle_meshes:
                if object_uses_mesh_collision(obstacle, default_collision_mode) and obstacle.name not in warned_no_mesh:
                    warned_no_mesh.add(obstacle.name)
                    print(f"[NoCollision] '{obstacle.name}' has no collision mesh; pair will use AABB fallback.")
                continue
            pose = obstacle.get_initial_pose()
            assert pose is not None and isinstance(
                pose, Pose
            ), f"MESH collision requires fixed obstacle '{obstacle.name}' to have a fixed Pose initial_pose"
            assert abs(pose.rotation_xyzw[0]) < 1e-6 and abs(pose.rotation_xyzw[1]) < 1e-6, (
                f"MESH collision requires fixed obstacle '{obstacle.name}' to have identity or "
                f"pure-Z rotation, got rotation_xyzw={pose.rotation_xyzw}. "
                "Roll/pitch fixed obstacles are not supported in MESH mode."
            )
            if child_spheres is None:
                continue
            obstacle_bbox = obstacle.get_bounding_box().to(device)
            obstacle_bbox_min, obstacle_bbox_max = _collision_bounds(
                obstacle_meshes, obstacle_bbox, state.batch_size, device
            )
            pairs.append(
                MeshPairEntry(
                    subject=child,
                    obstacle=obstacle,
                    obstacle_is_fixed=True,
                    fixed_obstacle_pos=torch.tensor(pose.position_xyz, dtype=torch.float32, device=device),
                    fixed_obstacle_yaw=yaw_from_quat_xyzw(pose.rotation_xyzw),
                    centers_local=child_spheres[:, :3],
                    subject_applies_yaw=child_applies_yaw,
                    radii=child_spheres[:, 3],
                    subject_bbox_min=c_bbox_min,
                    subject_bbox_max=c_bbox_max,
                    subject_bbox_includes_yaw=child_bbox_includes_yaw,
                    obstacle_bbox_min=obstacle_bbox_min,
                    obstacle_bbox_max=obstacle_bbox_max,
                    obstacle_bbox_includes_yaw=False,
                    warp_meshes=tuple(manager.get_warp_mesh(mesh, obj=obstacle) for mesh in obstacle_meshes),
                )
            )

        # Non-anchor pairs (bidirectional): forward + reverse
        for j in range(i + 1, len(non_anchor_objects)):
            other = non_anchor_objects[j]
            if (id(child), id(other)) in on_pairs:
                continue
            other_uses_mesh = object_uses_mesh_collision(other, default_collision_mode)
            other_meshes = manager.get_collision_meshes(other) if other_uses_mesh else ()
            other_bbox = state.get_bbox(other).to(device)
            other_bbox_is_invariant = other_bbox.is_batch_invariant()
            if not other_meshes and not child_meshes:
                if other_uses_mesh and other.name not in warned_no_mesh:
                    warned_no_mesh.add(other.name)
                    fallback = (
                        "using an AABB-sphere approximation for mesh-obstacle pairs"
                        if other_bbox_is_invariant
                        else "pair will use AABB fallback for varying per-env bboxes"
                    )
                    print(f"[NoCollision] '{other.name}' has no collision mesh; {fallback}.")
                continue
            o_bbox_min, o_bbox_max = _collision_bounds(other_meshes, other_bbox, state.batch_size, device)
            other_bbox_includes_yaw = bboxes_include_yaw and not other_meshes

            if other_meshes and child_spheres is not None:
                # forward: child's mesh/spheres or AABB-sphere approximation → other's mesh
                pairs.append(
                    MeshPairEntry(
                        subject=child,
                        obstacle=other,
                        obstacle_is_fixed=False,
                        fixed_obstacle_pos=None,
                        fixed_obstacle_yaw=0.0,
                        centers_local=child_spheres[:, :3],
                        subject_applies_yaw=child_applies_yaw,
                        radii=child_spheres[:, 3],
                        subject_bbox_min=c_bbox_min,
                        subject_bbox_max=c_bbox_max,
                        subject_bbox_includes_yaw=child_bbox_includes_yaw,
                        obstacle_bbox_min=o_bbox_min,
                        obstacle_bbox_max=o_bbox_max,
                        obstacle_bbox_includes_yaw=other_bbox_includes_yaw,
                        warp_meshes=tuple(manager.get_warp_mesh(mesh, obj=other) for mesh in other_meshes),
                    )
                )

            if child_meshes:
                # reverse: other's mesh/spheres or AABB-sphere approximation → child's mesh
                other_spheres = _get_subject_spheres(other_meshes, other_bbox, other, manager, device)
                if other_spheres is None:
                    continue
                other_applies_yaw = bool(other_meshes) or not bboxes_include_yaw
                pairs.append(
                    MeshPairEntry(
                        subject=other,
                        obstacle=child,
                        obstacle_is_fixed=False,
                        fixed_obstacle_pos=None,
                        fixed_obstacle_yaw=0.0,
                        centers_local=other_spheres[:, :3],
                        subject_applies_yaw=other_applies_yaw,
                        radii=other_spheres[:, 3],
                        subject_bbox_min=o_bbox_min,
                        subject_bbox_max=o_bbox_max,
                        subject_bbox_includes_yaw=other_bbox_includes_yaw,
                        obstacle_bbox_min=c_bbox_min,
                        obstacle_bbox_max=c_bbox_max,
                        obstacle_bbox_includes_yaw=child_bbox_includes_yaw,
                        warp_meshes=tuple(manager.get_warp_mesh(mesh, obj=child) for mesh in child_meshes),
                    )
                )

    return pairs


def _get_subject_spheres(
    meshes: tuple[trimesh.Trimesh, ...],
    bbox: AxisAlignedBoundingBox,
    obj: PlaceableAsset,
    manager: WarpMeshAndSphereCache,
    device: torch.device,
) -> torch.Tensor | None:
    """Return (S, 4) query spheres; return None for varying meshless bboxes."""
    if meshes:
        return manager.get_query_spheres_for_meshes(meshes, obj=obj).to(device)
    if not bbox.is_batch_invariant():
        return None
    center = bbox.center[0].detach().cpu().numpy()
    extents = bbox.size[0].detach().cpu().numpy()
    box_mesh = trimesh.creation.box(extents=extents)
    box_mesh.apply_translation(center)
    return manager.get_query_spheres(box_mesh).to(device)


def _collision_bounds(
    meshes: tuple[trimesh.Trimesh, ...],
    fallback_bbox: AxisAlignedBoundingBox,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return local broadphase bounds covering the collision representation."""
    if not meshes:
        return fallback_bbox.min_point.expand(batch_size, 3), fallback_bbox.max_point.expand(batch_size, 3)

    component_bounds = np.stack([mesh.bounds for mesh in meshes])
    bbox_min = torch.tensor(component_bounds[:, 0].min(axis=0), dtype=torch.float32, device=device)
    bbox_max = torch.tensor(component_bounds[:, 1].max(axis=0), dtype=torch.float32, device=device)
    return bbox_min.expand(batch_size, 3), bbox_max.expand(batch_size, 3)


def _finalize_mesh_cache(entries: list[MeshPairEntry], device: torch.device) -> MeshPairCache | None:
    """Stack collected pair entries into a MeshPairCache; None when no pairs qualify."""
    if not entries:
        return None

    mesh_id_map: dict[int, int] = {}
    mesh_id_values: list[int] = []
    query_sphere_ids: list[int] = []
    mesh_idx_per_query: list[int] = []
    pair_slices: list[tuple[int, int]] = []
    offset = 0

    for entry in entries:
        n_spheres = entry.centers_local.shape[0]
        sphere_ids = list(range(offset, offset + n_spheres))
        for warp_mesh in entry.warp_meshes:
            mesh_key = id(warp_mesh)
            if mesh_key not in mesh_id_map:
                mesh_id_map[mesh_key] = len(mesh_id_values)
                mesh_id_values.append(warp_mesh.id)
            query_sphere_ids.extend(sphere_ids)
            mesh_idx_per_query.extend([mesh_id_map[mesh_key]] * n_spheres)
        pair_slices.append((offset, offset + n_spheres))
        offset += n_spheres

    pair_sphere_count = torch.tensor([e - s for s, e in pair_slices], dtype=torch.float32, device=device)
    sphere_pair_id = torch.repeat_interleave(torch.arange(len(pair_slices), device=device), pair_sphere_count.long())
    fixed_obstacle_positions = torch.stack([
        (
            entry.fixed_obstacle_pos
            if entry.fixed_obstacle_pos is not None
            else torch.zeros(3, dtype=torch.float32, device=device)
        )
        for entry in entries
    ])

    return MeshPairCache(
        all_centers_local=torch.cat([e.centers_local for e in entries], dim=0),
        all_radii=torch.cat([e.radii for e in entries], dim=0),
        pair_subject_objs=[e.subject for e in entries],
        pair_obstacle_objs=[e.obstacle for e in entries],
        pair_subject_applies_yaw=[e.subject_applies_yaw for e in entries],
        pair_obstacle_is_fixed=[e.obstacle_is_fixed for e in entries],
        pair_fixed_obstacle_pos=[e.fixed_obstacle_pos for e in entries],
        pair_fixed_obstacle_yaw=[e.fixed_obstacle_yaw for e in entries],
        pair_subject_bbox_min=torch.stack([e.subject_bbox_min for e in entries]),
        pair_subject_bbox_max=torch.stack([e.subject_bbox_max for e in entries]),
        pair_subject_bbox_includes_yaw=[e.subject_bbox_includes_yaw for e in entries],
        pair_obstacle_bbox_min=torch.stack([e.obstacle_bbox_min for e in entries]),
        pair_obstacle_bbox_max=torch.stack([e.obstacle_bbox_max for e in entries]),
        pair_obstacle_bbox_includes_yaw=[e.obstacle_bbox_includes_yaw for e in entries],
        pair_max_radius=torch.tensor([e.radii.max().item() for e in entries], device=device),
        sphere_pair_id=sphere_pair_id,
        query_sphere_id=torch.tensor(query_sphere_ids, dtype=torch.long, device=device),
        query_mesh_idx=torch.tensor(mesh_idx_per_query, dtype=torch.int32, device=device),
        pair_sphere_count=pair_sphere_count,
        mesh_id_array=wp.array(np.array(mesh_id_values, dtype=np.uint64), dtype=wp.uint64, device=str(device)),
        num_pairs=len(entries),
        total_spheres=offset,
        pair_fixed_obstacle_pos_tensor=fixed_obstacle_positions,
        pair_fixed_obstacle_yaw_tensor=torch.tensor(
            [entry.fixed_obstacle_yaw for entry in entries], dtype=torch.float32, device=device
        ),
        pair_subject_bbox_includes_yaw_tensor=torch.tensor(
            [entry.subject_bbox_includes_yaw for entry in entries], dtype=torch.bool, device=device
        ),
        pair_obstacle_bbox_includes_yaw_tensor=torch.tensor(
            [entry.obstacle_bbox_includes_yaw for entry in entries], dtype=torch.bool, device=device
        ),
    )


def _rotate_bbox_extents(
    bbox_min: torch.Tensor, bbox_max: torch.Tensor, yaws: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the AABB enclosing a Z-rotated bbox around the object origin."""
    min_x, min_y = bbox_min[:, 0], bbox_min[:, 1]
    max_x, max_y = bbox_max[:, 0], bbox_max[:, 1]
    corners_x = torch.stack([min_x, max_x, max_x, min_x], dim=1)
    corners_y = torch.stack([min_y, min_y, max_y, max_y], dim=1)
    cos_y = torch.cos(yaws).unsqueeze(1)
    sin_y = torch.sin(yaws).unsqueeze(1)
    rot_x = corners_x * cos_y - corners_y * sin_y
    rot_y = corners_x * sin_y + corners_y * cos_y
    rotated_min = torch.stack([rot_x.min(dim=1).values, rot_y.min(dim=1).values, bbox_min[:, 2]], dim=1)
    rotated_max = torch.stack([rot_x.max(dim=1).values, rot_y.max(dim=1).values, bbox_max[:, 2]], dim=1)
    return rotated_min, rotated_max


def _rotate_bbox_extents_batched(
    bbox_min: torch.Tensor, bbox_max: torch.Tensor, yaws: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate bounding-box extents shaped (B, P, 3) by yaws shaped (B, P)."""
    batch_size, num_pairs, _ = bbox_min.shape
    rotated_min, rotated_max = _rotate_bbox_extents(
        bbox_min.reshape(batch_size * num_pairs, 3),
        bbox_max.reshape(batch_size * num_pairs, 3),
        yaws.reshape(batch_size * num_pairs),
    )
    return (
        rotated_min.reshape(batch_size, num_pairs, 3),
        rotated_max.reshape(batch_size, num_pairs, 3),
    )
