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
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from isaaclab_arena.relations.collision_mode import CollisionMode, object_uses_mesh_collision
from isaaclab_arena.relations.mesh_pair_cache import MeshPairCache, MeshPairEntry
from isaaclab_arena.relations.placement_asset import PlaceableAsset
from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.warp_sdf_kernels import has_sdf_sentinel, multi_mesh_sdf
from isaaclab_arena.utils.bounding_box import OrientedBoundingBox
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from isaaclab_arena.relations.collision_object import CollisionObject
    from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache


def transform_points_between_frames(
    points_source: torch.Tensor,
    source_position: torch.Tensor,
    source_rotation_xyzw: torch.Tensor,
    target_position: torch.Tensor,
    target_rotation_xyzw: torch.Tensor,
) -> torch.Tensor:
    """Transform source-local points into a target-local frame using xyzw quaternions."""
    count = points_source.shape[0]
    source_rotation = source_rotation_xyzw.reshape(-1, 4)
    target_rotation = target_rotation_xyzw.reshape(-1, 4)
    if source_rotation.shape[0] == 1:
        source_rotation = source_rotation.expand(count, 4)
    if target_rotation.shape[0] == 1:
        target_rotation = target_rotation.expand(count, 4)
    assert source_rotation.shape[0] == count and target_rotation.shape[0] == count
    points_world = quat_apply(source_rotation, points_source) + source_position
    return quat_apply_inverse(target_rotation, points_world - target_position)


def _bbox_for_batch(bbox: OrientedBoundingBox, batch_size: int) -> OrientedBoundingBox:
    """Broadcast one asset-local box or return its complete candidate batch."""
    assert bbox.num_envs in (1, batch_size), f"Expected one box or B={batch_size}, got {bbox.num_envs}."
    return OrientedBoundingBox.from_tensors_unchecked(
        bbox.center.expand(batch_size, 3),
        bbox.half_extents.expand(batch_size, 3),
        bbox.rotation_xyzw.expand(batch_size, 4),
    )


def _concatenate_box_batches(boxes: list[OrientedBoundingBox]) -> OrientedBoundingBox:
    """Concatenate pair-major box batches without revalidating tensors."""
    return OrientedBoundingBox.from_tensors_unchecked(
        torch.cat([bbox.center for bbox in boxes], dim=0),
        torch.cat([bbox.half_extents for bbox in boxes], dim=0),
        torch.cat([bbox.rotation_xyzw for bbox in boxes], dim=0),
    )


def compute_no_overlap_loss_mesh(
    state: RelationSolverState,
    mesh_cache: MeshPairCache | None,
    mesh_manager: WarpMeshAndSphereCache,
    clearance_m: float,
    slope: float,
    debug: bool,
) -> torch.Tensor:
    """Batched per-env sphere-to-SDF penetration loss (one Warp launch for all candidates).

    Args:
        state: Current solver state with positions and batch info.
        mesh_cache: Precomputed collision pair data (None = no pairs).
        mesh_manager: Warp mesh/sphere cache associated with the pair cache.
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
    batch_size = state.batch_size
    position_cache = {obj: state.get_position(obj) for obj in state.optimizable_objects}
    rotation_cache = {obj: state.get_rotation(obj) for obj in state.optimizable_objects}

    subject_positions = torch.stack([position_cache[obj] for obj in mesh_cache.pair_subject_objs])
    subject_rotations = torch.stack([rotation_cache[obj] for obj in mesh_cache.pair_subject_objs])
    obstacle_position_batches: list[torch.Tensor] = []
    obstacle_rotation_batches: list[torch.Tensor] = []
    obstacle_box_batches: list[OrientedBoundingBox] = []
    for p, obstacle in enumerate(mesh_cache.pair_obstacle_objs):
        if mesh_cache.pair_obstacle_is_fixed[p]:
            fixed_position = mesh_cache.pair_fixed_obstacle_pos[p]
            fixed_rotation = mesh_cache.pair_fixed_obstacle_rotation[p]
            assert fixed_position is not None and fixed_rotation is not None
            obstacle_position_batches.append(fixed_position.expand(batch_size, 3))
            obstacle_rotation_batches.append(fixed_rotation.expand(batch_size, 4))
            obstacle_box_batches.append(_bbox_for_batch(state.get_fixed_obstacle_world_bbox(obstacle), batch_size))
            continue

        assert isinstance(obstacle, PlaceableAsset)
        position = position_cache[obstacle].detach()
        rotation = rotation_cache[obstacle].detach()
        obstacle_position_batches.append(position)
        obstacle_rotation_batches.append(rotation)
        obstacle_box_batches.append(
            _bbox_for_batch(state.get_base_bbox(obstacle), batch_size).transformed_unchecked(position, rotation)
        )
    obstacle_positions = torch.stack(obstacle_position_batches)
    obstacle_rotations = torch.stack(obstacle_rotation_batches)

    subject_boxes = _concatenate_box_batches(
        [_bbox_for_batch(state.get_base_bbox(obj), batch_size) for obj in mesh_cache.pair_subject_objs]
    ).transformed_unchecked(
        subject_positions.reshape(-1, 3),
        subject_rotations.reshape(-1, 4),
    )
    obstacle_boxes = _concatenate_box_batches(obstacle_box_batches)
    subject_min, subject_max = subject_boxes.get_axis_aligned_bounds()
    obstacle_min, obstacle_max = obstacle_boxes.get_axis_aligned_bounds()
    broadphase_margin = (mesh_cache.pair_max_radius.repeat_interleave(batch_size) + clearance_m).unsqueeze(1)
    separated = ((subject_min - broadphase_margin) > obstacle_max).any(dim=1) | (
        (obstacle_min - broadphase_margin) > subject_max
    ).any(dim=1)
    active_pair_env = ~separated
    if not active_pair_env.any():
        return total_loss

    environment_ids = torch.arange(batch_size, device=device, dtype=torch.long)
    sphere_pair_env_ids = (mesh_cache.sphere_pair_id.unsqueeze(1) * batch_size + environment_ids.unsqueeze(0)).reshape(
        -1
    )
    active_sphere_mask = active_pair_env[sphere_pair_env_ids]
    active_pair_env_ids = sphere_pair_env_ids[active_sphere_mask]
    active_centers = transform_points_between_frames(
        mesh_cache.all_centers_local.repeat_interleave(batch_size, dim=0)[active_sphere_mask],
        subject_positions.reshape(-1, 3)[active_pair_env_ids],
        subject_rotations.reshape(-1, 4)[active_pair_env_ids],
        obstacle_positions.reshape(-1, 3)[active_pair_env_ids],
        obstacle_rotations.reshape(-1, 4)[active_pair_env_ids],
    )
    active_radii = mesh_cache.all_radii.repeat_interleave(batch_size)[active_sphere_mask]
    active_mesh_idx = mesh_cache.sphere_mesh_idx.repeat_interleave(batch_size)[active_sphere_mask].contiguous()
    sdf_values = multi_mesh_sdf(
        active_centers,
        mesh_cache.mesh_id_array,
        wp.from_torch(active_mesh_idx, dtype=wp.int32),
    )
    assert not has_sdf_sentinel(
        sdf_values
    ), "MESH collision query could not resolve a target face; the promised collision geometry is unsupported."
    penetration = torch.relu(active_radii + clearance_m - sdf_values)

    pair_env_sum = torch.zeros(num_pairs * batch_size, device=device, dtype=penetration.dtype)
    pair_env_sum.index_add_(0, active_pair_env_ids, penetration)
    pair_env_count = mesh_cache.pair_sphere_count.repeat_interleave(batch_size)
    pair_env_mean = (pair_env_sum / pair_env_count).reshape(num_pairs, batch_size)
    total_loss = slope * pair_env_mean.sum(dim=0)

    if debug:
        print(f"  [NoOverlap MESH] total_loss={total_loss.tolist()}")

    return total_loss


def prepare_mesh_collision_cache(
    state: RelationSolverState,
    mesh_manager: WarpMeshAndSphereCache,
    on_pairs: set[tuple[int, int]],
    warned_no_mesh: set[str],
    default_collision_mode: CollisionMode = CollisionMode.MESH,
) -> MeshPairCache | None:
    """Precompute static per-pair mesh collision data.

    Args:
        state: Solver state with object info and batch size.
        mesh_manager: Warp mesh/sphere cache.
        on_pairs: Set of (id(a), id(b)) pairs linked by On relations (skipped).
        warned_no_mesh: Mutable set tracking which objects have already been warned about.
        default_collision_mode: Collision mode used by objects without a per-object override.

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
    )
    return _finalize_mesh_cache(all_pairs, device)


def _collect_mesh_pairs(
    state: RelationSolverState,
    manager: WarpMeshAndSphereCache,
    non_anchor_objects: list[PlaceableAsset],
    fixed_obstacles: list[PlaceableAsset | CollisionObject],
    on_pairs: set[tuple[int, int]],
    device: torch.device,
    warned_no_mesh: set[str],
    default_collision_mode: CollisionMode,
) -> list[MeshPairEntry]:
    """Collect all directed mesh pairs (forward + reverse)."""
    pairs: list[MeshPairEntry] = []

    for i, child in enumerate(non_anchor_objects):
        child_uses_mesh = object_uses_mesh_collision(child, default_collision_mode)
        child_mesh = manager.get_collision_mesh(child) if child_uses_mesh else None
        child_bbox = state.get_base_bbox(child).to(device)
        child_bbox_is_invariant = child_bbox.is_batch_invariant()
        if child_uses_mesh and child_mesh is None and child.name not in warned_no_mesh:
            warned_no_mesh.add(child.name)
            fallback = (
                "using an OBB-sphere approximation for mesh-obstacle pairs"
                if child_bbox_is_invariant
                else "pair will use OBB fallback for varying per-env bboxes"
            )
            print(f"[NoCollision] '{child.name}' has no collision mesh; {fallback}.")
        child_spheres = _get_subject_spheres(child_mesh, child_bbox, child, manager, device)

        for obstacle in fixed_obstacles:
            if (id(child), id(obstacle)) in on_pairs:
                continue
            obstacle_mesh = (
                manager.get_collision_mesh(obstacle)
                if object_uses_mesh_collision(obstacle, default_collision_mode)
                else None
            )
            if obstacle_mesh is None:
                if object_uses_mesh_collision(obstacle, default_collision_mode) and obstacle.name not in warned_no_mesh:
                    warned_no_mesh.add(obstacle.name)
                    print(f"[NoCollision] '{obstacle.name}' has no collision mesh; pair will use OBB fallback.")
                continue
            pose = obstacle.get_initial_pose()
            assert isinstance(
                pose, Pose
            ), f"MESH collision requires fixed obstacle '{obstacle.name}' to have a fixed Pose initial_pose"
            if child_spheres is None:
                continue
            pairs.append(
                MeshPairEntry(
                    subject=child,
                    obstacle=obstacle,
                    obstacle_is_fixed=True,
                    fixed_obstacle_pos=torch.tensor(pose.position_xyz, dtype=torch.float32, device=device),
                    fixed_obstacle_rotation=torch.tensor(pose.rotation_xyzw, dtype=torch.float32, device=device),
                    centers_local=child_spheres[:, :3],
                    radii=child_spheres[:, 3],
                    warp_mesh=manager.get_warp_mesh(obstacle_mesh, obj=obstacle),
                )
            )

        for j in range(i + 1, len(non_anchor_objects)):
            other = non_anchor_objects[j]
            if (id(child), id(other)) in on_pairs:
                continue
            other_uses_mesh = object_uses_mesh_collision(other, default_collision_mode)
            other_mesh = manager.get_collision_mesh(other) if other_uses_mesh else None
            other_bbox = state.get_base_bbox(other).to(device)
            other_bbox_is_invariant = other_bbox.is_batch_invariant()
            if other_mesh is None and child_mesh is None:
                if other_uses_mesh and other.name not in warned_no_mesh:
                    warned_no_mesh.add(other.name)
                    fallback = (
                        "using an OBB-sphere approximation for mesh-obstacle pairs"
                        if other_bbox_is_invariant
                        else "pair will use OBB fallback for varying per-env bboxes"
                    )
                    print(f"[NoCollision] '{other.name}' has no collision mesh; {fallback}.")
                continue
            child_target_mesh = child_mesh if child_mesh is not None else _bbox_proxy_mesh(child_bbox)
            other_target_mesh = other_mesh if other_mesh is not None else _bbox_proxy_mesh(other_bbox)
            if other_target_mesh is not None and child_spheres is not None:
                pairs.append(
                    MeshPairEntry(
                        subject=child,
                        obstacle=other,
                        obstacle_is_fixed=False,
                        fixed_obstacle_pos=None,
                        fixed_obstacle_rotation=None,
                        centers_local=child_spheres[:, :3],
                        radii=child_spheres[:, 3],
                        warp_mesh=manager.get_warp_mesh(
                            other_target_mesh,
                            obj=other if other_mesh is not None else None,
                        ),
                    )
                )

            if child_target_mesh is not None:
                other_spheres = _get_subject_spheres(other_mesh, other_bbox, other, manager, device)
                if other_spheres is None:
                    continue
                pairs.append(
                    MeshPairEntry(
                        subject=other,
                        obstacle=child,
                        obstacle_is_fixed=False,
                        fixed_obstacle_pos=None,
                        fixed_obstacle_rotation=None,
                        centers_local=other_spheres[:, :3],
                        radii=other_spheres[:, 3],
                        warp_mesh=manager.get_warp_mesh(
                            child_target_mesh,
                            obj=child if child_mesh is not None else None,
                        ),
                    )
                )

    return pairs


def _get_subject_spheres(
    mesh: trimesh.Trimesh | None,
    bbox: OrientedBoundingBox,
    obj: PlaceableAsset,
    manager: WarpMeshAndSphereCache,
    device: torch.device,
) -> torch.Tensor | None:
    """Return (S, 4) query spheres; return None for varying meshless bboxes."""
    if mesh is not None:
        return manager.get_query_spheres(mesh, obj=obj).to(device)
    box_mesh = _bbox_proxy_mesh(bbox)
    if box_mesh is None:
        return None
    return manager.get_query_spheres(box_mesh).to(device)


def _bbox_proxy_mesh(bbox: OrientedBoundingBox) -> trimesh.Trimesh | None:
    """Return one oriented box mesh for a batch-invariant OBB."""
    if not bbox.is_batch_invariant():
        return None
    center = bbox.center[0].detach().cpu().numpy()
    extents = (2.0 * bbox.half_extents[0]).detach().cpu().numpy()
    box_mesh = trimesh.creation.box(extents=extents)
    rotation_xyzw = bbox.rotation_xyzw[0].detach().cpu().numpy()
    transform = trimesh.transformations.quaternion_matrix(np.roll(rotation_xyzw, 1))
    transform[:3, 3] = center
    box_mesh.apply_transform(transform)
    return box_mesh


def _finalize_mesh_cache(entries: list[MeshPairEntry], device: torch.device) -> MeshPairCache | None:
    """Stack collected pair entries into a MeshPairCache; None when no pairs qualify."""
    if not entries:
        return None

    mesh_id_map: dict[int, int] = {}
    mesh_id_values: list[int] = []
    mesh_idx_per_sphere: list[int] = []
    for entry in entries:
        n_spheres = entry.centers_local.shape[0]
        mesh_key = id(entry.warp_mesh)
        if mesh_key not in mesh_id_map:
            mesh_id_map[mesh_key] = len(mesh_id_values)
            mesh_id_values.append(entry.warp_mesh.id)
        mesh_idx_per_sphere.extend([mesh_id_map[mesh_key]] * n_spheres)

    pair_sphere_count = torch.tensor(
        [entry.centers_local.shape[0] for entry in entries], dtype=torch.float32, device=device
    )
    sphere_pair_id = torch.repeat_interleave(torch.arange(len(entries), device=device), pair_sphere_count.long())

    return MeshPairCache(
        all_centers_local=torch.cat([e.centers_local for e in entries], dim=0),
        all_radii=torch.cat([e.radii for e in entries], dim=0),
        pair_subject_objs=[e.subject for e in entries],
        pair_obstacle_objs=[e.obstacle for e in entries],
        pair_obstacle_is_fixed=[e.obstacle_is_fixed for e in entries],
        pair_fixed_obstacle_pos=[e.fixed_obstacle_pos for e in entries],
        pair_fixed_obstacle_rotation=[e.fixed_obstacle_rotation for e in entries],
        pair_max_radius=torch.tensor([e.radii.max().item() for e in entries], device=device),
        sphere_pair_id=sphere_pair_id,
        sphere_mesh_idx=torch.tensor(mesh_idx_per_sphere, dtype=torch.int32, device=device),
        pair_sphere_count=pair_sphere_count,
        mesh_id_array=wp.array(np.array(mesh_id_values, dtype=np.uint64), dtype=wp.uint64, device=str(device)),
        num_pairs=len(entries),
        total_spheres=int(pair_sphere_count.sum().item()),
    )
