# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Mesh-SDF loss for the ``In`` relation: keep a child inside a container's cavity proxy."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab_arena.relations.relation_solver_state import RelationSolverState
from isaaclab_arena.relations.relations import In
from isaaclab_arena.relations.warp_mesh_manager import WarpMeshAndSphereCache
from isaaclab_arena.relations.warp_sdf_kernels import clamp_sdf_sentinel, mesh_sdf
from isaaclab_arena.utils.pose import Pose, centers_in_target_frame, yaw_from_quat_xyzw

if TYPE_CHECKING:
    from isaaclab_arena.assets.object_base import ObjectBase


def bbox_corners_local(obj: ObjectBase, device: torch.device) -> torch.Tensor:
    """Return the 8 corners of obj's local axis-aligned bounding box, shape (8, 3)."""
    bbox = obj.get_bounding_box().to(device)
    lo = bbox.min_point[0]
    hi = bbox.max_point[0]
    corners = torch.stack([
        torch.stack([lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1], lo[2] if i & 4 else hi[2]])
        for i in range(8)
    ])
    return corners


def compute_in_loss_mesh(
    state: RelationSolverState,
    mesh_manager: WarpMeshAndSphereCache,
    orientations: list[dict[ObjectBase, float]] | None,
    slope: float,
    debug: bool,
) -> torch.Tensor:
    """Per-env loss pulling each ``In`` child's bounding-box corners inside its container cavity proxy.

    Uses the parent's authored cavity-proxy SDF (negative = inside): a corner outside the cavity, or
    within ``margin_m`` of the wall, contributes ``relu(sdf + margin_m)``. The child-parent no-overlap
    term is skipped elsewhere, so the outer shell does not push the child back out.

    Args:
        state: Solver state with positions and batch info.
        mesh_manager: Warp mesh cache providing the cavity proxy BVH.
        orientations: Per-env yaw (radians about Z) per object, or None.
        slope: Gradient magnitude for the containment loss.
        debug: Print per-relation loss when True.
    """
    device = state.device
    total_loss = torch.zeros(state.batch_size, device=device, dtype=torch.float32)

    for child in state.optimizable_objects:
        for relation in child.get_spatial_relations():
            if not isinstance(relation, In):
                continue
            parent = relation.parent
            cavity_mesh = mesh_manager.get_cavity_warp_mesh(parent)
            assert cavity_mesh is not None, (
                f"In parent '{parent.name}' has no usable cavity: provide an explicit get_cavity_mesh() "
                "proxy, or ensure its collision mesh can be capped into a watertight interior."
            )

            # The container may be a fixed anchor or itself optimizable (e.g. a bowl On a table). Use its
            # solved position either way; for an optimizable container, detach it so the child moves into
            # the container while the container is positioned by its own relations.
            parent_is_anchor = parent in state.anchor_objects
            parent_fixed_yaw = 0.0
            if parent_is_anchor:
                parent_pose = parent.get_initial_pose()
                assert isinstance(parent_pose, Pose)
                parent_fixed_yaw = yaw_from_quat_xyzw(parent_pose.rotation_xyzw)

            corners_local = bbox_corners_local(child, device)
            child_positions = state.get_position(child)
            parent_positions = state.get_position(parent)

            for b in range(state.batch_size):
                child_yaw = orientations[b].get(child, 0.0) if orientations is not None else 0.0
                if parent_is_anchor:
                    parent_pos_b = parent_positions[b]
                    parent_yaw = parent_fixed_yaw
                else:
                    parent_pos_b = parent_positions[b].detach()
                    parent_yaw = orientations[b].get(parent, 0.0) if orientations is not None else 0.0
                points = centers_in_target_frame(corners_local, child_yaw, parent_yaw, child_positions[b] - parent_pos_b)
                sdf = mesh_sdf(points, cavity_mesh)
                mesh_manager.warn_sdf_sentinel(sdf)
                sdf = clamp_sdf_sentinel(sdf)
                outside = torch.relu(sdf + relation.margin_m)
                total_loss[b] = total_loss[b] + slope * relation.relation_loss_weight * outside.mean()

            if debug:
                print(f"  [In] '{child.name}' in '{parent.name}': loss={total_loss.tolist()}")

    return total_loss
