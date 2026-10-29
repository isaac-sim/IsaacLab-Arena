# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free IK feasibility utilities that operate on a CuroboIKSolver instance."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from curobo.scene import Cuboid, Scene
from curobo.types import JointState

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.utils.pose import Pose

if TYPE_CHECKING:
    from curobo.inverse_kinematics import InverseKinematics

    from isaaclab_arena_curobo.ik_solver import CuroboIKSolver


@dataclass
class AABBCollisionCuboid:
    """A collision obstacle described by an axis-aligned bounding box in the world frame.

    ``dims_xyz`` are full extents (edge lengths), matching cuRobo's ``Cuboid.dims``.
    """

    name: str
    dims_xyz: tuple[float, float, float]
    pose_W_O: Pose = field(default_factory=Pose.identity)


def get_aabb_collision_cuboid_for_object(
    obj: ObjectBase, pos_w: tuple[float, float, float], quat_w_xyzw: tuple[float, ...]
) -> AABBCollisionCuboid:
    """Axis-aligned bounding-box collision cuboid for an object at its layout pose (world frame).

    The bounding box is object-local, so its center offset is rotated by the object's world orientation
    and added to the root position -- placing e.g. a table box at its true mid-height rather than at the
    root.
    """
    bbox = obj.get_bounding_box()
    dims = tuple(float(v) for v in bbox.size[0].tolist())
    quat_t = torch.tensor(quat_w_xyzw, dtype=torch.float32)
    rotation = math_utils.matrix_from_quat(quat_t.unsqueeze(0))[0]
    center_world = torch.tensor(pos_w, dtype=torch.float32) + rotation @ bbox.center[0].to(torch.float32)
    return AABBCollisionCuboid(
        name=obj.name,
        dims_xyz=dims,
        pose_W_O=Pose(
            position_xyz=tuple(float(v) for v in center_world.tolist()),
            rotation_xyzw=tuple(float(v) for v in quat_w_xyzw),
        ),
    )


def cuboids_from_mesh(mesh: trimesh.Trimesh, name_prefix: str, voxel_size_m: float = 0.05) -> list[AABBCollisionCuboid]:
    """Cover a world-frame mesh's surface with world-axis-aligned cuboids built on a voxel grid.

    Points sampled on the surface mark the voxels it passes through, and runs of marked voxels are merged greedily
    (along x, then y, then z) into boxes, so a flat panel becomes a few cuboids however it is triangulated. The
    cuboids inflate the surface by up to one voxel, and closed volumes stay hollow.

    Args:
        mesh: Mesh in world coordinates.
        name_prefix: Prefix of the cuboid names, which are numbered per box.
        voxel_size_m: Voxel edge length; the coarser, the fewer cuboids and the more the surface is inflated.

    Returns:
        The merged cuboids.
    """
    # About four samples per voxel of surface area, so every voxel the surface crosses gets at least one.
    num_samples = int(np.ceil(4.0 * mesh.area / voxel_size_m**2))
    samples, _ = trimesh.sample.sample_surface(mesh, num_samples, seed=0)
    points = np.vstack([samples, mesh.vertices])
    origin = points.min(axis=0)
    voxel_indices = np.floor((points - origin) / voxel_size_m).astype(np.int64)
    occupied = np.zeros(voxel_indices.max(axis=0) + 1, dtype=bool)
    occupied[tuple(voxel_indices.T)] = True

    unmerged = occupied.copy()
    cuboids = []
    for i, j, k in np.argwhere(occupied):
        if unmerged[i, j, k]:
            i_end = i + 1
            while i_end < unmerged.shape[0] and unmerged[i_end, j, k]:
                i_end += 1
            j_end = j + 1
            while j_end < unmerged.shape[1] and unmerged[i:i_end, j_end, k].all():
                j_end += 1
            k_end = k + 1
            while k_end < unmerged.shape[2] and unmerged[i:i_end, j:j_end, k_end].all():
                k_end += 1
            unmerged[i:i_end, j:j_end, k:k_end] = False
            lower = origin + voxel_size_m * np.array([i, j, k])
            upper = origin + voxel_size_m * np.array([i_end, j_end, k_end])
            cuboids.append(
                AABBCollisionCuboid(
                    name=f"{name_prefix}_{len(cuboids)}",
                    dims_xyz=tuple(float(v) for v in upper - lower),
                    pose_W_O=Pose(position_xyz=tuple(float(v) for v in (lower + upper) / 2.0)),
                )
            )
    return cuboids


def scene_from_cuboids(
    cuboids: list[AABBCollisionCuboid],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
) -> Scene:
    """Build a cuRobo ``Scene`` of cuboids expressed in the robot base frame.

    Each obstacle's world pose is transformed into the robot base frame, all obstacles at once on the CPU. Include
    anchor objects (e.g. a table) here as static cuboids.
    """
    if not cuboids:
        return Scene(cuboid=[])
    # Frames: W = world, R = robot base, O = obstacle.
    t_W_O = torch.tensor([c.pose_W_O.position_xyz for c in cuboids], dtype=torch.float64)
    q_W_O_xyzw = torch.tensor([c.pose_W_O.rotation_xyzw for c in cuboids], dtype=torch.float64)
    t_W_R = torch.tensor(robot_base_pos_w, dtype=torch.float64)
    q_W_R_xyzw = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float64)
    R_R_W = math_utils.matrix_from_quat(q_W_R_xyzw).T
    t_R_O = (t_W_O - t_W_R) @ R_R_W.T
    q_R_O_xyzw = math_utils.quat_mul(math_utils.quat_inv(q_W_R_xyzw).expand_as(q_W_O_xyzw), q_W_O_xyzw)
    # cuRobo Cuboid pose is [x, y, z, qw, qx, qy, qz].
    poses = torch.cat([t_R_O, math_utils.convert_quat(q_R_O_xyzw, to="wxyz")], dim=1).tolist()
    return Scene(cuboid=[Cuboid(name=c.name, pose=pose, dims=list(c.dims_xyz)) for c, pose in zip(cuboids, poses)])


@dataclass
class IKFeasibility:
    """One batched IK solver's solved results."""

    feasible: torch.Tensor
    """Per-pose verdict: converged within the thresholds, and collision-free when that was required."""

    position_error: torch.Tensor
    """Per-pose IK position error (m) of the returned solution."""

    rotation_error: torch.Tensor
    """Per-pose IK rotation error (rad) of the returned solution."""

    joint_positions: torch.Tensor
    """Joint configuration solved per pose, of length joint_dim of the robot."""


def solve_ik_feasibility(
    solver: CuroboIKSolver,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.2,
    require_collision_free: bool = False,
) -> IKFeasibility:
    """Batched IK feasibility of all ``target_poses`` against a cuRobo IK solver, as a single solve for one layout.

    Args:
        solver: The ``CuroboIKSolver`` providing tensor device conversion and cuRobo pose construction.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad, geodesic angle) to count as feasible.
        require_collision_free: Also require a collision-free joint solution, not just pose convergence.

    Returns:
        An ``IKFeasibility`` object holding the per-pose verdict, plus the errors and joint configuration.
    """
    ik_solver = solver.ik_solver
    target_poses = solver._to_curobo_device(target_poses)
    positions, rotations = math_utils.unmake_pose(target_poses)
    goal_pose = solver._make_goal_tool_pose(
        solver._make_pose(
            position=positions,
            quaternion=math_utils.quat_from_matrix(rotations),  # xyzw
            quat_is_xyzw=True,
        )
    )

    ik_seed = None
    if seed_config is not None:
        ik_seed = solver._to_curobo_device(seed_config)
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)

    # The gripper is meant to touch what it grasps, so its own links are disabled to detect collisions with the grasped object.
    muted_links = list(solver.hand_link_names) if require_collision_free else []
    with _disabled_link_spheres(ik_solver, muted_links):
        ik_result = ik_solver.solve_pose(goal_pose, seed_config=ik_seed)

    num_poses = positions.shape[0]
    pos_err = ik_result.position_error.view(num_poses, -1)
    rot_err = ik_result.rotation_error.view(num_poses, -1)

    ok = (pos_err < position_threshold) & (rot_err < rotation_threshold)
    if require_collision_free:
        # cuRobo folds collision-free-ness into ``success`` (success = converged AND feasible).
        ok = ok & ik_result.success.view(num_poses, -1).bool()
    feasible = ok.any(dim=1)

    # rank among poses that passed the thresholds if exists, otherwise return the closest pose
    ranked_pos_err = torch.where(ok, pos_err, torch.full_like(pos_err, float("inf")))
    best_idx = torch.where(feasible.unsqueeze(1), ranked_pos_err, pos_err).argmin(dim=1, keepdim=True)
    best_pos_err = pos_err.gather(1, best_idx).squeeze(1)
    best_rot_err = rot_err.gather(1, best_idx).squeeze(1)
    solutions = ik_result.solution.view(num_poses, pos_err.shape[1], -1)
    best_solution = solutions.gather(1, best_idx.unsqueeze(-1).expand(-1, -1, solutions.shape[-1])).squeeze(1)

    solver.logger.debug(f"Batch IK feasibility: {int(feasible.sum().item())}/{num_poses} feasible")
    return IKFeasibility(feasible, best_pos_err, best_rot_err, best_solution)


def hand_sphere_mask(solver: CuroboIKSolver) -> torch.Tensor:
    """Mask over the robot's collision spheres selecting the hand links.

    Args:
        solver: The ``CuroboIKSolver`` that owns the cuRobo solver and names the embodiment's hand links.

    Returns:
        ``(num_spheres,)`` bool tensor aligned with the spheres ``robot_collision_spheres`` returns.
    """
    kinematics_config = solver.ik_solver.kinematics.kinematics_config
    sphere_link_indices = kinematics_config.link_sphere_idx_map
    mask = torch.zeros(sphere_link_indices.shape[0], dtype=torch.bool, device=sphere_link_indices.device)
    for name in solver.hand_link_names:
        mask[kinematics_config.get_sphere_index_from_link_name(name)] = True
    return mask


def robot_collision_spheres(solver: CuroboIKSolver, joint_positions: torch.Tensor) -> torch.Tensor:
    """Forward-kinematics the robot's collision spheres at each given joint configuration. Curobo collision-checks use these spheres.

    Args:
        solver: The ``CuroboIKSolver`` providing kinematics and tensor device conversion.
        joint_positions: shape of (b, dof), b:number of poses, dof:number of joints.

    Returns:
        shape of (b, num_spheres, 4), b:number of poses, 4:x, y, z, radius; cuRobo leaves unused
        spheres in with a negative radius.
    """
    kinematics = solver.ik_solver.kinematics
    joint_state = JointState.from_position(
        solver._to_curobo_device(joint_positions), joint_names=kinematics.joint_names
    )
    # cuRobo returns spheres per (batch, horizon); a single configuration per pose has a horizon of one.
    return kinematics.compute_kinematics(joint_state).robot_spheres[:, 0]


@contextmanager
def _disabled_link_spheres(ik_solver: InverseKinematics, link_names: Sequence[str]) -> Iterator[None]:
    """Mute the collision spheres of given links for the duration of the block, then restore them.

    Args:
        ik_solver: The cuRobo IK solver whose kinematics carry the spheres.
        link_names: Links to mute; an empty sequence makes the block a no-op.
    """
    kinematics_config = ik_solver.kinematics.kinematics_config
    for name in link_names:
        assert name in kinematics_config.link_name_to_idx_map, (
            f"Link '{name}' has no collision spheres in this robot config. "
            f"Known: {sorted(kinematics_config.link_name_to_idx_map)}."
        )
    # save the spheres to restore them later as all edits are in place
    saved_spheres = {name: kinematics_config.get_link_spheres(name).clone() for name in link_names}
    for name in link_names:
        kinematics_config.disable_link_spheres(name)
    try:
        yield
    # restore the spheres because they are edited in place on the solver's shared kinematics
    finally:
        for name, spheres in saved_spheres.items():
            kinematics_config.update_link_spheres(name, spheres)
