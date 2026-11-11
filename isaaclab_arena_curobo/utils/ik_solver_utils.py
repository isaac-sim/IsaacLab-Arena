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
from curobo.scene import Cuboid, Mesh, Scene
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


def crop_mesh_near_point(mesh_W: trimesh.Trimesh, t_W: Sequence[float], radius_m: float) -> trimesh.Trimesh:
    """Return the triangles of a world-frame mesh whose bounding boxes come within ``radius_m`` of a point.

    Triangles are kept or dropped whole, so the crop covers every surface within the radius and keeps large
    triangles (e.g. a floor) that only pass through it. Returns an empty mesh when nothing is that close.

    Args:
        mesh_W: Mesh in world coordinates.
        t_W: Center of the crop, in the world frame.
        radius_m: Crop radius (m).
    """
    # (num_faces, 3 corners, xyz)
    triangles_W = mesh_W.triangles
    # Each triangle's bounding box: per-axis min and max over its three corners.
    # Element-wise over the three corners; a min/max reduction along axis 1 is ~10x slower on kitchen meshes.
    lower_W = np.minimum(np.minimum(triangles_W[:, 0], triangles_W[:, 1]), triangles_W[:, 2])
    upper_W = np.maximum(np.maximum(triangles_W[:, 0], triangles_W[:, 1]), triangles_W[:, 2])
    point_W = np.asarray(t_W, dtype=np.float64)
    # Point-to-box distance: per axis, how far the point lies outside the box (0 when inside that axis's range).
    distances = np.linalg.norm(np.maximum(np.maximum(lower_W - point_W, point_W - upper_W), 0.0), axis=1)
    # Keep whole triangles whose box comes within the radius.
    kept_faces = mesh_W.faces[distances < radius_m]
    # Keep only the vertices the kept faces use, and renumber the faces to index into that smaller vertex list.
    kept_vertex_ids, faces = np.unique(kept_faces, return_inverse=True)
    return trimesh.Trimesh(vertices=mesh_W.vertices[kept_vertex_ids], faces=faces.reshape(-1, 3), process=False)


def curobo_mesh_in_robot_frame(
    mesh_W: trimesh.Trimesh,
    name: str,
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
) -> Mesh:
    """Return a world-frame mesh as a cuRobo ``Mesh`` obstacle with its vertices in the robot base frame.

    The vertices are moved rather than the pose set, so the obstacle sits at the identity pose.
    """
    # Frames: W = world, R = robot base.
    t_W_R = np.asarray(robot_base_pos_w, dtype=np.float64)
    R_W_R = math_utils.matrix_from_quat(torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float64)).numpy()
    vertices_R = (np.asarray(mesh_W.vertices) - t_W_R) @ R_W_R
    return Mesh(
        name=name,
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        vertices=vertices_R.astype(np.float32),
        faces=np.asarray(mesh_W.faces, dtype=np.int32),
    )


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
    # Stack every obstacle's world pose into (N, 3) positions and (N, 4) quaternions.
    t_W_O = torch.tensor([c.pose_W_O.position_xyz for c in cuboids], dtype=torch.float64)
    q_W_O_xyzw = torch.tensor([c.pose_W_O.rotation_xyzw for c in cuboids], dtype=torch.float64)
    t_W_R = torch.tensor(robot_base_pos_w, dtype=torch.float64)
    q_W_R_xyzw = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float64)
    # T_R_O = T_W_R^-1 * T_W_O. Position: offset from the base, rotated into the base frame.
    R_R_W = math_utils.matrix_from_quat(q_W_R_xyzw).T
    t_R_O = (t_W_O - t_W_R) @ R_R_W.T
    # Orientation: undo the base rotation.
    q_R_O_xyzw = math_utils.quat_mul(math_utils.quat_inv(q_W_R_xyzw).expand_as(q_W_O_xyzw), q_W_O_xyzw)
    # cuRobo Cuboid pose is [x, y, z, qw, qx, qy, qz].
    poses = torch.cat([t_R_O, math_utils.convert_quat(q_R_O_xyzw, to="wxyz")], dim=1).tolist()
    # Cuboid dims are the full side lengths of the object's bounding box; the pose sits at its center.
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
