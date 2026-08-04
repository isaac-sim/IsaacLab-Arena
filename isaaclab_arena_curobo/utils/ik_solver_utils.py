# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free IK feasibility utilities that operate on a CuroboPlanner instance."""

from __future__ import annotations

import torch
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from curobo.geom.types import Cuboid, WorldConfig

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.utils.device import resolve_cuda_device
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_curobo.ik_solver import IKFeasibility
from isaaclab_arena_curobo.utils.frame_utils import world_pose_to_robot_frame

if TYPE_CHECKING:
    from curobo.wrap.reacher.ik_solver import IKSolver
    from isaaclab_mimic.motion_planners.curobo.curobo_planner import CuroboPlanner

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


def world_config_from_cuboids(
    cuboids: list[AABBCollisionCuboid],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    device: str | torch.device | None = None,
):
    """Build a cuRobo ``WorldConfig`` of cuboids expressed in the robot base frame.

    Each obstacle's world pose is transformed into the robot base frame. Include anchor objects (e.g. a table) here as static cuboids.
    """

    dev = resolve_cuda_device(device)
    robot_pos = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    robot_quat = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    curobo_cuboids = []
    for c in cuboids:
        pos_w = torch.tensor(c.pose_W_O.position_xyz, dtype=torch.float32, device=dev)
        quat_w_xyzw = torch.tensor(c.pose_W_O.rotation_xyzw, dtype=torch.float32, device=dev)
        t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(pos_w, quat_w_xyzw, robot_pos, robot_quat)
        q_R_O_wxyz = math_utils.convert_quat(q_R_O_xyzw, to="wxyz")
        # cuRobo Cuboid pose is [x, y, z, qw, qx, qy, qz].
        pose = t_R_O.tolist() + q_R_O_wxyz.tolist()
        curobo_cuboids.append(Cuboid(name=c.name, pose=pose, dims=list(c.dims_xyz)))
    return WorldConfig(cuboid=curobo_cuboids)


def solve_ik_feasibility(
    ik_solver_context: CuroboIKSolver | CuroboPlanner,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
    require_collision_free: bool = False,
) -> IKFeasibility:
    """Batched IK feasibility of all ``target_poses`` against a cuRobo IK solver. Shared between sim-free and env-coupled paths.
    Runs a single ``solve_batch`` for one layout.

    Args:
        ik_solver_context: The host that owns the solver and supplies device/pose plumbing -- a ``CuroboPlanner`` (env-coupled) or a ``CuroboIKSolver``.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad) to count as feasible.
        require_collision_free: Also require a collision-free joint solution, not just pose convergence.

    Returns:
        An ``IKFeasibility`` holding the per-pose verdict, position/rotation errors, and the joint
        configuration solved per pose.
    """
    ik_solver = _resolve_ik_solver(ik_solver_context)
    target_poses = ik_solver_context._to_curobo_device(target_poses)
    positions, rotations = math_utils.unmake_pose(target_poses)
    goal_pose = ik_solver_context._make_pose(
        position=positions,
        quaternion=math_utils.quat_from_matrix(rotations),  # xyzw
        quat_is_xyzw=True,
    )

    ik_seed = None
    if seed_config is not None:
        ik_seed = ik_solver_context._to_curobo_device(seed_config)
        while ik_seed.dim() < 3:
            ik_seed = ik_seed.unsqueeze(0)

    # The gripper is meant to touch what it grasps, so its own links are disabled to detect collisions with the grasped object.
    muted_links = _resolve_hand_link_names(ik_solver_context) if require_collision_free else []
    with _disabled_link_spheres(ik_solver, muted_links):
        ik_result = ik_solver.solve_batch(goal_pose, seed_config=ik_seed)

    num_poses = positions.shape[0]
    pos_err = ik_result.position_error.view(num_poses, -1)
    rot_err = ik_result.rotation_error.view(num_poses, -1)

    ok = (pos_err < position_threshold) & (rot_err < rotation_threshold)
    if require_collision_free:
        # cuRobo folds collision-free-ness into ``success`` (success = converged AND feasible).
        ok = ok & ik_result.success.view(num_poses, -1).bool()
    feasible = ok.any(dim=1)

    # Best-seed errors, reported for logging/return only (not part of the accept decision).
    best_idx = pos_err.argmin(dim=1, keepdim=True)
    best_pos_err = pos_err.gather(1, best_idx).squeeze(1)
    best_rot_err = rot_err.gather(1, best_idx).squeeze(1)
    solutions = ik_result.solution.view(num_poses, pos_err.shape[1], -1)
    best_solution = solutions.gather(1, best_idx.unsqueeze(-1).expand(-1, -1, solutions.shape[-1])).squeeze(1)

    ik_solver_context.logger.debug(f"Batch IK feasibility: {int(feasible.sum().item())}/{num_poses} feasible")
    return IKFeasibility(feasible, best_pos_err, best_rot_err, best_solution)


def hand_sphere_mask(ik_solver_context: CuroboIKSolver | CuroboPlanner) -> torch.Tensor:
    """Mask over the robot's collision spheres selecting the hand links.

    Args:
        ik_solver_context: The host that owns the solver and names the embodiment's hand links.

    Returns:
        ``(num_spheres,)`` bool tensor aligned with the spheres ``robot_collision_spheres`` returns.
    """
    kinematics_config = _resolve_ik_solver(ik_solver_context).kinematics.kinematics_config
    sphere_link_indices = kinematics_config.link_sphere_idx_map
    mask = torch.zeros(sphere_link_indices.shape[0], dtype=torch.bool, device=sphere_link_indices.device)
    for name in _resolve_hand_link_names(ik_solver_context):
        mask[kinematics_config.get_sphere_index_from_link_name(name)] = True
    return mask


def robot_collision_spheres(
    ik_solver_context: CuroboIKSolver | CuroboPlanner, joint_positions: torch.Tensor
) -> torch.Tensor:
    """Forward-kinematics the robot's collision spheres at each given joint configuration. Curobo collision-checks use these spheres.

    Args:
        ik_solver_context: The host that owns the solver and supplies device plumbing.
        joint_positions: shape of (b, dof), b:number of poses, dof:number of joints.

    Returns:
        shape of (b, num_spheres, 4), b:number of poses, 4:x, y, z, radius; cuRobo leaves unused
        spheres in with a negative radius.
    """
    ik_solver = _resolve_ik_solver(ik_solver_context)
    joint_positions = ik_solver_context._to_curobo_device(joint_positions)
    return ik_solver.kinematics.get_state(joint_positions).link_spheres_tensor


def _resolve_ik_solver(ik_solver_context: CuroboIKSolver | CuroboPlanner) -> IKSolver:
    """Get the cuRobo ``IKSolver`` from a host, whichever way it exposes it.

    ``CuroboIKSolver`` holds it as ``ik_solver``; the upstream ``CuroboPlanner`` (not ours to change)
    holds it as ``motion_gen.ik_solver``. Centralizing the lookup lets the solve take just the host.
    """
    ik_solver = getattr(ik_solver_context, "ik_solver", None)
    if ik_solver is None:
        ik_solver = ik_solver_context.motion_gen.ik_solver
    return ik_solver


def _resolve_hand_link_names(ik_solver_context: CuroboIKSolver | CuroboPlanner) -> list[str]:
    """Get the embodiment's hand link names from a host, whichever way it exposes them."""
    hand_link_names = getattr(ik_solver_context, "hand_link_names", None)
    if hand_link_names is None:
        # CuroboPlanner holds them on its config.
        hand_link_names = ik_solver_context.config.hand_link_names
    return list(hand_link_names)


@contextmanager
def _disabled_link_spheres(ik_solver: IKSolver, link_names: Sequence[str]) -> Iterator[None]:
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
