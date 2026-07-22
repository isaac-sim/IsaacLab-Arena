# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sim-free IK feasibility utilities that operate on a CuroboPlanner instance."""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import isaaclab.utils.math as math_utils
from curobo.geom.types import Cuboid, WorldConfig

from isaaclab_arena.assets.object_base import ObjectBase
from isaaclab_arena.utils.device import resolve_cuda_device
from isaaclab_arena.utils.pose import Pose
from isaaclab_arena_curobo.curobo_frame_utils import top_down_grasp_matrix, world_pose_to_robot_frame

if TYPE_CHECKING:
    import logging

    from curobo.types.math import Pose as CuroboPose
    from curobo.wrap.reacher.ik_solver import IKSolver


class IKSolverContext(Protocol):
    """The host that owns a cuRobo IK solver plus the device/pose plumbing to drive it."""

    logger: logging.Logger

    def _to_curobo_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Move a tensor onto the cuRobo device/dtype."""

    def _make_pose(self, position: torch.Tensor, quaternion: torch.Tensor, *, quat_is_xyzw: bool = True) -> CuroboPose:
        """Build a cuRobo ``Pose`` on the cuRobo device."""


def resolve_ik_solver(ik_solver_context: IKSolverContext) -> IKSolver:
    """Get the cuRobo ``IKSolver`` from a host, whichever way it exposes it.

    ``CuroboIKSolver`` holds it as ``ik_solver``; the upstream ``CuroboPlanner`` (not ours to change)
    holds it as ``motion_gen.ik_solver``. Centralizing the lookup lets the solve take just the host.
    """
    ik_solver = getattr(ik_solver_context, "ik_solver", None)
    if ik_solver is None:
        ik_solver = ik_solver_context.motion_gen.ik_solver
    return ik_solver


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


def top_down_grasp_pose_from_world_poses(
    object_pos_w: tuple[float, float, float],
    object_quat_w_xyzw: tuple[float, float, float, float],
    robot_base_pos_w: tuple[float, float, float],
    robot_base_quat_w_xyzw: tuple[float, float, float, float],
    grasp_z_offset: float = 0.02,
    align_yaw_to_object: bool = True,
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Top-down grasp pose at an object's center, in the robot base frame, as a 4x4 transform."""
    dev = resolve_cuda_device(device)
    # Pure-math computation of the object pose in the robot base frame
    t_W_O = torch.tensor(object_pos_w, dtype=torch.float32, device=dev)
    q_W_O_xyzw = torch.tensor(object_quat_w_xyzw, dtype=torch.float32, device=dev)
    t_W_R = torch.tensor(robot_base_pos_w, dtype=torch.float32, device=dev)
    q_W_R_xyzw = torch.tensor(robot_base_quat_w_xyzw, dtype=torch.float32, device=dev)

    t_R_O, q_R_O_xyzw = world_pose_to_robot_frame(t_W_O, q_W_O_xyzw, t_W_R, q_W_R_xyzw)
    return top_down_grasp_matrix(t_R_O, q_R_O_xyzw, grasp_z_offset, align_yaw_to_object)


def solve_ik_feasibility(
    ik_solver_context: IKSolverContext,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
    require_collision_free: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched IK feasibility of all ``target_poses`` against a cuRobo IK solver. Shared between sim-free and env-coupled paths.
    Runs a single ``solve_batch`` for one layout.

    Args:
        ik_solver_context: The host that owns the solver and supplies device/pose plumbing -- a ``CuroboPlanner`` (env-coupled) or a ``CuroboIKSolver``.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad) to count as feasible.
        require_collision_free: Also require a collision-free joint solution (cuRobo ``success``), not
            just pose convergence. The caller must first exclude the grasped object's own contact --
            e.g. disable the hand-link spheres -- or the gripper over its target reads as a collision.

    Returns:
        ``(feasible, position_error, rotation_error)``, each length ``b`` and aligned with the input;
        errors are the best-seed values per pose.
    """
    ik_solver = resolve_ik_solver(ik_solver_context)
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

    ik_result = ik_solver.solve_batch(goal_pose, seed_config=ik_seed)

    num_poses = positions.shape[0]
    pos_err = ik_result.position_error.view(num_poses, -1)
    rot_err = ik_result.rotation_error.view(num_poses, -1)

    ok = (pos_err < position_threshold) & (rot_err < rotation_threshold)
    # TODO(xinjieyao, 2026-07-15): Support collision-free IK by disabling the hand-link spheres during collision checking
    assert require_collision_free is False, "Collision-free IK is not supported yet. Needs extra machinery."
    if require_collision_free:
        # cuRobo folds collision-free-ness into ``success`` (success = converged AND feasible).
        ok = ok & ik_result.success.view(num_poses, -1).bool()
    feasible = ok.any(dim=1)

    # Best-seed errors, reported for logging/return only (not part of the accept decision).
    best_idx = pos_err.argmin(dim=1, keepdim=True)
    best_pos_err = pos_err.gather(1, best_idx).squeeze(1)
    best_rot_err = rot_err.gather(1, best_idx).squeeze(1)

    ik_solver_context.logger.debug(f"Batch IK feasibility: {int(feasible.sum().item())}/{num_poses} feasible")
    return feasible, best_pos_err, best_rot_err


def check_ik_feasibility(
    ik_solver_context: IKSolverContext,
    target_poses: torch.Tensor,
    seed_config: torch.Tensor | None = None,
    position_threshold: float = 0.01,
    rotation_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched IK feasibility of all ``target_poses`` against a context's cuRobo IK solver.

    Serves both paths with the same math and thresholds; the only difference is where the solver comes
    from. Pass a ``CuroboIKSolver`` (build-time, exposes ``ik_solver``) or a ``CuroboPlanner``
    (env-coupled, exposes ``motion_gen.ik_solver``).

    Args:
        ik_solver_context: A ``CuroboIKSolver`` or ``CuroboPlanner`` supplying the pose plumbing and IK solver.
        target_poses: ``(b, 4, 4)`` end-effector goal transforms in the robot base frame.
        seed_config: Optional joint seed tensor.
        position_threshold: Max position error (m) to count as feasible.
        rotation_threshold: Max rotation error (rad) to count as feasible.

    Returns:
        ``(feasible, position_error, rotation_error)``, each length ``b`` and aligned with the input;
        errors are the best-seed values per pose.
    """
    return solve_ik_feasibility(
        ik_solver_context,
        target_poses,
        seed_config=seed_config,
        position_threshold=position_threshold,
        rotation_threshold=rotation_threshold,
        # TODO(xinjieyao, 2026-07-21): Support collision-free IK by disabling the hand-link spheres during collision checking
        require_collision_free=getattr(ik_solver_context, "require_collision_free", False),
    )
